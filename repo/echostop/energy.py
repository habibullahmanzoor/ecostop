"""GPU energy measurement (plan §18).

Samples instantaneous GPU power draw via NVML in a background thread and
integrates it (trapezoidal rule) to get energy in joules. Exposes per-epoch
energy `e_t`, per-epoch runtime `r_t`, and mean power `P_bar_t = e_t / r_t` —
the incremental quantities RQ0 needs (plan §2, §12), not just cumulative
totals.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import pynvml


@dataclass
class EpochEnergy:
    epoch: int
    energy_joules: float
    duration_seconds: float
    mean_power_watts: float
    n_samples: int


class EnergyMeter:
    """Background NVML power sampler with epoch-boundary energy accounting.

    Usage:
        meter = EnergyMeter(device_index=0, sample_interval_s=0.02)
        meter.start()
        meter.begin_epoch(epoch=0)
        ... train one epoch ...
        rec = meter.end_epoch()   # EpochEnergy for that epoch
        ...
        meter.stop()
    """

    def __init__(self, device_index: int = 0, sample_interval_s: float = 0.02):
        self.device_index = device_index
        self.sample_interval_s = sample_interval_s

        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self._name = pynvml.nvmlDeviceGetName(self._handle)
        if isinstance(self._name, bytes):
            self._name = self._name.decode("utf-8")

        self._lock = threading.Lock()
        # (timestamp_s, power_w) samples since the last epoch boundary
        self._samples: list[tuple[float, float]] = []
        self._epoch_start_time: float | None = None
        self._current_epoch: int | None = None

        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_flag.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass

    def __enter__(self) -> "EnergyMeter":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- polling ---------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop_flag.is_set():
            t = time.monotonic()
            try:
                milliwatts = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                watts = milliwatts / 1000.0
            except pynvml.NVMLError:
                watts = float("nan")
            with self._lock:
                self._samples.append((t, watts))
            time.sleep(self.sample_interval_s)

    # -- epoch accounting -------------------------------------------------

    def begin_epoch(self, epoch: int) -> None:
        with self._lock:
            self._samples.clear()
            self._epoch_start_time = time.monotonic()
            self._current_epoch = epoch
            # seed with one sample so a very short epoch still integrates
            try:
                milliwatts = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                watts = milliwatts / 1000.0
            except pynvml.NVMLError:
                watts = float("nan")
            self._samples.append((self._epoch_start_time, watts))

    def end_epoch(self) -> EpochEnergy:
        end_time = time.monotonic()
        with self._lock:
            samples = list(self._samples)
            epoch = self._current_epoch
            start_time = self._epoch_start_time

        if epoch is None or start_time is None:
            raise RuntimeError("end_epoch() called without a matching begin_epoch()")

        # append a final sample at the epoch boundary for a clean integral
        try:
            milliwatts = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            watts = milliwatts / 1000.0
        except pynvml.NVMLError:
            watts = samples[-1][1] if samples else float("nan")
        samples.append((end_time, watts))

        energy_j = _trapezoidal_energy(samples)
        duration_s = end_time - start_time
        mean_power_w = energy_j / duration_s if duration_s > 0 else float("nan")

        return EpochEnergy(
            epoch=epoch,
            energy_joules=energy_j,
            duration_seconds=duration_s,
            mean_power_watts=mean_power_w,
            n_samples=len(samples),
        )

    # -- diagnostics -------------------------------------------------------

    def idle_power_watts(self, sample_seconds: float = 2.0) -> float:
        """Sample power at rest to establish the idle baseline (plan §18)."""
        readings = []
        t_end = time.monotonic() + sample_seconds
        while time.monotonic() < t_end:
            try:
                readings.append(pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0)
            except pynvml.NVMLError:
                pass
            time.sleep(self.sample_interval_s)
        return sum(readings) / len(readings) if readings else float("nan")

    @property
    def device_name(self) -> str:
        return self._name


def _trapezoidal_energy(samples: list[tuple[float, float]]) -> float:
    """Integrate power-vs-time samples (trapezoidal rule) to get joules."""
    if len(samples) < 2:
        return 0.0
    energy = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        energy += 0.5 * (p0 + p1) * dt
    return energy
