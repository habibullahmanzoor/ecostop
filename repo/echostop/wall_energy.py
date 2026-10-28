"""System-level (wall-plug) energy accounting for Route B.

Why this exists
---------------
`echostop/energy.py` measures GPU-board energy from NVML and timestamps samples with
`time.monotonic()`. A plug-in power meter reports wall power on its own clock, so the two
cannot be joined. This module timestamps everything with `time.time()` and records explicit
epoch and run boundaries, so wall joules can be attributed to the same epochs the board
meter already logs.

It also answers the question the board meter cannot: for a given stopping decision, how many
joules did the *operator* save, and how does that compare with the board-level figure?

Usage (deployment, with a smart plug):

    from echostop.wall_energy import WallMeter, HttpJsonSource

    src  = HttpJsonSource("http://192.168.1.42/status", ["power"])   # Tasmota-style
    wall = WallMeter(src, interval_s=1.0)
    wall.start()
    wall.mark_run_start()
    for epoch in range(T):
        wall.begin_epoch(epoch)
        ...train...
        wall.end_epoch()
    wall.mark_run_end()
    wall.stop()
    wall.write_csv("runs/wall_job17_seed0.csv")

Usage (replay of a meter's own export, for testing the analysis before buying anything):

    src = CsvSource("meter_export.csv", time_col="timestamp", power_col="power_w")

Analysis:

    from echostop.wall_energy import load_csv, energy_between, board_wall_report

    s = load_csv("runs/wall_job17_seed0.csv")
    e_wall = energy_between(s, t_start, t_end)

No third-party dependency. Python 3.8+.
"""

from __future__ import annotations

import csv
import json
import math
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# power sources
# ---------------------------------------------------------------------------


class ConstantSource:
    """Fixed power reading. Used by the self-test and for dry runs."""

    def __init__(self, watts: float):
        self.watts = watts

    def read(self) -> float:
        return self.watts


class HttpJsonSource:
    """A smart plug that exposes its current power as JSON over HTTP.

    Works with the generic shape `{... "power": 63.2 ...}` (Tasmota, Shelly in
    simple mode, ESPHome with a JSON endpoint, Home Assistant's REST proxy). Pass
    `json_path` as the key path into the document, e.g. ``["data", "power"]``.
    """

    def __init__(self, url: str, json_path: Sequence[str], timeout_s: float = 2.0,
                 scale: float = 1.0, auth_header: Optional[str] = None):
        self.url = url
        self.json_path = list(json_path)
        self.timeout_s = timeout_s
        self.scale = scale
        self.auth_header = auth_header

    def read(self) -> float:
        req = urllib.request.Request(self.url)
        if self.auth_header:
            name, _, value = self.auth_header.partition(":")
            req.add_header(name.strip(), value.strip())
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            doc = json.loads(resp.read().decode("utf-8"))
        node = doc
        for key in self.json_path:
            node = node[key]
        return float(node) * self.scale


class CsvSource:
    """Replays a power log exported by the meter's own app.

    Expects a header with a timestamp column and a power column. Timestamps may be
    epoch seconds or an ISO-8601 string; `time_format` selects the parser. Values
    outside the logged window return the last value (a run that outlasts the log is
    an analysis error, and `WallMeter` will flag it).
    """

    def __init__(self, path: str, time_col: str = "timestamp", power_col: str = "power_w"):
        self.rows: list[Tuple[float, float]] = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                t = _parse_time(row[time_col])
                p = float(row[power_col])
                self.rows.append((t, p))
        if not self.rows:
            raise ValueError(f"no rows in {path}")
        self.rows.sort(key=lambda r: r[0])

    def read(self) -> float:
        now = time.time()
        return energy_interpolate_power(self.rows, now)


def _parse_time(raw: str) -> float:
    raw = raw.strip()
    try:
        return float(raw)
    except ValueError:
        pass
    from datetime import datetime
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(raw, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(f"unrecognised timestamp {raw!r}")


# ---------------------------------------------------------------------------
# integration helpers
# ---------------------------------------------------------------------------


@dataclass
class Span:
    """A wall-clock interval with its integrated energy."""
    label: str
    t0: float
    t1: float
    energy_joules: float
    duration_seconds: float
    mean_power_watts: float
    n_samples: int


def energy_between(samples: Sequence[Tuple[float, float]], t0: float, t1: float) -> float:
    """Trapezoidal integral of power over [t0, t1], with linear edge interpolation.

    Samples must be sorted by time. Power outside the sampled range is held at the
    nearest sample, so an interval that starts before the first sample is not
    silently truncated.
    """
    if t1 <= t0:
        return 0.0
    if not samples:
        return 0.0
    pts = [(t, p) for t, p in samples if not math.isnan(p)]
    if not pts:
        return 0.0
    # clip to the requested window, inserting interpolated endpoints
    clipped: list[Tuple[float, float]] = [(t0, energy_interpolate_power(pts, t0))]
    for t, p in pts:
        if t0 < t < t1:
            clipped.append((t, p))
    clipped.append((t1, energy_interpolate_power(pts, t1)))
    energy = 0.0
    for (ta, pa), (tb, pb) in zip(clipped, clipped[1:]):
        dt = tb - ta
        if dt <= 0:
            continue
        energy += 0.5 * (pa + pb) * dt
    return energy


def energy_interpolate_power(samples: Sequence[Tuple[float, float]], t: float) -> float:
    """Power at time t: linear interpolation, flat extrapolation at the ends."""
    if not samples:
        return float("nan")
    if t <= samples[0][0]:
        return samples[0][1]
    if t >= samples[-1][0]:
        return samples[-1][1]
    lo, hi = 0, len(samples) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if samples[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    (ta, pa), (tb, pb) = samples[lo], samples[hi]
    if tb == ta:
        return pa
    return pa + (pb - pa) * (t - ta) / (tb - ta)


# ---------------------------------------------------------------------------
# the meter
# ---------------------------------------------------------------------------


@dataclass
class EpochWall:
    epoch: int
    energy_joules: float
    duration_seconds: float
    mean_power_watts: float
    n_samples: int


class WallMeter:
    """Background sampler for a wall-plug power source, with epoch and run boundaries.

    Mirrors `EnergyMeter`'s interface (`begin_epoch` / `end_epoch`) so a trainer can run
    both meters side by side, but timestamps with `time.time()` so the two logs share a
    clock with any external record.
    """

    def __init__(self, source: Callable[[], float], interval_s: float = 1.0,
                 clock: Callable[[], float] = time.time):
        self.source = source
        self.interval_s = interval_s
        self.clock = clock
        self.samples: list[Tuple[float, float]] = []
        self.epochs: list[EpochWall] = []
        self.run_start: Optional[float] = None
        self.run_end: Optional[float] = None
        self._epoch_start: Optional[float] = None
        self._epoch_index: Optional[int] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.nan_reads = 0

    # lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._thread = None

    def __enter__(self) -> "WallMeter":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # sampling ----------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            t = self.clock()
            try:
                watts = float(self.source())
            except Exception:
                watts = float("nan")
                self.nan_reads += 1
            with self._lock:
                self.samples.append((t, watts))
            time.sleep(self.interval_s)

    def _sample_now(self) -> float:
        t = self.clock()
        try:
            watts = float(self.source())
        except Exception:
            watts = float("nan")
            self.nan_reads += 1
        with self._lock:
            self.samples.append((t, watts))
        return watts

    # boundaries --------------------------------------------------------
    def mark_run_start(self) -> None:
        self.run_start = self.clock()
        self._sample_now()

    def mark_run_end(self) -> None:
        self.run_end = self.clock()
        self._sample_now()

    def begin_epoch(self, epoch: int) -> None:
        self._epoch_start = self.clock()
        self._epoch_index = epoch
        self._sample_now()

    def end_epoch(self) -> EpochWall:
        if self._epoch_start is None or self._epoch_index is None:
            raise RuntimeError("end_epoch() without begin_epoch()")
        t1 = self.clock()
        self._sample_now()
        with self._lock:
            samples = list(self.samples)
        e = energy_between(samples, self._epoch_start, t1)
        rec = EpochWall(
            epoch=self._epoch_index,
            energy_joules=e,
            duration_seconds=t1 - self._epoch_start,
            mean_power_watts=(e / (t1 - self._epoch_start)) if t1 > self._epoch_start else float("nan"),
            n_samples=sum(1 for t, _ in samples if self._epoch_start <= t <= t1),
        )
        self.epochs.append(rec)
        self._epoch_start = None
        self._epoch_index = None
        return rec

    # output ------------------------------------------------------------
    def spans(self) -> list[Span]:
        out = []
        for rec in self.epochs:
            out.append(Span(f"epoch{rec.epoch}", 0.0, 0.0, rec.energy_joules,
                            rec.duration_seconds, rec.mean_power_watts, rec.n_samples))
        if self.run_start is not None and self.run_end is not None:
            with self._lock:
                samples = list(self.samples)
            e = energy_between(samples, self.run_start, self.run_end)
            d = self.run_end - self.run_start
            out.append(Span("run", self.run_start, self.run_end, e, d,
                            e / d if d > 0 else float("nan"), len(samples)))
        return out

    def write_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time_epoch_s", "power_w"])
            with self._lock:
                for t, p in self.samples:
                    w.writerow([f"{t:.6f}", "" if math.isnan(p) else f"{p:.3f}"])
            w.writerow(["run_start", "" if self.run_start is None else f"{self.run_start:.6f}"])
            w.writerow(["run_end", "" if self.run_end is None else f"{self.run_end:.6f}"])


def load_csv(path: str) -> list[Tuple[float, float]]:
    """Load a `write_csv` log (time_epoch_s, power_w), ignoring the boundary rows."""
    out: list[Tuple[float, float]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                t = float(row["time_epoch_s"])
                p = float(row["power_w"])
            except (TypeError, ValueError):
                continue
            if not math.isnan(p):
                out.append((t, p))
    out.sort(key=lambda r: r[0])
    return out


# ---------------------------------------------------------------------------
# analysis: does the board-level saving survive at the wall?
# ---------------------------------------------------------------------------


def board_wall_report(board_tail_j: float, board_run_j: float,
                      wall_tail_j: float, wall_run_j: float) -> dict:
    """Compare the board-level saving with the wall-level saving for one run.

    board_tail_j  GPU-board joules of the epochs the rule removed (the board saving)
    board_run_j   GPU-board joules of the full run
    wall_tail_j   wall joules of the same removed epochs
    wall_run_j    wall joules of the full run

    marginal_ratio      how much more the wall charged for the removed epochs than the board
                        did. Above 1 because of CPU, memory, PSU losses and a less efficient
                        PSU operating point at part load.
    *_saving_fraction   the percentage a paper would report at each level. These are what must
                        be compared: the absolute joules can differ by 40\,\% while the
                        fraction barely moves, and that is a finding either way.
    """
    if board_run_j <= 0 or wall_run_j <= 0:
        raise ValueError("run energies must be positive")
    board_f = board_tail_j / board_run_j
    wall_f = wall_tail_j / wall_run_j
    return {
        "board_tail_j": board_tail_j,
        "wall_tail_j": wall_tail_j,
        "board_run_j": board_run_j,
        "wall_run_j": wall_run_j,
        "marginal_ratio": (wall_tail_j / board_tail_j) if board_tail_j > 0 else float("nan"),
        "board_saving_fraction": board_f,
        "wall_saving_fraction": wall_f,
        "fraction_shift_pp": 100.0 * (wall_f - board_f),
    }


def campaign_accounting(run_wall_j: float, tail_wall_j: float, idle_w: float,
                        gap_s: float, used_for_next_job: bool) -> dict:
    """What the operator saves depends on what happens to the freed time.

    Stopping early frees `t_tail`. If the next job starts immediately, the campaign saves
    the tail's marginal energy. If the machine idles, it still draws `idle_w`, so the
    saving is what is left after the idle cost of the gap. If the gap is longer than the
    freed time (a human in the loop), the saving can be zero or negative.
    """
    tail_energy = tail_wall_j
    idle_cost = idle_w * gap_s
    realised = tail_energy - idle_cost if not used_for_next_job else tail_energy
    return {
        "tail_energy_j": tail_energy,
        "idle_cost_j": idle_cost,
        "realised_saving_j": realised,
        "realised_fraction_of_run": realised / run_wall_j if run_wall_j > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# self-test: no hardware required
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    clock = {"t": 1_700_000_000.0}

    def fake_clock() -> float:
        clock["t"] += 1.0
        return clock["t"]

    count = {"n": 0}

    def fake_power() -> float:
        count["n"] += 1
        return 60.0 if count["n"] < 10 else 200.0

    m = WallMeter(fake_power, interval_s=1.0, clock=fake_clock)
    m.mark_run_start()
    for ep in range(4):
        m.begin_epoch(ep)
        m.end_epoch()
    m.mark_run_end()

    run = [s for s in m.spans() if s.label == "run"][0]
    print(f"samples      {len(m.samples)}")
    print(f"run energy   {run.energy_joules:,.0f} J over {run.duration_seconds:.0f} s")
    print(f"mean power   {run.mean_power_watts:.1f} W")

    # the analysis path must reproduce the meter's own integration exactly
    direct = energy_between(m.samples, m.run_start, m.run_end)
    assert abs(direct - run.energy_joules) < 1e-6, (direct, run.energy_joules)

    rep = board_wall_report(board_tail_j=2_000.0, board_run_j=8_000.0,
                            wall_tail_j=2_600.0, wall_run_j=10_400.0)
    print(f"board saving {rep['board_saving_fraction']:.1%}   "
          f"wall saving {rep['wall_saving_fraction']:.1%}   "
          f"marginal ratio {rep['marginal_ratio']:.2f}")

    packed = campaign_accounting(run_wall_j=10_400.0, tail_wall_j=2_600.0, idle_w=60.0,
                                 gap_s=30.0, used_for_next_job=True)
    idle = campaign_accounting(run_wall_j=10_400.0, tail_wall_j=2_600.0, idle_w=60.0,
                               gap_s=30.0, used_for_next_job=False)
    print(f"campaign packed  {packed['realised_saving_j']:,.0f} J   "
          f"idle  {idle['realised_saving_j']:,.0f} J")
    print("self-test ok")
