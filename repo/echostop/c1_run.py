"""Operational wrapper for the C1 training run. NOT part of the hash-locked scientific code (c1_train.py is untouched).

DEVIATION from the pre-run idle-GPU guard in c1_train.py, introduced 2026-10-12 ~17:00 (documented in
protocols/DEVIATIONS_AND_DECISIONS.md): the original guard waits while the mean GPU utilisation over 1 s is > 10%. While
the machine was in interactive use, desktop applications (VS Code, MSI Center) kept the REPORTED utilisation at ~19% while the board drew
~6 W (idle), so the trainer waited for 43 minutes (16:07-16:50) for a GPU that was effectively idle.

Relaxed guard: a run still waits when utilisation > 10% AND board power >= IDLE_POWER_W (a competing workload). Below that
power the utilisation is desktop compositing, which adds a few watts, proportional to time; savings are within-run ratios
of cumulative energy, so a time-proportional background draw cancels to first order.
Both values are kept in every run's meta.json: pre_util_raw_pct_mean (as measured) and pre_util_pct_mean (the value the
guard used), plus pre_guard. Runs trained before the switch used the original guard (pre_guard absent).
It affects only WHEN a run starts; it changes no data, recipe, seed, or analysis."""
from __future__ import annotations

import echostop.c1_train as T

IDLE_POWER_W = 25.0
_orig = T.gpu_state


def gpu_state():
    s = _orig()
    s["pre_util_raw_pct_mean"] = s["pre_util_pct_mean"]
    s["pre_guard"] = f"util>10 and power>={IDLE_POWER_W}W"
    if s["pre_power_w_mean"] < IDLE_POWER_W:
        s["pre_util_pct_mean"] = 0.0
    return s


T.gpu_state = gpu_state

def keep_awake():
    """Hold ES_CONTINUOUS | ES_SYSTEM_REQUIRED for the life of this process. Windows Modern Standby otherwise freezes desktop
    apps when the screen turns off (observed 2026-10-12 19:15 -> 2026-10-13 07:42), which corrupts the NVML energy integral.
    Non-persistent: released when the process exits. Does not keep the display on."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        print("keep-awake request set (system required)", flush=True)
    except Exception as e:
        print(f"keep-awake not available: {e}", flush=True)


if __name__ == "__main__":
    keep_awake()
    T.main()
