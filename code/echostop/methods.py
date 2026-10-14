"""Stopping methods, plan §48 Step 7 -- three progressively stronger
policies, evaluated against the oracle/patience baselines already built in
gating_tests.py.

1. gain_per_joule_stop      -- baseline B7 (plan §19)
2. predictive_point_stop    -- point-estimate curve extrapolation (§9 Predictor 2)
3. ecostop_online_uncalibrated -- adds residual-based uncertainty to get a
   raw continuation score s_t (plan §6.3); stops against the *nominal*
   threshold rho, since no calibration has been run yet (plan §6.6 --
   this is "risk-aware", not "risk-controlled", until §6.6.3's offline
   calibration exists).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm


def _exp_saturation(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a - b * np.exp(-c * t)


SAFEGUARD_FRACTION = 0.125  # see schedule_relative_safeguard's docstring


def schedule_relative_safeguard(n_epochs: int) -> int:
    """min_epochs and plateau_gate were fixed absolute constants (5) tuned
    implicitly against ~40-epoch vision/tabular runs -- a 5-epoch floor is
    12.5% of a 40-epoch schedule, negligible. Found on the NLP leg (session
    diagnostic, SST-2 + DistilBERT, 10-epoch fine-tuning runs): the SAME
    fixed floor is 50% of a short schedule's total budget, so the method
    can't act until the run is half over regardless of how confident the
    underlying signal already is (s_t hit 0.0000 -- maximal confidence --
    at the very first epoch it was allowed to evaluate). Scale the floor
    relative to the run's own length instead of hardcoding it: 0.125 was
    chosen because it exactly reproduces the previously-validated 5/40
    behavior on vision, degrades gracefully on tabular (60 epochs -> 8,
    verified via sweep: still >=94% feasible at the pre-fix gamma_hat), and
    matches the ad hoc fix validated on NLP (10 epochs -> 2, recovering
    ~41% energy saved vs. the fixed floor's ~14%).
    """
    return max(2, round(SAFEGUARD_FRACTION * n_epochs))


PLATEAU_FLAT_TOL = 0.10  # pp of "improvement" still counted as flat -- see
# fit_exp_saturation's docstring and calibration.py's PLATEAU_FLAT_TOL for
# the step-decay diagnostic this addresses (strict non-improvement resets
# the flat count on any single noise-level uptick, even mid-plateau)


def _epochs_flat(b: np.ndarray, t: int) -> int:
    epochs_flat = 0
    for s in range(t, 0, -1):
        if b[s] - b[s - 1] > PLATEAU_FLAT_TOL:
            break
        epochs_flat += 1
    return epochs_flat


def fit_exp_saturation(t_arr: np.ndarray, q_arr: np.ndarray, window: int = 15):
    """Fit Q(t) = a - b*exp(-c*t) to the observed prefix (plan §9 Predictor 2).
    Returns (a, b, c, residual_std) or None if the fit fails / too little data.

    Bug fixed #1: `a` must NOT be lower-bounded at `q_arr.max()` (= current B_t).
    That bound sits exactly on the current best, so any locally-flat/noisy
    stretch of data lets curve_fit trivially satisfy the objective by pinning
    `a` to the boundary (predicted_gain = 0.00 exactly) -- a degenerate
    solution, not a genuine extrapolation. Leave `a` effectively unconstrained
    during the fit and floor the *result* at B_t afterward instead, so the
    optimizer isn't incentivized toward the degenerate corner.

    Bug fixed #2 (found on step-decay data, session diagnostic): fitting the
    FULL prefix from epoch 0 forces one 3-parameter smooth curve to explain
    both the early steep-rise phase and later flat/step regions at once. For
    step-decay schedules the residuals from the early phase stay large no
    matter how converged the run currently is, so `resid_std` (and therefore
    the uncertainty-weighted score s_t) stays inflated long after the local
    trend has visibly flattened -- predicted_gain correctly collapses to ~0
    but sigma never shrinks enough to let s_t drop below any reasonable
    threshold. Fit only the trailing `window` epochs instead of the full
    history: recent residuals reflect recent (local) fit quality, which is
    what the extrapolation from "now" actually depends on.

    Bug fixed #3 (found immediately after #2, same diagnostic): a short,
    near-flat window under-determines the 3-parameter exponential -- `a` and
    `c` trade off freely when the window shows almost no curvature, and
    curve_fit can "resolve" that by driving `a` to its upper bound with a
    tiny `c`, producing a huge phantom predicted gain with an artificially
    tiny residual (false confidence that a big improvement is still coming,
    exactly the plateau this predictor is supposed to catch). `a` is a
    validation-accuracy percentage, so bound it at the physical ceiling
    (100) instead of an arbitrary 200. When the optimizer pins `a` against
    that ceiling, the window's curvature can't identify a finite asymptote
    at all -- fall back to the "already converged" reading (a = current
    best, sigma = local spread around it).

    Bug fixed #4 (found testing #3's fix on other epochs of the same run):
    even when curve_fit lands on an interior (non-boundary) solution, a
    short window can still leave the decay rate `c` weakly identified --
    e.g. a slow, barely-curving rise fits the 15 in-window points almost
    perfectly (tiny residual std) while implying an asymptote far beyond
    what those 15 points can actually pin down (huge true extrapolation
    uncertainty). Residual std around the fitted curve measures in-sample
    noise, not confidence in the extrapolated asymptote -- it's the wrong
    quantity here. Use the parameter standard error on `a` from curve_fit's
    own covariance matrix instead: it grows exactly when `a` trades off
    against `c` (poorly-determined decay rate -> poorly-determined
    asymptote), which is precisely the failure mode observed.
    """
    if len(t_arr) < 4:
        return None
    if len(t_arr) > window:
        t_arr = t_arr[-window:]
        q_arr = q_arr[-window:]
    q_max = q_arr.max()
    a_upper = 100.0
    a0 = min(q_max + 1.0, a_upper)  # initial guess must stay inside the bounds below
    b0 = max(a0 - q_arr[0], 1.0)
    c0 = 0.1
    try:
        popt, pcov = curve_fit(
            _exp_saturation, t_arr, q_arr,
            p0=[a0, b0, c0],
            bounds=([0.0, 0, 1e-4], [a_upper, 200.0, 5.0]),
            maxfev=5000,
        )
    except RuntimeError:
        return None
    a, b, c = popt
    if a >= a_upper - 0.5:
        # optimizer pinned against the physical ceiling: no finite
        # asymptote is identifiable from this window -- read as converged
        a = q_max
        resid = q_arr - q_max
        sigma = float(np.std(resid)) if len(resid) > 1 else 1.0
    else:
        a = max(a, q_max)  # floor post-hoc: asymptote can't be below what's observed
        perr_a = np.sqrt(pcov[0, 0]) if np.isfinite(pcov[0, 0]) else np.inf
        sigma = perr_a if np.isfinite(perr_a) else 100.0  # non-finite -> maximally uncertain
    sigma = max(sigma, 0.05)  # floor so sigma is never ~0
    return a, b, c, sigma


def _simulate(df: pd.DataFrame, decision_fn, min_epochs: int, patience_k: int):
    """Shared simulation loop: call decision_fn(t) -> bool ("should stop
    signal fired at t") each epoch; stop after patience_k consecutive fires,
    never before min_epochs (plan §15 safeguards)."""
    consecutive = 0
    for t in range(len(df)):
        if t < min_epochs:
            continue
        fire = decision_fn(t)
        if fire:
            consecutive += 1
            if consecutive >= patience_k:
                return t
        else:
            consecutive = 0
    return len(df) - 1  # never fired -> ran to completion


def gain_per_joule_stop(df: pd.DataFrame, tau: float, window: int = 3,
                         min_epochs: int = 5, patience_k: int = 2) -> int:
    """B7 (plan §5, §19): stop when recent (B_t - B_{t-window}) / (E_t - E_{t-window}) < tau."""
    b = df["B_t"].values
    e = df["cumulative_energy_joules"].values

    def decision(t: int) -> bool:
        t0 = max(0, t - window)
        d_q = b[t] - b[t0]
        d_e = e[t] - e[t0]
        if d_e <= 0:
            return True
        return (d_q / d_e) < tau

    return _simulate(df, decision, min_epochs, patience_k)


def predictive_point_stop(df: pd.DataFrame, eps: float,
                           min_epochs: int | None = None, patience_k: int = 2,
                           plateau_gate: int | None = None) -> int:
    """Point-estimate predictive stopping (§9 Predictor 2, no uncertainty):
    fit the saturation curve to the prefix, predict the asymptote `a`,
    stop when (a - B_t) < eps.

    plan §15 safeguard, implemented explicitly: a plain exponential-saturation
    fit reads a mid-training plateau (routine under cosine LR decay, where
    accuracy resumes climbing as LR keeps annealing) as convergence -- the
    fitted asymptote collapses to the current best with near-zero predicted
    gain, firing a false stop. Gate the signal behind `plateau_gate`
    epochs of genuinely no improvement in B_t, so a temporary flat stretch
    can't fire alone; this is exactly the "plateau recovery" / "consecutive
    decision requirement" safeguards the plan specifies, not an ad hoc patch.

    min_epochs/plateau_gate default to schedule_relative_safeguard(len(df))
    rather than a fixed constant -- see its docstring for why (found on the
    NLP leg: a fixed 5-epoch floor is half the budget of a 10-epoch run).
    """
    if min_epochs is None:
        min_epochs = schedule_relative_safeguard(len(df))
    if plateau_gate is None:
        plateau_gate = schedule_relative_safeguard(len(df))
    t_all = df["epoch"].values.astype(float)
    q_all = df["val_acc"].values.astype(float)
    b = df["B_t"].values

    def decision(t: int) -> bool:
        fit = fit_exp_saturation(t_all[: t + 1], q_all[: t + 1])
        if fit is None:
            return False
        a, _, _, _ = fit
        predicted_gain = a - b[t]
        return predicted_gain < eps and _epochs_flat(b, t) >= plateau_gate

    return _simulate(df, decision, min_epochs, patience_k)


def ecostop_online_uncalibrated(df: pd.DataFrame, eps: float, rho: float,
                                 min_epochs: int | None = None, patience_k: int = 2,
                                 plateau_gate: int | None = None) -> tuple[int, list]:
    """plan §6.3: s_t = P(G_t > eps | H_t), stop when s_t < rho (nominal --
    UNCALIBRATED, see module docstring). Uncertainty comes from the residual
    std of the curve fit, treating predicted future gain as Normal(mean, std).

    Same plateau vulnerability as predictive_point_stop applies here too:
    the residual std around a degenerate flat fit is ALSO near-zero, so the
    "uncertainty-aware" score collapses to false confidence rather than
    protecting against the false positive -- naive residual-based sigma is
    not a substitute for the plan's explicit plateau safeguard (§15).

    min_epochs/plateau_gate default to schedule_relative_safeguard(len(df));
    see its docstring.

    Returns (stop_epoch, [(t, s_t), ...] trace) for later calibration work."""
    if min_epochs is None:
        min_epochs = schedule_relative_safeguard(len(df))
    if plateau_gate is None:
        plateau_gate = schedule_relative_safeguard(len(df))
    t_all = df["epoch"].values.astype(float)
    q_all = df["val_acc"].values.astype(float)
    b = df["B_t"].values
    trace = []

    def decision(t: int) -> bool:
        fit = fit_exp_saturation(t_all[: t + 1], q_all[: t + 1])
        if fit is None:
            trace.append((t, 1.0))  # can't estimate -> assume worth continuing
            return False
        a, _, _, sigma = fit
        predicted_gain_mean = a - b[t]
        # s_t = P(true future gain > eps), gain ~ Normal(predicted_gain_mean, sigma)
        s_t = 1.0 - norm.cdf(eps, loc=predicted_gain_mean, scale=sigma)
        s_t = float(np.clip(s_t, 0.0, 1.0))
        trace.append((t, s_t))
        return s_t < rho and _epochs_flat(b, t) >= plateau_gate

    stop_epoch = _simulate(df, decision, min_epochs, patience_k)
    return stop_epoch, trace
