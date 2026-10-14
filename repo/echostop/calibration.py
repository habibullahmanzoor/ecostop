"""Route A calibration (plan §6.6.3): select a deployable threshold gamma_hat
via LTT-style per-candidate concentration bound + multiplicity correction,
targeting P(L_gamma = 1) <= rho with confidence 1-beta.

IMPORTANT LIMITATION, stated up front and echoed in every report this module
prints: the plan's own §25.1 sample-size table says a *single fixed* rho=0.05
target needs n=59 independent jobs with zero observed failures just to reach
that bound at beta=0.05, and the real (grid-search, multiplicity-corrected)
procedure needs more than that. We have 4 cells here. Nothing this module
produces should be read as a certified (rho, beta) claim -- it validates that
the calibration MACHINERY is implemented correctly, using real data, at a
scale far too small to be statistically meaningful. See §6.6.2 on why the
independence unit is the JOB, not the seed -- multiple seeds of one cell are
NOT independent draws for calibration purposes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from echostop.methods import fit_exp_saturation, schedule_relative_safeguard
from scipy.stats import norm


def forward_fill_audit(df: pd.DataFrame) -> np.ndarray:
    """A_t = Q_{j_t}^test (plan §11.0.1): the test-set score of whatever
    checkpoint validation had selected as of epoch t. test_acc_at_checkpoint
    is only logged on improvement epochs; forward-fill to get A_t everywhere."""
    raw = pd.to_numeric(df["test_acc_at_checkpoint"], errors="coerce").values
    a_t = np.copy(raw)
    last = np.nan
    for i in range(len(a_t)):
        if not np.isnan(a_t[i]):
            last = a_t[i]
        else:
            a_t[i] = last
    return a_t


PLATEAU_FLAT_TOL = 0.10  # pp; see fit_exp_saturation's docstring bug #2-4 and
# methods.py's PLATEAU_FLAT_TOL for the step-decay diagnostic this addresses


def precompute_score_trace(df: pd.DataFrame, eps: float, min_epochs: int | None = None) -> list[dict]:
    """The expensive part (curve refit at every epoch) does NOT depend on
    gamma -- only the final `s_t < gamma` comparison does. Precompute once
    per run and reuse across the whole gamma grid, instead of redoing the
    full curve-fit sweep once per candidate (a ~20x redundant cost that's
    what made the first calibration run take the better part of an hour).

    epochs_flat bug (found on step-decay data, session diagnostic): counting
    STRICTLY non-improving epochs (`b[s] > b[s-1]`) means any single epoch
    with a tiny (e.g. 0.05pp) genuine-noise uptick resets the count to zero,
    even during a converged low-LR fine-tuning phase where B_t keeps
    creeping up by fractions of a point every few epochs. Allow up to
    PLATEAU_FLAT_TOL of "improvement" to still count as flat -- verified
    against the vision cosine pilot (identical 11/12 feasibility, 1/12
    fired-early, at flat_tol=0 vs 0.10 -- no regression) while letting the
    step-decay corpus's genuine post-drop plateaus register as flat.

    min_epochs defaults to schedule_relative_safeguard(len(df)) rather than
    a fixed constant -- see methods.py for why (found on the NLP leg: a
    fixed 5-epoch floor is half the budget of a 10-epoch fine-tuning run).
    """
    if min_epochs is None:
        min_epochs = schedule_relative_safeguard(len(df))
    t_all = df["epoch"].values.astype(float)
    q_all = df["val_acc"].values.astype(float)
    b = df["B_t"].values
    trace = []
    for t in range(len(df)):
        if t < min_epochs:
            trace.append({"t": t, "s_t": None, "epochs_flat": None})
            continue
        fit = fit_exp_saturation(t_all[: t + 1], q_all[: t + 1])
        if fit is None:
            trace.append({"t": t, "s_t": None, "epochs_flat": None})
            continue
        a, _, _, sigma = fit
        predicted_gain_mean = a - b[t]
        s_t = 1.0 - norm.cdf(eps, loc=predicted_gain_mean, scale=sigma)
        epochs_flat = 0
        for s in range(t, 0, -1):
            if b[s] - b[s - 1] > PLATEAU_FLAT_TOL:
                break
            epochs_flat += 1
        trace.append({"t": t, "s_t": s_t, "epochs_flat": epochs_flat})
    return trace


def simulate_from_trace(trace: list[dict], gamma: float, n_epochs: int,
                         patience_k: int = 2, plateau_gate: int | None = None) -> int:
    """Cheap, gamma-dependent part: apply a threshold to a precomputed
    score trace. This is what actually varies across the LTT grid search.

    plateau_gate defaults to schedule_relative_safeguard(n_epochs); see
    methods.py for why a fixed constant doesn't generalize across domains
    with very different schedule lengths (vision/tabular vs. short NLP
    fine-tuning runs)."""
    if plateau_gate is None:
        plateau_gate = schedule_relative_safeguard(n_epochs)
    consecutive = 0
    for row in trace:
        t = row["t"]
        fire = (
            row["s_t"] is not None
            and row["s_t"] < gamma
            and row["epochs_flat"] >= plateau_gate
        )
        if fire:
            consecutive += 1
            if consecutive >= patience_k:
                return t
        else:
            consecutive = 0
    return n_epochs - 1


def simulate_policy_fixed_gamma(df: pd.DataFrame, gamma: float, eps: float,
                                 min_epochs: int | None = None, patience_k: int = 2,
                                 plateau_gate: int | None = None) -> int:
    """Deploy a FIXED, pre-specified gamma live -- this is what "the policy
    uses threshold gamma" means operationally (plan §6.3, §14 pseudocode).
    Convenience wrapper for one-off calls; ltt_calibrate uses the split
    precompute/simulate path above instead of calling this in a loop, since
    that redoes the expensive part once per gamma unnecessarily."""
    trace = precompute_score_trace(df, eps, min_epochs)
    return simulate_from_trace(trace, gamma, len(df), patience_k, plateau_gate)


def audited_loss(df: pd.DataFrame, stop_epoch: int, eps: float) -> int:
    """L_gamma(r) = 1[A_T(r) - A_{tau_gamma}(r) > eps] (plan §6.6.3)."""
    a_t = forward_fill_audit(df)
    a_final = a_t[-1]
    a_stop = a_t[stop_epoch]
    return int((a_final - a_stop) > eps)


def hoeffding_ucb(losses: list[int], beta_corrected: float) -> float:
    """One-sided Hoeffding upper confidence bound on a Bernoulli mean."""
    n = len(losses)
    p_hat = float(np.mean(losses))
    t = np.sqrt(np.log(1.0 / beta_corrected) / (2 * n))
    return min(p_hat + t, 1.0)


def ltt_calibrate(calibration_dfs: list[pd.DataFrame], gamma_grid: np.ndarray,
                   eps: float, rho: float, beta: float) -> dict:
    """LTT (plan §6.6.3): reframe calibration as multiple hypothesis testing.
    For each candidate gamma (ascending = more aggressive/more stopping),
    test H0: risk(gamma) > rho via a Hoeffding UCB with Bonferroni correction
    across the grid; certify gammas whose UCB <= rho; select the MOST
    aggressive (largest) certified gamma -- maximizes energy saved subject to
    the risk constraint.
    """
    beta_corrected = beta / len(gamma_grid)
    # precompute once per calibration job, reuse across the whole gamma grid
    traces = [precompute_score_trace(df, eps) for df in calibration_dfs]

    certified = []
    ucbs = []
    for gamma in gamma_grid:
        losses = []
        for df, trace in zip(calibration_dfs, traces):
            t_stop = simulate_from_trace(trace, gamma, len(df))
            losses.append(audited_loss(df, t_stop, eps))
        ucb = hoeffding_ucb(losses, beta_corrected)
        ucbs.append(ucb)
        if ucb <= rho:
            certified.append(gamma)

    gamma_hat = max(certified) if certified else None
    return {
        "gamma_grid": gamma_grid,
        "ucbs": ucbs,
        "certified": certified,
        "gamma_hat": gamma_hat,
        "beta_corrected": beta_corrected,
        "n_calibration_jobs": len(calibration_dfs),
    }
