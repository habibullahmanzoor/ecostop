"""C1: the stopping-rule families and operating-point logic, shared by c1_freeze.py and c1_eval.py (hash-locked).

Families (each an ordered list, conservative -> aggressive, of (label, fn(df, trace) -> stop epoch)):
  EcoStop-SR                 curve score < gamma, flat gate max(1, round(T/8)), k = min(2, max(1, round(T/20)))  [primary]
  EcoStop (paper gates)      the rule of the submitted paper: flat gate max(2, round(T/8)), k = 2
  LC-PFN + SR gates          pretrained LC-PFN score with the SR gates (secondary; needs LC-PFN traces)
  Patience p=fT, min-delta   patience max(1, round(f T)), min improvement 0.1 pp  [primary baseline: strongest calibrated
                             baseline on the development corpus]
  Patience p=fT              the same without min improvement
  Patience fixed p           patience p epochs (p = 20 ... 1), the uncalibrated practice
  Fixed budget fraction      stop at a fixed fraction of the budget
Every family is evaluated together with the never-stop point (run to completion), so a zero-risk operating point exists.
"""
from __future__ import annotations

import numpy as np

from echostop import tierA
from echostop.ablation import sim
from echostop.fv2_srgates import g_sr, k_sr
from echostop.lcpfn_real import PFN_GRID

ECO, ECO_P, PFN_SR = "EcoStop-SR", "EcoStop (paper gates)", "LC-PFN + SR gates"
PAT_MD, PAT, PAT_FIX, FIXFRAC = "Patience p=fT, min-delta 0.1", "Patience p=fT", "Patience fixed p", "Fixed budget fraction"
PRIMARY_RULE, PRIMARY_BASE, PRIMARY_RISK = ECO, PAT_MD, 0.10
FAMILIES = [ECO, ECO_P, PFN_SR, PAT_MD, PAT, PAT_FIX, FIXFRAC]
RISKS = (0.05, 0.10, 0.15, 0.20)


def families(with_pfn: bool = True):
    pol = tierA.build_policies(None)
    f = {}
    f[ECO] = [(f"g={g}", lambda df, tr, g=g: sim(tr["eco"], g, len(df), g_sr(len(df)), k_sr(len(df)))) for g in tierA.GAMMAS]
    f[ECO_P] = [(f"g={g}", lambda df, tr, g=g: sim(tr["eco"], g, len(df), max(2, round(0.125 * len(df))), 2)) for g in tierA.GAMMAS]
    if with_pfn:
        f[PFN_SR] = [(f"g={g}", lambda df, tr, g=g: sim(tr["pfn"], g, len(df), g_sr(len(df)), k_sr(len(df)))) for g in PFN_GRID]
    f[PAT_MD] = [(lab, lambda df, tr, fn=fn: fn(df, None)) for lab, fn in pol["Patience (p=f*T, min-delta 0.1)"]]
    f[PAT] = [(lab, lambda df, tr, fn=fn: fn(df, None)) for lab, fn in pol["Patience (p=f*T)"]]
    f[PAT_FIX] = [(lab, lambda df, tr, fn=fn: fn(df, None)) for lab, fn in pol["Patience (fixed p)"]]
    f[FIXFRAC] = [(lab, lambda df, tr, fn=fn: fn(df, None)) for lab, fn in pol["Fixed budget fraction"]]
    return f


def with_never_stop(loss, saved, es):
    """append the run-to-completion point (zero risk, zero saving) to [P, J] matrices"""
    z = np.zeros((1, loss.shape[1]))
    return np.vstack([loss, z]), np.vstack([saved, z]), np.vstack([es, z])


def frontier_index(loss, r):
    """index of the most energy-saving setting with job-level risk <= r is chosen by the caller; this returns the feasible mask"""
    return loss.mean(1) <= r + 1e-12


def fixed_sequence_empirical(loss, r):
    """Most aggressive index i such that the empirical job-level risk of EVERY setting up to i is <= r (no confidence bound).
    Returns -1 (never stop) if even the most conservative setting exceeds r."""
    ok = -1
    for i in range(loss.shape[0]):
        if loss[i].mean() <= r + 1e-12:
            ok = i
        else:
            break
    return ok
