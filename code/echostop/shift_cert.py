"""Go/no-go for the novelty claim: does re-weighting the certificate to a declared
deployment mix restore validity across schedule families that the calibration pool
under-represents?

All inputs are the cached pooled matrices (cache/pool_mats.pkl): per-threshold loss
and energy saved for 259 jobs. No GPU, no training, no new measurement.

Outputs figures/tierA/shift_cert.csv and a printed table.
"""
from __future__ import annotations

import collections
import math
import pickle
import sys

import numpy as np
import pandas as pd

BETA = 0.05
RHOS = (0.10, 0.15, 0.20)
RULES = ("EcoStop", "Patience (p=f*T, min-delta 0.1)")   # the paper's two headline rules


def h1(a: float, b: float) -> float:
    """Binary KL divergence, with the boundary cases the loss actually hits."""
    a = min(max(a, 0.0), 1.0)
    if b <= 0.0 or b >= 1.0:
        return 0.0
    out = 0.0
    if a > 0:
        out += a * math.log(a / b)
    if a < 1:
        out += (1 - a) * math.log((1 - a) / (1 - b))
    return max(out, 0.0)


def weighted_hoeffding(L: np.ndarray, w: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """p-values for H0: E_w[L] >= rho and the weighted means, for every threshold row.

    For fixed weights w >= 0 and i.i.d. L_j in [0, 1],
        P(sum_j w_j L_j / sum_j w_j <= t) <= exp(-n_eff * h1(t, E_w L)),
    with n_eff = (sum w)^2 / sum w^2 (Hoeffding for weighted averages), so
        p = exp(-n_eff * h1(min(mean, rho), rho))
    is a valid p-value for the null. With w == 1 this reduces to the paper's
    Hoeffding bound at w = 1, which is what makes the comparison clean.
    """
    sw = float(w.sum())
    if sw <= 0:
        return np.ones(L.shape[0]), np.full(L.shape[0], np.nan)
    mean = (L * w).sum(1) / sw
    n_eff = sw * sw / float((w * w).sum())
    p = np.array([math.exp(-n_eff * h1(min(m, rho), rho)) for m in mean])
    return np.minimum(p, 1.0), mean


def select(p: np.ndarray) -> int | None:
    """Fixed-sequence: walk conservative -> aggressive, stop at the first p above beta."""
    last = None
    for i, pi in enumerate(p):
        if pi <= BETA:
            last = i
        else:
            break
    return last


def main() -> None:
    M, dom, grp, tag, et = pickle.load(open("cache/pool_mats.pkl", "rb"))
    grp = [str(g) for g in grp]
    tag = np.array([str(t) for t in tag])
    dom = np.array(dom)

    def family(name: str, t: str, d: str) -> str:
        """Schedule family, not domain: the grouping the certificate is stratified by.

        Cosine-with-warm-up is treated as the cosine family (warm-up is a descriptor,
        not a family); step decay and one-cycle are separate families because the
        plateau-then-jump mechanism is specific to them.
        """
        base = name.split(":", 1)[1]
        if d == "nlp":
            return "linear"                      # DistilBERT / ELECTRA / small BERT: linear decay
        if t == "b1" and d == "vision":
            if base.endswith("_step"):
                return "step"                    # ResNet-18, step decay
            if "vgg_small" in base:
                return "onecycle"                # VGG-style, one-cycle
        return "cosine"                          # incl. MobileNetV3 / ViT-Tiny warm-up

    fam = np.array([family(g, t, d) for g, t, d in zip(grp, tag, dom)])
    cal = np.where(tag == "dev")[0]
    n_cal = len(cal)
    counts = collections.Counter(fam[cal])
    piP = {f: counts[f] / n_cal for f in counts}

    print("calibration (dev) family mass over 142 jobs:")
    for f, c in counts.most_common():
        print(f"   {f:14s} {c:4d}  pi_P = {c / n_cal:.3f}")
    missing = sorted(set(fam) - set(counts))
    print("families absent from calibration:", missing)

    # declared deployment mixes
    mixes = {
        "M0 observed (current practice)": dict(piP),
        "M1 equal over all 4 families": {f: 0.25 for f in ("cosine", "linear", "step", "onecycle")},
        "M2 equal over the 2 supported": {"cosine": 0.5, "linear": 0.5},
        "M3 energy-matched (vision/NLP half each)": {"cosine": 0.5, "linear": 0.5},
    }

    rows = []
    for mix_name, piQ in mixes.items():
        # weight per calibration job = pi_Q(f) / pi_P(f); zero for unsupported families
        w = np.zeros(n_cal)
        for f, q in piQ.items():
            if f in piP and q > 0:
                idx = cal[fam[cal] == f]
                w[fam[cal] == f] = q / piP[f]
        supported = sorted({f for f, q in piQ.items() if q > 0 and f in piP})
        unsupported = sorted({f for f, q in piQ.items() if q > 0 and f not in piP})

        for rho in RHOS:
            for rule in RULES:
                L = M[rule]["L"]
                ES = M[rule]["ES"]
                p, mean = weighted_hoeffding(L[:, cal], w, rho)
                i_hat = select(p)
                for ev_name in ("b1", "c1"):
                    ev = np.where(tag == ev_name)[0]
                    adm = np.array([fam[j] in supported for j in ev])
                    if i_hat is None or adm.sum() == 0:
                        rows.append(dict(mix=mix_name, rho=rho, rule=rule, eval=ev_name,
                                         certified=False, risk=np.nan, risk_adm=np.nan,
                                         coverage_energy=0.0, EW_adm=np.nan, EW_policy=0.0,
                                         n_adm=int(adm.sum())))
                        continue
                    risk = float(L[i_hat, ev].mean())
                    risk_adm = float(L[i_hat, ev][adm].mean())
                    cov = float(et[ev][adm].sum() / et[ev].sum())
                    ew_adm = float(100 * ES[i_hat, ev][adm].sum() / et[ev][adm].sum())
                    ew_pol = float(100 * ES[i_hat, ev][adm].sum() / et[ev].sum())
                    rows.append(dict(mix=mix_name, rho=rho, rule=rule, eval=ev_name,
                                     certified=True, risk=risk, risk_adm=risk_adm,
                                     coverage_energy=cov, EW_adm=ew_adm, EW_policy=ew_pol,
                                     n_adm=int(adm.sum())))

    df = pd.DataFrame(rows)
    df.to_csv("figures/tierA/shift_cert.csv", index=False)

    pd.set_option("display.width", 220, "display.max_colwidth", 40)
    for rule in RULES:
        print(f"\n=== {rule} ===")
        sub = df[df.rule == rule]
        piv = sub.pivot_table(index=["mix", "rho"], columns="eval",
                              values=["risk_adm", "coverage_energy", "EW_policy"], sort=False)
        print(piv.round(3).to_string())

    # ---------------------------------------------------------------- per-family breakdown
    # The evidence for "the failure axis is the schedule family, not the domain":
    # at the observed mix, the same domain (vision) ranges from ok to broken.
    print("\n=== per-family risk at the observed mix (the axis that actually matters) ===")
    brk = []
    for rho in RHOS:
        for rule in RULES:
            L, ES = M[rule]["L"], M[rule]["ES"]
            w = np.ones(n_cal)
            p, _ = weighted_hoeffding(L[:, cal], w, rho)
            i_hat = select(p)
            if i_hat is None:
                continue
            for ev_name in ("b1", "c1"):
                ev = np.where(tag == ev_name)[0]
                for f in sorted(set(fam[ev])):
                    m = ev[fam[ev] == f]
                    brk.append(dict(rule=rule, rho=rho, eval=ev_name, family=f, n=len(m),
                                    risk=float(L[i_hat, m].mean()),
                                    EW=float(100 * ES[i_hat, m].sum() / et[m].sum()),
                                    energy_share_of_eval=float(100 * et[m].sum() / et[ev].sum())))
    bdf = pd.DataFrame(brk)
    bdf.to_csv("figures/tierA/shift_family_breakdown.csv", index=False)
    print(bdf.pivot_table(index=["eval", "family"], columns=["rule", "rho"], values="risk").round(3).to_string())

    # the arithmetic claim: a pooled certificate is vacuous below pi_d = rho
    print("\n=== domain mass vs energy share (the vacuity arithmetic) ===")
    for d in sorted(set(dom)):
        m = dom == d
        print(f"   {d:8s} jobs {m.sum():3d} ({m.mean():.3f} of jobs)   energy share {100 * et[m].sum() / et.sum():5.1f}%")
    for rho in RHOS:
        for d in sorted(set(dom)):
            pi_d = (dom == d).mean()
            bound = min(1.0, rho / pi_d)
            if bound >= 1.0:
                print(f"   rho={rho:.2f}  {d:8s} pi_d={pi_d:.3f} -> implied domain bound {bound:.2f} (VACUOUS)")


if __name__ == "__main__":
    main()
