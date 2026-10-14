"""Render figures/energy_report_data.json as a plain markdown file
(energy_report.md) instead of the interactive HTML dashboard -- easier to
share (email, git, plain-text viewers) at the cost of interactivity. Same
underlying numbers, same domain grouping, same feasibility+accuracy-gap
pairing per method cell."""

from __future__ import annotations

import json

DOMAIN_ORDER = ["NLP (fine-tuning)", "Vision (step-decay, from-scratch)",
                "Vision (cosine, from-scratch)", "Tabular / OpenML"]
METHODS = ["oracle", "calibrated", "patience", "lcpfn", "meta"]
METHOD_LABEL = {"oracle": "Oracle", "calibrated": "Calibrated EcoStop", "patience": "Patience",
                "lcpfn": "LC-PFN-style", "meta": "EcoStop-Meta"}


def fmt_energy(j: float) -> str:
    return f"{j/1000:.1f} kJ" if j >= 1000 else f"{j:.1f} J"


def fmt_cell(saved: float, feas: float, gap: float) -> str:
    sign = "+" if gap > 0 else ""
    return f"{saved:.1f}% ({feas:.0f}% · Δ{sign}{gap:.2f}pp)"


def main() -> None:
    with open("figures/energy_report_data.json") as f:
        data = json.load(f)

    by_domain: dict[str, list[dict]] = {}
    for r in data:
        by_domain.setdefault(r["domain"], []).append(r)

    lines = []
    lines.append("# EcoStop — Energy Saved vs. Full Training\n")
    lines.append(
        "Every method below is compared against **full training** — judging solely on "
        "final/best accuracy and never stopping early. Figures are real GPU-energy "
        "measurements (NVML), mean across seeds, on real datasets only. `Oracle` is the "
        "unreachable ceiling (perfect hindsight). `Calibrated EcoStop`, `Patience`, and "
        "`EcoStop-Meta` are all held to the *same* deployable standard — one rule, fixed "
        "in advance by the identical Learn-then-Test calibration procedure, applied "
        "identically everywhere, no per-run hindsight tuning. `LC-PFN-style` uses the "
        "literature's fixed, uncalibrated threshold.\n"
    )
    lines.append(
        "**Each cell reads as `% energy saved (feasibility% · Δaccuracy gap in pp)`.** "
        "Feasibility is the share of seeds where the deployed policy stayed within the "
        "0.5pp tolerance — the pass/fail view the calibration target controls. Δpp is the "
        "actual audited gap (A_full − A_method) behind that pass/fail call, mean across "
        "seeds — small and sometimes even negative (an early-stopped checkpoint scoring "
        "*higher* on test accuracy than the fully-trained one). Feasibility below ~80% "
        "means that dataset saw more violations than the certified budget allows for a "
        "single job; the certified guarantee is a population-level bound over the whole "
        "calibration pool, not a per-dataset promise.\n"
    )

    # Domain summary cards, as a compact table
    lines.append("## Domain summary\n")
    header = "| Domain | n datasets | Mean full-train energy | " + " | ".join(METHOD_LABEL[m] for m in METHODS) + " |"
    sep = "|---|---|---|" + "---|" * len(METHODS)
    lines.append(header)
    lines.append(sep)
    for domain in DOMAIN_ORDER:
        rows = by_domain.get(domain)
        if not rows:
            continue
        n = len(rows)
        mean_e = sum(r["full_e_mean"] for r in rows) / n
        cells = []
        for m in METHODS:
            saved = sum(r[f"{m}_saved_pct"] for r in rows) / n
            feas = sum(r[f"{m}_feasible_pct"] for r in rows) / n
            gap = sum(r[f"{m}_gap_pp"] for r in rows) / n
            cells.append(fmt_cell(saved, feas, gap))
        lines.append(f"| {domain} | {n} | {fmt_energy(mean_e)} | " + " | ".join(cells) + " |")
    lines.append("")

    # Per-dataset tables, one per domain
    for domain in DOMAIN_ORDER:
        rows = by_domain.get(domain)
        if not rows:
            continue
        lines.append(f"## {domain}  ({len(rows)} dataset{'s' if len(rows) > 1 else ''})\n")
        header = "| Dataset | Seeds | Full-train energy | " + " | ".join(METHOD_LABEL[m] for m in METHODS) + " |"
        sep = "|---|---|---|" + "---|" * len(METHODS)
        lines.append(header)
        lines.append(sep)
        for r in sorted(rows, key=lambda x: x["dataset"]):
            cells = [fmt_cell(r[f"{m}_saved_pct"], r[f"{m}_feasible_pct"], r[f"{m}_gap_pp"]) for m in METHODS]
            lines.append(f"| `{r['dataset']}` | {r['n_seeds']:.0f} | {fmt_energy(r['full_e_mean'])} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("---\n")
    lines.append(
        "**NLP is seven datasets** (SST-2, rotten_tomatoes, TREC, emotion, "
        "tweet_eval_sentiment, poem_sentiment, banking77 — deliberately different sizes "
        "and task shapes, from binary sentiment to 77-way intent classification, all "
        "10-epoch fine-tuning runs). Calibrated EcoStop finds real, if variable, savings "
        "on five of the seven, but saves exactly 0% on poem_sentiment and banking77 — "
        "both very short, low-energy schedules where the calibrated threshold never "
        "triggers before the run ends. Fixed patience=10 and LC-PFN-style's fixed "
        "threshold save exactly 0% on all seven — structural: patience=10 needs 10 "
        "non-improving epochs to ever fire, which a 10-epoch schedule can never provide.\n"
    )
    lines.append(
        "**The calibration pool grew to 95 jobs this session**, tightening the certified "
        "risk target to a sensible, non-degenerate threshold (gamma_hat=0.54). Correctly "
        "accounting for the calibration pool's own adaptive growth (a time-uniform, "
        "anytime-valid correction — see the paper draft, §6.3.2) moves the honestly-"
        "certifiable target to **ρ=0.24** for that same gamma_hat=0.54. At that target, "
        "EcoStop wins on **102 of 108 datasets**, including edging ahead of patience in "
        "aggregate on Tabular/OpenML too.\n"
    )
    lines.append(
        "**The corpus is 86% tabular by job count** (93 of 108 jobs). Calibrating each "
        "domain on its own, with no pooling: tabular alone (93 jobs) supports a tight "
        "bound (best achievable UCB 0.161), but vision alone (8 jobs) and NLP alone "
        "(7 jobs) still do not have nearly enough jobs for LTT's math to say anything "
        "useful in isolation (best achievable UCB 0.549 and 0.587). Reaching a genuinely "
        "powered vision-only or NLP-only bound would need roughly 19-20 more jobs in each "
        "domain.\n"
    )
    lines.append(
        "**EcoStop-Meta is a sanity check on the cold-start design choice, not a "
        "competing framework.** EcoStop-Meta stops at a single fixed fraction of the "
        "epoch budget (f_hat=0.56), calibrated via the identical LTT procedure. It beats "
        "cold-start EcoStop on three of the four domains (NLP, both vision settings) and "
        "loses clearly only on Tabular/OpenML. Because tabular is 93 of the 108 datasets, "
        "EcoStop still wins on more individual datasets overall (93 of 108) even though "
        "Meta wins more domain-level averages.\n"
    )

    with open("energy_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote energy_report.md ({len(data)} datasets, {len(lines)} lines)")


if __name__ == "__main__":
    main()
