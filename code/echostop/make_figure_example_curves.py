"""Example stopping-curve figure (session addition, replacing figure1/
figure2's role): those two showed real accuracy-vs-energy curves, but only
for 2 of the corpus's 136 datasets, and were never cited inline. No other
figure in the paper shows what a training curve and a stopping decision
actually look like -- every other figure aggregates to a domain mean, a
distribution, or a per-dataset endpoint value. This is the mechanism
figure: 8 real, deliberately diverse examples (one tiny/tabular, one
large/tabular, two vision-cosine spanning a failure and a success case,
one step-decay, two NLP spanning a failure and a success case, one from
the newer vision-diversity batch), each panel a real validation curve with
every method's stop point marked.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from echostop.calibration import precompute_score_trace, simulate_from_trace
from echostop.gating_tests import eps_tolerance_oracle, patience_stop, meta_fraction_stop

GAMMA_HAT = 0.54
P_HAT = 10
F_HAT = 0.927
EPS = 0.5

METHOD_COLOR = {"oracle": "#000000", "online": "#D55E00", "patience": "#0072B2", "meta": "#999999"}
METHOD_LABEL = {"oracle": "Oracle", "online": "EcoStop", "patience": "Patience", "meta": "EcoStop-Meta"}

# (title, path) -- all real logged runs, seed 0 of each
EXAMPLES = [
    ("Tabular: Adult\n(EcoStop saves 68%)", "logs/calibration_openml/openml_adult_seed0.csv"),
    ("Tabular: Hayes-Roth\n(160 samples)", "logs/calibration_openml/openml_hayes-roth_seed0.csv"),
    ("NLP: SST-2\n(EcoStop saves 54%)", "logs/nlp_pilot/sst2_distilbert_seed0.csv"),
    ("NLP: Banking77\n(EcoStop saves 0%)", "logs/nlp_pilot/banking77_distilbert_seed0.csv"),
    ("Vision (cosine): MNIST\n(EcoStop saves 64%)", "logs/pilot/mnist_resnet9_seed0.csv"),
    ("Vision (cosine): CIFAR-10\n(EcoStop saves 0%)", "logs/pilot/cifar10_resnet9_seed0.csv"),
    ("Vision (cosine): GTSRB\n(EcoStop saves 25%)", "logs/pilot/gtsrb_resnet9_seed0.csv"),
    ("Vision (step-decay): CIFAR-10\n(EcoStop saves 12%)", "logs/step_decay_test/stepdecay_cifar10_resnet9_seed0.csv"),
]


def stops_for(df: pd.DataFrame) -> dict:
    t_oracle, _ = eps_tolerance_oracle(df, EPS)
    trace = precompute_score_trace(df, EPS)
    t_online = simulate_from_trace(trace, GAMMA_HAT, len(df))
    t_pat, _ = patience_stop(df, P_HAT)
    t_meta, _ = meta_fraction_stop(df, F_HAT)
    return {"oracle": t_oracle, "online": t_online, "patience": t_pat, "meta": t_meta}


def dodge(positions: dict, x_span: float, tol_frac: float = 0.02, spread_frac: float = 0.035) -> dict:
    """When two methods land on (near-)identical stop points, one marker/line
    completely hides the other -- a real reader-facing bug, not cosmetic
    (EcoStop and Patience tie exactly on 2 of these 8 panels). Nudge
    near-coincident points apart just enough that every method stays
    visible, without moving points that are already genuinely distinct."""
    items = sorted(positions.items(), key=lambda kv: kv[1])
    groups = [[items[0]]]
    for method, x in items[1:]:
        if x - groups[-1][-1][1] <= tol_frac * x_span:
            groups[-1].append((method, x))
        else:
            groups.append([(method, x)])
    out = {}
    for group in groups:
        k = len(group)
        offsets = [(j - (k - 1) / 2) * spread_frac * x_span for j in range(k)]
        for (method, x), off in zip(group, offsets):
            out[method] = x + off
    return out


def main() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))

    for i, (ax, (title, path)) in enumerate(zip(axes.flat, EXAMPLES)):
        row, col = divmod(i, 4)
        df = pd.read_csv(path)
        df["B_t"] = df["val_acc"].cummax()
        energy_kj = df["cumulative_energy_joules"] / 1000.0

        ax.plot(energy_kj, df["B_t"], color="#333333", linewidth=2.2, zorder=2)
        stops = stops_for(df)
        raw_x = {m: energy_kj.iloc[t] for m, t in stops.items()}
        x_span = energy_kj.iloc[-1] - energy_kj.iloc[0]
        dodged_x = dodge(raw_x, x_span)
        for method, t in stops.items():
            x = dodged_x[method]
            ax.axvline(x, color=METHOD_COLOR[method], linestyle="--",
                       linewidth=1.8, alpha=0.85, zorder=1)
            ax.plot(x, df["B_t"].iloc[t], "o", color=METHOD_COLOR[method],
                    markersize=9, zorder=3, markeredgecolor="white", markeredgewidth=0.8)

        ax.set_title(title, fontsize=19)
        if row == 1:
            ax.set_xlabel("Cumulative energy (kJ)", fontsize=18)
        if col == 0:
            ax.set_ylabel("Validation accuracy (%)", fontsize=18)
        ax.tick_params(labelsize=17)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#eeeeee", linewidth=0.6, zorder=0)

    handles = [plt.Line2D([0], [0], marker="o", color=METHOD_COLOR[m], linestyle="--",
                          markersize=11, label=METHOD_LABEL[m]) for m in ["oracle", "online", "patience", "meta"]]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=20, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle("Eight real validation curves with every method's stop point marked,\n"
                 "spanning tabular, NLP, and vision",
                 fontsize=25)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.11, hspace=0.4, wspace=0.22)
    out = "figures/figure6.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
