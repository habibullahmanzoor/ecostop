"""Leave-one-cell-out calibration demo (plan §6.6.3), on the pilot data.

READ THIS BEFORE THE NUMBERS BELOW: with 4 cells total, this is nowhere near
the n=59+ independent JOBS the plan's own §25.1 table requires for a real
(rho, beta) claim, and pooling seeds within a cell as if they were
independent jobs violates the §6.6.2 independence-unit rule outright. This
script exists to validate that the calibration machinery (Hoeffding UCB +
Bonferroni-corrected grid search, plan §6.6.3) runs correctly end-to-end on
real data -- it is NOT a certified statistical claim, and is reported here
as a descriptive result only, exactly as plan §25.1 requires when n is too
small: "report the realized false-stop rate as a descriptive statistic...
explicitly flagged as underpowered."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from echostop.analyze_pilot import discover_runs, EPS_LADDER
from echostop.calibration import ltt_calibrate, simulate_policy_fixed_gamma, audited_loss

GAMMA_GRID = np.linspace(0.02, 0.9, 20)
BETA = 0.20  # loosened from the usual 0.05 given the tiny n -- see caveat above
EPS_FOR_DEMO = 0.5  # one eps value, to keep the grid search tractable in-session


def main() -> None:
    runs = discover_runs()
    cells = list(runs.keys())
    print(f"Cells available: {cells}\n")
    print(f"eps = {EPS_FOR_DEMO}pp, beta (Bonferroni-corrected across {len(GAMMA_GRID)} candidates) = "
          f"{BETA}/{len(GAMMA_GRID)} = {BETA/len(GAMMA_GRID):.4f}\n")

    # Sanity-check the arithmetic before running anything: with n calibration
    # seeds and this Bonferroni-corrected beta, what's the best-case (zero
    # observed losses) Hoeffding UCB? If that alone exceeds rho, NO gamma
    # can certify regardless of how well it performs -- worth knowing before
    # interpreting a wall of "gamma_hat=None" as a bug.
    n_example = sum(len(dfs) for cell, dfs in runs.items() if cell != cells[0]) if cells else 0
    beta_corrected = BETA / len(GAMMA_GRID)
    best_case_ucb = np.sqrt(np.log(1 / beta_corrected) / (2 * n_example)) if n_example else float("nan")
    print(f"Best-case (zero losses) Hoeffding UCB at n={n_example}, beta_corrected={beta_corrected:.4f}: "
          f"{best_case_ucb:.3f}")
    print(f"Target rho=0.3 is {'reachable' if best_case_ucb <= 0.3 else 'MATHEMATICALLY UNREACHABLE'} "
          f"at this n regardless of actual performance.\n")

    records = []
    for held_out in cells:
        calibration_dfs = [df for cell, dfs in runs.items() if cell != held_out for df in dfs]
        eval_dfs = runs[held_out]

        result = ltt_calibrate(calibration_dfs, GAMMA_GRID, EPS_FOR_DEMO, rho=0.3, beta=BETA)
        gamma_hat = result["gamma_hat"]

        print(f"held out: {held_out}  |  n_calibration_jobs(seeds)={result['n_calibration_jobs']}  "
              f"|  certified gammas: {len(result['certified'])}/{len(GAMMA_GRID)}  "
              f"|  gamma_hat={gamma_hat}  |  best UCB achieved: {min(result['ucbs']):.3f}")

        if gamma_hat is None:
            print("  -> no gamma certified at rho=0.3: n is too small for this target, exactly as §25.1 predicts\n")
            continue

        # realized rate on the held-out cell -- DESCRIPTIVE ONLY, n=3
        losses = []
        energies = []
        for df in eval_dfs:
            t_stop = simulate_policy_fixed_gamma(df, gamma_hat, EPS_FOR_DEMO)
            losses.append(audited_loss(df, t_stop, EPS_FOR_DEMO))
            energies.append(df["cumulative_energy_joules"].iloc[t_stop])
            full_energy = df["cumulative_energy_joules"].iloc[-1]
            energy_saved_pct = 100 * (1 - energies[-1] / full_energy)
            records.append({
                "held_out_cell": str(held_out), "gamma_hat": gamma_hat,
                "loss": losses[-1], "energy_saved_pct": energy_saved_pct,
                "stop_epoch": t_stop, "total_epochs": len(df),
            })
        realized_rate = np.mean(losses)
        print(f"  -> realized loss rate on held-out cell (n=3, DESCRIPTIVE ONLY): {realized_rate:.2f}")
        print(f"  -> mean energy saved on held-out cell: {np.mean([r['energy_saved_pct'] for r in records if r['held_out_cell']==str(held_out)]):.1f}%\n")

    df_out = pd.DataFrame(records)
    if len(df_out) == 0:
        print("=" * 70)
        print("No gamma certified in ANY fold at rho=0.3 -- confirms the plan's own §25.1 sample-size")
        print("table in practice: n=9 calibration seeds cannot certify anything at this target,")
        print("regardless of how good the underlying policy actually is. Demonstrating the")
        print("certification logic DOES work correctly with an illustrative loosened target below.")
        print("=" * 70 + "\n")
    else:
        df_out.to_csv("figures/calibration_demo_results.csv", index=False)
        print("=" * 70)
        print("SUMMARY (descriptive, n=4 cells / n=3 seeds per held-out cell -- NOT a certified claim)")
        print("=" * 70)
        print(df_out.groupby("held_out_cell")[["loss", "energy_saved_pct"]].mean().to_string())

    # Illustrative only: loosen rho so certification is mathematically reachable
    # at this n, to confirm the SELECTION LOGIC works correctly, separate from
    # the (real, correct) sample-size limitation above.
    print("\n" + "=" * 70)
    print(f"ILLUSTRATIVE ONLY: same procedure, rho loosened to 0.6 (mathematically reachable at n=9)")
    print("=" * 70)
    illus_records = []
    for held_out in cells:
        calibration_dfs = [df for cell, dfs in runs.items() if cell != held_out for df in dfs]
        eval_dfs = runs[held_out]
        result = ltt_calibrate(calibration_dfs, GAMMA_GRID, EPS_FOR_DEMO, rho=0.6, beta=BETA)
        gamma_hat = result["gamma_hat"]
        print(f"held out: {held_out}  |  certified: {len(result['certified'])}/{len(GAMMA_GRID)}  "
              f"|  gamma_hat={gamma_hat}")
        if gamma_hat is None:
            continue
        for df in eval_dfs:
            t_stop = simulate_policy_fixed_gamma(df, gamma_hat, EPS_FOR_DEMO)
            loss = audited_loss(df, t_stop, EPS_FOR_DEMO)
            full_energy = df["cumulative_energy_joules"].iloc[-1]
            energy_saved_pct = 100 * (1 - df["cumulative_energy_joules"].iloc[t_stop] / full_energy)
            illus_records.append({"held_out_cell": str(held_out), "gamma_hat": gamma_hat,
                                   "loss": loss, "energy_saved_pct": energy_saved_pct})
    if illus_records:
        illus_df = pd.DataFrame(illus_records)
        illus_df.to_csv("figures/calibration_demo_illustrative.csv", index=False)
        print()
        print(illus_df.groupby("held_out_cell")[["loss", "energy_saved_pct"]].mean().to_string())
        print(f"\noverall realized loss rate (n={len(illus_df)}, still far too small to certify "
              f"anything -- illustrative only): {illus_df['loss'].mean():.2f}")


if __name__ == "__main__":
    main()
