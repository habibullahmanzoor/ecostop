# EcoStop-Bench

Energy-metered learning curves and a matched-risk protocol for comparing early-stopping rules.

923 training runs, of which 259 jobs and 897 runs carry the matched-risk analysis, with per-epoch validation and test accuracy and GPU-board energy measured through NVML on a single
NVIDIA RTX 4080 Laptop GPU. Companion to the paper *Early stopping, audited: curve extrapolation adds nothing, and the schedule decides*.

## Files

| File | Content |
|---|---|
| `runs.csv.gz` | one row per (run, epoch), 37,090 rows |
| `bench.py` | loader, certification, rules, out-of-fold evaluation (numpy, pandas, scipy only) |
| `export_data.py` | how `runs.csv.gz` was built from the raw logs |

## Data dictionary (`runs.csv.gz`)

| Column | Meaning |
|---|---|
| `set` | `c1` (82-job confirmatory fine-tuning study: 41 datasets x 2 recipes, 246 runs; calibrate on the other sets and apply unchanged), `dev` (142-job development corpus, 535 runs), `dev_extra` (step-decay CIFAR-10, 3 runs), `fresh` (35 jobs, 116 runs, thresholds frozen before training), `regime_freezethaw` (6 jobs, 18 runs), `repeat` (5 identical repeats of one job) |
| `job`, `seed` | a job is a (dataset, model, recipe) triple; seeds are repeated runs of one job and are NOT independent |
| `domain` | `vision`, `tabular`, `nlp` |
| `schedule_family` | `cosine`, `tab_cosine`, `nlp_linear`, `cosine_warmup`, `step`, `onecycle`, `cosine_freeze12` |
| `horizon_T` | epochs of the run |
| `epoch` | 0-based |
| `val_acc`, `test_acc_at_best_val` | validation accuracy (%) and the test accuracy of the checkpoint with the best validation accuracy up to this epoch |
| `train_loss` | mean training loss of the epoch |
| `lr_multiplier` | learning rate relative to its maximum, reconstructed from the recipe |
| `cumulative_energy_J` | GPU-board energy since the start of the run (trapezoidal integral of NVML power) |

## Protocol

* **Loss.** A run violates if `A_T - A_tau > eps` (eps = 0.5 pp), where `A_t` is the test accuracy of the validation-selected
  checkpoint (test data are never used to choose a checkpoint). A job's loss is the fraction of its seeds that violate: the
  probability that one deployed run of that job violates.
* **Certification.** Fixed-sequence testing over an ordered rule parameter with Hoeffding-Bentkus p-values (beta = 0.05); the last
  certified setting is deployed, none certified means the run goes to completion.
* **Evaluation.** 5-fold cross-fitting stratified by (domain, schedule family), repeated; every job is scored with a setting
  calibrated on other folds. Report realised risk, mean per-job saving, and the **energy-weighted saving** (total joules saved
  over total joules), which is what a data centre sees. Compare rules at **matched realised risk**, not at a hand-picked setting.

```python
from bench import load, evaluate
runs = load("runs.csv.gz")                     # development corpus
evaluate(runs, "patience_fraction", rho=0.20)
```

Add a rule by writing `rule(run, theta) -> stop_epoch` and registering `RULES["name"] = (rule, thetas)` with `thetas` ordered from
conservative to aggressive.

## Reference results (out-of-fold, rho = 0.20, beta = 0.05, development corpus)

| Rule | Risk | Mean saved (%) | Energy-weighted (%) |
|---|---|---|---|
| Patience, fixed p | 0.106 | 48.9 | 4.2 |
| Patience, p proportional to T, min-delta 0.1 | 0.104 | 52.8 | 31.8 |
| EcoStop (curve score + gates) | 0.115 | 50.4 | 30.9 |
| Fixed budget fraction | 0.105 | 22.7 | 21.8 |

`bench.py` reproduces the patience and fixed-budget rows to within 0.2 pp (different fold seeds).

## Caveats

* One GPU (RTX 4080 Laptop). Energy is GPU-board energy, not whole-system energy or emissions.
* The NVML power value updates about every 0.5 s. Epochs shorter than that (tabular jobs) have per-epoch energy close to duration
  times a held power value; the energy-weighted metric is dominated by vision and NLP and is unaffected.
* Tabular jobs are 65% of the jobs but a tiny share of the energy.
* Five identical repeats of one job differ by 1.95% in energy (coefficient of variation) and, on a very small dataset, by 68-82%
  in best validation accuracy: GPU training is not deterministic.
* A certificate describes the population it was calibrated on. The fresh set shows it failing for step-decay and one-cycle
  schedules absent from calibration at rho = 0.20.

## Citation and licence

To be added on release (repository URL, DOI, licence).

## Notes on the `c1` set (added 2026-10-14)
* 41 Hugging Face text-classification datasets, none in the development or fresh sets; two recipes (ELECTRA-small, 8 epochs; BERT L6-H512,
  12 epochs), 3 seeds each. Fine-tuning with linear decay; energy is NVML GPU-board energy.
* The study was pre-registered (dataset list, recipes, rules and analysis hash-locked before training). Three runs whose energy integral spanned a
  Windows Modern Standby suspend (an epoch of 510 s to 12.4 h against a median of 10-12 s) were quarantined by an outcome-blind rule
  (longest epoch > 5x the run median and > 60 s) and re-run with the same seed; the exported set contains only the valid re-runs.
* The machine was shared with other desktop workloads during part of the run (see the pre-run GPU state in the paper's threats section).

