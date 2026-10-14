# EcoStop: benchmark, protocols, code and results

Companion artefact of the paper *EcoStop: Certified Early Stopping for Energy-Efficient Training*.

> Repository: https://github.com/habibullahmanzoor/ecostop. Licence: MIT (see `LICENSE`). A versioned archive with a DOI will be deposited on Zenodo (`[DOI]`).

## About

**EcoStop** is a framework and benchmark for *certified early stopping*: stopping a training run as soon as it is safe to,
so that energy is saved while a statistical guarantee bounds how often final accuracy is lost.

* **Problem.** Early stopping is the cheapest lever for cutting training energy, but stopping rules are usually compared at
  hand-picked settings that hide the accuracy they risk.
* **Method.** Stopping rules (learning-curve score, calibrated patience, LC-PFN) are replayed on measured training runs and
  certified with Learn-then-Test fixed-sequence testing and betting p-values, with Bonferroni-corrected rule selection and a
  scope card that abstains outside the calibration support.
* **Data.** EcoStop-Bench: 923 GPU-metered (NVML) training runs, 271 jobs, 37,090 epoch rows across vision, tabular and NLP
  fine-tuning, including a pre-registered confirmatory study on 41 new datasets.
* **Findings.** Certification beats standard certification at the same guarantee, but calibrated patience is as good as the
  learning-curve rule once risk is matched; the paper reports ties and negatives alongside the wins.

**Paper:** *EcoStop: Certified Early Stopping for Energy-Efficient Training* (Althobaiti, Manzoor, Manzoor, Ramzan).
**Code and data availability:** everything needed to recompute every number in the paper is in this repository.
**Licence:** MIT.

## Contents

| Folder | What it holds |
|---|---|
| `ecostop_bench/` | **EcoStop-Bench**: `runs.csv.gz` (923 NVML-metered training runs, 271 jobs, 37,090 epoch rows), a self-contained loader and matched-risk protocol (`bench.py`), the data dictionary (`README.md`) and the exporter. Sets: `dev`, `fresh`, `c1`, `dev_extra`, `regime_freezethaw`, `repeat`. |
| `protocols/` | Frozen, hash-locked protocols: `b1_protocol.json` (fresh set), `c1_protocol.json` and `c1_manifest.json` (confirmatory study, SHA-256 `F11AA18D...C814EC07`), the written analysis plans, and a log of every deviation and decision (`DEVIATIONS_AND_DECISIONS.md`). |
| `raw/` | Per-run CSV and metadata of the confirmatory study (`c1/`, including pre-run GPU state; quarantined runs in `c1/quarantine/`), the fresh set (`b1/`), the small-language-model pilot (`slm_pilot/`), GPU health logs and supervisor logs. |
| `code/` | All analysis code (`echostop/*.py`), `requirements.txt`, `matplotlibrc`. |
| `results/` | Every result table as CSV/JSON that the paper's numbers and figures are computed from. |
| `figures/` | The paper's figure PDFs (fonts embedded). |
| `paper/` | LaTeX source and bibliography. |
| `repo/` | A repository-layout copy (code, raw logs, small trace caches, frozen protocols) so that the analysis scripts run unchanged: `cd repo` and run the commands below. |
| `SHA256SUMS.txt` | SHA-256 of every file in this release. |

## Reproducing the headline results (CPU only, a few minutes each)

Run from `repo/` (it holds the logs and caches the scripts read):

```
cd repo
export USE_TF=0 CUDA_VISIBLE_DEVICES=""
pip install -r requirements.txt
python -m echostop.c1_framework_oof      # pooled out-of-fold framework comparison, validity coverage
python -m echostop.subgroup_wins         # regimes
python -m echostop.pool_vs_domain        # pooled versus per-domain certification
python -m echostop.break_even            # cost of failure
python -m echostop.crc_extension         # calibrator comparison
python -m echostop.c1_eval --no-pfn      # pre-registered C1 evaluation (refuses to run if a locked file changed)
python -m echostop.c1_crossfit           # cross-fitted C1 estimate (post hoc)
```

The benchmark can be used on its own: `from ecostop_bench.bench import load, evaluate` (see `ecostop_bench/README.md`).
The training scripts (`echostop/b1_train.py`, `c1_train.py`, `slm_train.py`) require a CUDA GPU and the NVML energy meter
(`echostop/energy.py`); their runs are already included under `raw/`.

## Notes a reader should know

* Energy is NVML board energy of one laptop GPU (RTX 4080 Laptop), sampled every 20 ms; it is not wall-plug system energy.
* The confirmatory study had interruptions (a 12.5 h Windows Modern Standby, four power-offs, a relaxed idle-GPU guard); all are
  documented in the paper and in `protocols/DEVIATIONS_AND_DECISIONS.md`. Three runs were quarantined by an outcome-blind rule and re-run.
* Freezes were made by local file hashing and are not independently time-stamped; depositing `c1_protocol.json` now is
  recommended.

## Citation

See `CITATION.cff`. URL: https://github.com/habibullahmanzoor/ecostop. DOI: `[DOI]`.
