# EcoStop: benchmark, protocols, code and results

Companion artefact of the paper *Early stopping in neural networks depends on the learning-rate schedule not the curve model*.

> Repository: https://github.com/habibullahmanzoor/ecostop. Licence: MIT (see `LICENSE`).
> Tag `v1.0` is the released artefact. Its benchmark data, frozen protocols and analysis code are the
> ones the paper uses; the manuscript itself has since been revised (see `paper/README.md`).

## About

**EcoStop** is a procedure and benchmark for *certified early stopping*: stopping a training run as soon as it is safe to,
so that energy is saved while a statistical guarantee bounds how often final accuracy is lost.

* **Problem.** Early stopping is the cheapest lever for cutting training energy, but stopping rules are usually compared at
  hand-picked settings that hide the accuracy they risk.
* **Method.** Stopping rules (learning-curve score, calibrated patience, LC-PFN) are replayed on measured training runs and
  certified with fixed-sequence testing and betting p-values, **stratified by declared learning-rate schedule family**, with
  a gate that refuses to stop while a scheduled drop still lies ahead and abstention outside the calibration support.
* **Data.** EcoStop-Bench: 923 GPU-metered (NVML) training runs, 37,090 epoch rows (259 jobs and 897 runs carry the matched-risk analysis) across vision, tabular and NLP
  fine-tuning, including a pre-registered confirmatory study on 41 new datasets.
* **Findings.** (a) The stratum is the schedule, not the domain: within one domain realised risk ranges from 0.158
  (cosine-annealed) to 0.556 (one-cycle), and a domain partition cannot be both valid and useful, whereas stratifying by
  declared schedule family holds the target while keeping 76% of the fresh energy. (b) Refusing to stop while a declared
  drop lies ahead removes all six violations in twelve step-decay runs, at a measured cost in saving. (c) At matched
  certified risk, learning-curve stopping -- including the pretrained LC-PFN -- is indistinguishable from a patience counter
  proportional to the schedule, and a pre-registered confirmatory study found no advantage for the curve rule.

**Paper:** *Early stopping in neural networks depends on the learning-rate schedule not the curve model* (Althobaiti, Manzoor, Manzoor, Ramzan).
**Code and data availability:** everything needed to recompute every number in the paper is in this repository.
**Licence:** MIT.

## Contents

| Folder | What it holds |
|---|---|
| `ecostop_bench/` | **EcoStop-Bench**: `runs.csv.gz` (923 NVML-metered training runs, 37,090 epoch rows (259 jobs / 897 runs analysed)), a self-contained loader and matched-risk protocol (`bench.py`), the data dictionary (`README.md`) and the exporter. Sets: `dev`, `fresh`, `c1`, `dev_extra`, `regime_freezethaw`, `repeat`. |
| `protocols/` | Frozen, hash-locked protocols: `b1_protocol.json` (fresh set), `c1_protocol.json` and `c1_manifest.json` (confirmatory study, SHA-256 `F11AA18D...C814EC07`), the written analysis plans, and a log of every deviation and decision (`DEVIATIONS_AND_DECISIONS.md`). |
| `raw/` | Per-run CSV and metadata of the confirmatory study (`c1/`, including pre-run GPU state; quarantined runs in `c1/quarantine/`), the fresh set (`b1/`), the small-language-model pilot (`slm_pilot/`), GPU health logs and supervisor logs. |
| `code/` | All analysis code (`echostop/*.py`), `requirements.txt`, `matplotlibrc`. |
| `results/` | Every result table as CSV/JSON that the paper's numbers and figures are computed from. |
| `figures/` | The paper's figure PDFs (fonts embedded). |
| `paper/` | Status of the manuscript. The manuscript source is **not published here**; see `paper/README.md`. |
| `repo/` | A repository-layout copy (code, raw logs, small trace caches, frozen protocols) so that the analysis scripts run unchanged: `cd repo` and run the commands below. |
| `SHA256SUMS.txt` | SHA-256 of every file in this release. |

## Reproducing the results (CPU only, a few minutes each)

Run from `repo/` (it holds the logs, result caches and protocols the scripts read):

```
cd repo
export USE_TF=0 CUDA_VISIBLE_DEVICES=""
pip install -r requirements.txt
python -m echostop.family_cert      # Q4: pooled vs per-domain vs per-family certification
python -m echostop.gate_families    # Q5: the declared-drop gate on the three fresh families
python -m echostop.gate_schedule    # Q5: the frozen two-epoch window that leaves the failure intact
python -m echostop.shift_cert       # Q3: reweighting to a declared mix, and the family breakdown
python -m echostop.where_we_win     # the per-stratum comparison behind the instability paragraph
python -m echostop.pool_vs_domain   # pooled versus per-domain certification
python -m echostop.break_even       # Q8: cost of failure
python -m echostop.crc_extension    # Q8: calibrator comparison
python -m echostop.c1_eval --no-pfn # Q9: pre-registered C1 evaluation (refuses to run if a locked file changed)
python -m echostop.c1_crossfit      # Q9: cross-fitted C1 estimate (post hoc)
```

`echostop/subgroup_wins` and `echostop/c1_framework_oof` remain in `code/` because they produce analyses
that appeared in an earlier version of the manuscript; the current paper does not report them.

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

See `CITATION.cff`. URL: https://github.com/habibullahmanzoor/ecostop. Version: tag `v1.0`.
* This README was revised to match the revised manuscript. Two statements it used to make are no
  longer supported and have been removed: that the framework beats standard certification at the same
  guarantee, and a job count that could not be derived from the released sets. The benchmark data,
  the protocols and the code are unchanged.
* The analyses behind Q3 to Q5 were added with this revision, together with the pooled matrices
  (`repo/cache/pool_mats.pkl`) they read and the B5/B5c protocol freezes in `protocols/`.
