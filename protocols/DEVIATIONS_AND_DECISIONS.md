# Deviations and decisions log

Neutral record of every departure from the pre-specified protocols and of the main analysis decisions, with dates (2026) and their
effect. Protocol hashes: `b1_protocol.json` (fresh set, frozen 2026-10-07), `c1_protocol.json` (confirmatory study C1, SHA-256
`F11AA18DE6D21EB37660917EEF1896415C5B297BC56FC851515CE352C814EC07`). Freezes are local file hashes and were not deposited with an
external time-stamping service.

## Confirmatory study C1 (41 datasets x 2 recipes x 3 seeds = 246 runs)

| Date | Event | Handling | Effect on the analysis |
|---|---|---|---|
| 10-12 | Arm A learning rate changed from 1e-4 to 3e-4 after a smoke test on one dataset showed it still improving at epoch 8 | Changed before the freeze, before any stopping rule was evaluated on C1 data | None |
| 10-12 | Dry run of the evaluation on 8 existing fresh NLP jobs found a stale label list in the frozen-threshold table | Fixed before the freeze | None |
| 10-12 | Four unclean power-offs of the laptop during training | Training resumed from completed runs only; locked-file hashes verified before every resumption; structure-only integrity check after each | None; runs finished before the interruption are unaffected |
| 10-12 | Idle-GPU guard (wait while GPU utilisation > 10%) stalled training for 43 min because desktop applications held reported utilisation near 19% at ~6 W | After 158 of 246 runs the guard was relaxed to also require board power >= 25 W (`echostop/c1_run.py`); every later run records the raw and the used utilisation | Changes only when a run starts |
| 10-12/13 | Windows Modern Standby suspended the machine (once for 12.5 h) | A keep-awake request was added to the trainer; epochs whose NVML integral spanned a suspension were detected by an outcome-blind rule | See next row |
| 10-13 | Three finished runs had an epoch of 510 s, 5,325 s and 44,754 s against a median of 10-12 s | Rule written before the evaluation: longest epoch > 5x the run median and > 60 s is measurement-invalid; the runs were moved to `raw/c1/quarantine` and re-run with the same seed; the rule was re-applied to all runs before evaluation (0 further flags) | Originals retained, not used |
| 10-13 | Two runs failed with `NVMLError_Unknown` after a resume | Re-run by the retrying supervisor | None |
| 10-13 | Interim look at 69 of 82 jobs at the request of an author | Disclosed; no change to data, rules or analysis afterwards | Final evaluation run once on all 246 runs |
| 10-13 | Framework check (`c1_framework_plan.md`) specified after the interim look | Reported as a held-out check, not a blind pre-registration | Its validity criterion was missed at rho = 0.20 (0.203 against 0.20) |
| 10-14 | Pre-registered interval found not to be centred on the point estimate (bootstrap re-selects the operating point in every resample) | Reported with its cause; a cross-fitted matched-risk estimate (`c1_crossfit.py`) added as a post hoc sensitivity analysis | The pre-registered result (H1 not supported) is unchanged |

## Other decisions

* EcoStop-SR (schedule-relative gate floors) was devised after the 142 development jobs had been analysed and is therefore tested
  on C1, not on the development data.
* Consensus rules (four variants), per-domain certification of the curve rule, reference-run stopping and a certified early-cooldown
  pilot were tried and did not improve on calibrated patience; they are reported or noted in the paper.
* The small-language-model pilot (Qwen2.5-0.5B, LoRA, `SLM_PLAN.md`) was stopped after 8 complete runs and one partial run because
  its first gate (validation-pass energy <= 10% of a run at quarter-epoch frequency) failed (13.4% on short text, 30% on long text).
* The regime map (vision/tabular versus NLP) and all subgroup intervals are post hoc and not corrected for multiplicity.
