# EcoStop for small language models: plan (2026-10-14, nothing run yet)

## 1. Why small language models, and what we are trying to show
* Fine-tuning holds 51% of the energy in our 259 jobs and 15 of the 20 pp that a hindsight oracle could still save. The
  oracle stops at ~30% of a fine-tuning run; our rules stop at ~55-60%.
* What we know already, and must not repeat: EcoStop's curve rule does NOT beat calibrated patience (held-out: 1 win,
  8 ties, 6 losses). The framework DOES beat standard certification at the same guarantee (pooled 259 jobs: +3.8 to +6.8 pp,
  8/9 intervals above zero; held-out C1: wins at rho 0.15/0.20, loses at 0.10).
* Target claim for the SLM study:
  **"At the same certified accuracy-loss guarantee, EcoStop saves more fine-tuning energy than the early stopping
  practitioners use (Hugging Face `EarlyStoppingCallback`) and than standard Learn-then-Test certification."**
  EcoStop = the framework (certificate + rule selection + scope card + the two new components in section 3), not the curve rule.

## 2. Why SLM fine-tuning suits the framework
* **Weak practitioner baseline.** Hugging Face Trainer stops on uncalibrated patience counted in evaluation calls, and
  default recipes train a fixed 3 epochs. Neither says how often accuracy is lost.
* **Decisions are made per evaluation step, not per epoch.** This is the lever that can reach the 15 pp gap (section 3).
* **A smooth signal is available.** With label log-likelihood scoring, validation loss is continuous and much less noisy
  than 0/1 accuracy on a few hundred examples.
* **Short, cheap runs** on models we already have (Qwen2.5-0.5B and 1.5B are cached).

## 3. What is new in the framework (candidate components, each must earn its place in S1)
| Component | Idea | Why it might win | Risk |
|---|---|---|---|
| C-a Step granularity | Evaluate every 1/4 epoch (about 12-24 decisions per run) and certify patience counted in evaluations | Waiting "2 epochs" becomes waiting "2 quarter-epochs", aimed straight at the fine-tuning gap | Each evaluation costs energy (counted); more evaluations mean more noisy "improvements" |
| C-b Surrogate signal | Detect the plateau on validation LOSS, certify on validation-selected test ACCURACY | Loss is smoother, so the certificate can afford a more aggressive setting | Loss and accuracy can disagree (loss rises while accuracy still improves) |
| C-c Betting certificate + Bonferroni selection | Already built; the selection correction was what kept it valid in the pooled test | Known +2 to +7 pp over standard certification | Costs power at strict rho (lost at rho 0.10 in C1) |
| C-d Scope card for SLMs | Descriptors: model family/size, LoRA vs full, task type, epochs; abstain outside | Needed for any model-shift claim | Descriptors must be fixed before the confirmatory set |

Not pursued again: new curve-rule variants, consensus rules, per-domain certification (all tested, no gain).

## 4. Experimental design
**Models.** Development: Qwen2.5-0.5B (cached) and SmolLM2-360M (ungated, small download). Held-out model for the confirmatory
set: Qwen2.5-1.5B (cached; larger) and, if feasible, SmolLM2-1.7B. Gated models (Llama, Gemma) are avoided.
**Fine-tuning.** LoRA (r = 16, attention + MLP), bf16, AdamW, linear warm-up 5% then cosine/linear decay, max length 256,
2-3 epochs (the Hugging Face default regime), batch 16 (gradient accumulation as needed). Best-validation adapter kept
(adapters are a few MB, so checkpoint selection is free).
**Tasks and metric.** Classification-style tasks with short label words (sentiment, topic, NLI, intent, paraphrase, yes/no QA),
scored by **label log-likelihood ranking** (no free generation: cheap, deterministic, exact accuracy). This keeps the
tolerance meaningful: loss = 1[A_T - A_tau > 0.5 pp] on the test accuracy of the validation-selected checkpoint, as in the paper.
Train capped at 4,000 rows, validation 500, test 2,000. Datasets disjoint from everything used so far and between phases.
**Energy.** NVML board energy for training AND every evaluation; savings are reported NET of evaluation energy, so a finer
evaluation schedule has to pay for itself.
**Baselines (all on the same runs).** Fixed 3 epochs (no stopping); HF `EarlyStoppingCallback` with patience 1, 3, 5 evaluations
(uncalibrated practice); epoch-level calibrated patience; LTT-HB with each single rule (standard certification); LC-PFN.

## 5. Phases, gates and compute (estimates to be replaced by the pilot's measurements)
| Phase | What | Size | GPU estimate | Gate to continue |
|---|---|---|---|---|
| **S0 pilot** | Install `peft` safely, build trainer, measure cost/noise/headroom | 6 tasks x Qwen2.5-0.5B x 2 seeds, eval every 1/8 epoch (sub-sampled later for 1/4, 1/2, 1) | 3-4 h | (a) one run <= 15 min; (b) evaluation energy <= 10% of a run at 1/4-epoch frequency; (c) oracle stops at <= 60% of the run (headroom exists); (d) step-level oracle saves >= 5 pp more than epoch-level after evaluation energy |
| **S1 development** | Calibration population; choose components and freeze framework v3 | ~28 tasks x 2 dev models x 3 seeds (~170 runs) | 25-35 h | Out-of-fold (dataset-grouped): framework v3 beats LTT-HB best single rule at rho 0.15 with CI > 0 AND validity coverage <= 5%; otherwise stop and publish the current paper |
| **S2 confirmatory** | Pre-registered, hash-locked as C1 | ~20 NEW tasks x (1 dev model + held-out Qwen2.5-1.5B) x 3 seeds (~120 runs) | 20-30 h | One primary test decides the claim |
| **Total** | | ~300 runs | **~50-70 GPU-h** (about 4-6 days of machine time) | |

## 6. Hypotheses (final wording frozen after S1, before S2)
* **H-primary (framework):** at rho = 0.15, EcoStop's net energy-weighted saving exceeds the best single-rule LTT-HB on the
  held-out tasks; 95% dataset-cluster bootstrap lower limit > 0; realised risk <= rho.
* H2 (practice): certified stopping dominates HF default patience (lower risk at equal saving, or more saving at equal risk).
* H3 (granularity): step-level certified stopping saves more NET energy than epoch-level certified stopping.
* H4 (surrogate): a loss-signal detector certified on accuracy beats accuracy-signal patience.
* H5 (model shift): with the scope card, validity holds on the held-out 1.5B model, or the card abstains.

## 7. Honest expectations
* H2 is near-certain but not novel on its own.
* H-primary is plausible: the effect so far is +4 to +7 pp; roughly 55-70% chance of a clean pass if S1 confirms it.
* H3/H4 are the novel bets and are genuinely uncertain; the pilot gate (S0-d) tells us cheaply whether H3 has any room.
* If S0 or S1 fails its gate we stop and publish the existing framework paper (section 9), having spent at most 4 or ~35 GPU-h.

## 8. Risks and how they are handled
| Risk | Handling |
|---|---|
| Installing `peft` downgrades torch (happened before with `lcpfn`) | Install with `--no-deps`, check `torch==2.9.1+cu128` afterwards, or use a separate venv |
| Laptop power-offs / Modern Standby (4 incidents during C1) | Keep-awake request, supervised resumable trainer, GPU health log, outcome-blind quarantine rule, no parallel heavy work |
| 12 GB GPU memory for 1.5B | LoRA + bf16 + gradient checkpointing; pilot measures peak memory |
| Evaluation noise on 500 examples | Label log-likelihood scoring; validation size is a pilot variable |
| Forking paths (choosing components after seeing data) | Components chosen only on S1; S2 pre-registered and hash-locked; the C1 interim-look lesson: one evaluation only |
| Exchangeability for certification | S1 tasks drawn by a written rule from a fixed pool; S2 tasks from the same pool, assigned before training |

## 9. What happens to the current paper
Two routes, to be decided by the authors:
1. **Submit the current framework paper now** (framework wins; rule null; C1 reported) and make the SLM study a second paper.
2. **Hold the paper** until S1 passes, then make SLM fine-tuning the main result and the existing 259 jobs the generality section.
Recommendation: decide after S0 (3-4 GPU-h). If the pilot shows real headroom at step granularity, route 2; otherwise route 1.

## 10. Immediate next steps (after approval)
1. Safe `peft` install + torch check.
2. Write the S0 task list and trainer (label log-likelihood eval, step-level logging, NVML energy per training and per eval).
3. Run S0, evaluate the four gates, report before any further compute.
