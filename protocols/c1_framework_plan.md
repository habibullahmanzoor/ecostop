# C1-F: confirmatory test of the EcoStop FRAMEWORK on the C1 jobs (specified 2026-10-13, before the full C1 set is evaluated)

Disclosure: an interim look at 69 of 82 C1 jobs was run on 2026-10-13 for the RULE comparison (c1_eval.py, S1-S3). The
single-rule frozen-threshold numbers in S3 were therefore seen in part. This framework test was specified after that look
and before the full set is evaluated; it is a held-out check on 41 new datasets, not a blind pre-registration. It is
reported as such.

## Question
At the same target rho and confidence, does the EcoStop framework (betting certificate + certified rule selection + scope
card) save more energy than standard Learn-then-Test (Hoeffding-Bentkus p-value, one rule) while keeping realised risk
<= rho?

## Procedure (all calibrated on the 177 existing jobs ONLY, then applied unchanged to C1)
* Baselines: LTT-HB with EcoStop (paper gates), with calibrated patience (p=fT, min-delta 0.1), with LC-PFN + gates.
* Framework: betting bound; certified selection among the same three rules (27 assignments collapse to 3 for a single
  domain); scope card (C1 jobs are all NLP fine-tuning with linear decay: in scope, so abstention is not exercised).
* Targets rho in {0.10, 0.15, 0.20}, beta = 0.05.
* Outcomes per target: realised risk (share of C1 jobs with loss), energy-weighted saving (EW), difference framework
  minus each baseline with a dataset-cluster bootstrap (10,000, seed 0).

## Pre-specified claims
C1 (validity): realised C1 risk of the framework <= rho at all three targets. A realised risk above rho is reported as a
validity failure whichever method it belongs to.
C2 (efficiency, primary): framework EW minus the BEST single-rule LTT-HB EW at rho = 0.15 has a 95% lower limit > 0.
Secondary: the same at rho = 0.10 and 0.20; versus each baseline.

## Decision
C1 and C2 both met -> "the framework beats standard certification on 41 new datasets" is stated as confirmed (held out).
Otherwise it is reported as not confirmed, and the paper keeps the development/fresh evidence with that limitation stated.
