# C1: confirmatory test of EcoStop-SR on short-schedule NLP fine-tuning. Analysis plan

Status: written and hash-locked (c1_freeze.py -> c1_protocol.json) BEFORE any C1 training run exists. The code that implements
this plan (c1_rules.py, c1_eval.py), the registry (c1_registry.py), the data rules (c1_data.py), the dataset list and sizes
(c1_manifest.json) and the method code the rules depend on are all hashed. The evaluation refuses to run if any of them changed.

## 1. Why, and what is already known
EcoStop's published rule has two fixed floors in its stopping gate (a flat count of at least 2 epochs, confirmed on 2
consecutive epochs). On 10-epoch fine-tuning this caps its saving (33.7% energy-weighted at any threshold). EcoStop-SR makes
both floors schedule-relative (c1_rules.py, echostop/fv2_srgates.py). The rule was found after the 142 development jobs and
the 35 B1 jobs had been analysed. On the development jobs it beats calibrated patience by +5 to +10 pp at matched risk with
intervals above zero; on the 8 B1 NLP jobs its point estimate is higher but the interval includes zero. Those jobs cannot
serve as confirmation. C1 is a new test: 41 datasets that no rule was designed or tuned on, two model families and two
schedule lengths (8 and 12 epochs) absent from the calibration pool, and a dataset-clustered analysis.

## 2. Data
Jobs = dataset x arm (A: ELECTRA-small, 8 epochs, lr 3e-4; B: BERT L6-H512, 12 epochs, lr 5e-5), 3 seeds each, as in
c1_registry.py. Datasets that passed the size rules in c1_data.py (41 of 43; 2 dropped by rule: fin_class, cf_corporate) are
trained; no dataset is added or removed afterwards. A job enters the analysis iff all 3 seeds completed all epochs. Jobs that
fail for technical reasons (crash, out-of-memory) are re-run once; a job still incomplete is dropped and listed in the
report. No exclusion is made on any result.

## 3. Quantities
Per run: loss = 1[A_T - A_tau > 0.5 pp] with A the test accuracy of the validation-selected checkpoint; saved energy =
E_T - E_tau (NVML joules). Per job: risk = share of seeds with loss = 1; energy saved = sum over seeds.
Energy-weighted saving (EW) of a setting = 100 x total energy saved / total energy of the full runs, over the jobs analysed.
The frontier EW_X(r) of a rule X is the largest EW among X's settings (including never stop) whose mean job-level risk
over the analysed jobs is <= r.

## 4. Primary hypothesis and test (one primary comparison)
H1: On short-schedule NLP fine-tuning, EcoStop-SR saves more energy than calibrated schedule-relative patience (patience
max(1, round(fT)) with a minimum improvement of 0.1 pp) at the same realised risk.
Statistic: Delta = EW_SR(0.10) - EW_patience(0.10).
Why 0.10: certification at the paper's main target rho = 0.15 delivers a realised risk of about 0.10 on the development
corpus (0.099 for EcoStop-SR, 0.084 for patience).
Inference: bootstrap over datasets (10,000 resamples, seed 0; both arms of a dataset move together), with each rule's setting
re-optimised inside every resample. Two-sided 95% percentile interval.
Decision: H1 is supported iff the lower limit of the interval is above 0. Otherwise it is reported as not supported, whatever
the point estimate. The paper's claim about EcoStop's rule follows this decision (section 8).

## 5. Secondary analyses (descriptive; they do not decide H1)
S1  The same frontier difference at risk caps 0.05, 0.10, 0.15, 0.20 for every rule (EcoStop-SR, EcoStop with the paper's
    gates, LC-PFN with SR gates, patience with and without minimum improvement, fixed-epoch patience, fixed budget fraction).
S2  Per arm (A: 8 epochs, B: 12 epochs).
S3  Frozen operating points, no tuning on C1: for every rule the setting certified on the 177 existing jobs (Hoeffding-Bentkus
    and betting bound, rho = 0.15 and 0.20, beta = 0.05, pooled) and the setting whose empirical risk is <= 0.10 or <= 0.15 on
    the 35 existing NLP jobs (fixed-sequence). Realised risk and EW on C1, with dataset-bootstrap intervals for differences
    from calibrated patience. A certified setting whose realised C1 risk exceeds its rho is reported as a validity failure.
S6  Sensitivity of the primary contrast: excluding jobs that barely learn (final test accuracy within 3 pp of the majority
    class), thinning EcoStop-SR's gamma grid to every other value, leaving each dataset out in turn.
S7  GPU state before each run (utilisation, board power) to document sharing.

## 6. What is not done
No dataset, job, rule, setting, risk level or exclusion is chosen after seeing C1 results. If the evaluation code has a bug it
is fixed in a new, separately hashed file and both versions are reported.

## 7. Power
Not computed in advance: the energy weights are concentrated and the development bootstrap is wide (n = 27 NLP jobs). We
expect roughly 50-70% probability of rejecting at the effect seen on development; a non-significant result is a possible
outcome and will be reported as such.

## 8. What the paper says afterwards
If H1 is supported: EcoStop-SR is presented as the final stopping rule with C1 as its pre-registered confirmation (effect and
interval as measured), together with S1-S3. If not supported: the paper reports the C1 result, keeps the framework
contribution (certification, betting bound, scope card, rule selection), and does not claim that EcoStop's rule beats
calibrated patience.
