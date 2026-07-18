# Which Examples Should an In-Context Tabular Model See? Curation Strategies Under Temporal Drift in Credit Risk

*Workshop draft — figures in `outputs/figures/`, tables in `outputs/stats/`. Results/Abstract filled from the cross-model sweep.*

## Abstract
We study how the *choice* of in-context examples affects an in-context tabular foundation model (TabPFN)
on credit-default prediction under temporal distribution shift. Across 2 real datasets (Lending Club,
Freddie Mac) and 5 time-separated splits, we compare 6 curation strategies, and we compare the in-context
model against a logistic regression and an XGBoost trained on the *identical* curated context, and against
tuned/fixed XGBoost trained on the full historical pool — with up to 5 seeds and paired-bootstrap
confidence intervals. We find: (1) **data efficiency** — a curated 1,024-row context reaches statistical
*parity* with a *tuned* full-pool XGBoost trained on 13k–100k rows (and beats an untuned one), but does not
significantly exceed it; (2) **the in-context prior helps** — on viable contexts TabPFN beats a logistic
regression on the same context in 85% of cells (+0.016 AUC) and XGBoost-on-context by +0.046; (3) **common
heuristics fail predictably** — uncertainty sampling (`high_confidence`) is *anti-predictive across every
model* (mean AUC 0.35), a selection-induced sign reversal, and diversity sampling collapses to a single
class under extreme imbalance. The failures are model-agnostic; the prior-advantage is not. Both
reproduce on a *second* in-context foundation model (Google TabFM): TabPFN and TabFM agree to within
0.016 AUC per cell (correlation 0.98), `high_confidence` inverts for both, and TabFM likewise beats a
logistic regression on the identical context.

## 1. Introduction
In-context tabular foundation models (TabPFN [Hollmann et al.], TabFM [Google 2026]) predict by conditioning
on a small labeled context rather than training. When the historical pool exceeds the context budget (TabPFN
caps at ~1k rows), *which* examples to include becomes a design choice. Prior work selects context to maximize
i.i.d. accuracy (per-query retrieval: LocalPFN, TabDPT, MixturePFN) or to compress (coreset/k-means/distillation).
We instead ask a deployment-flavored question: **under temporal distribution shift, which curation strategy
should a practitioner use, and how do common heuristics fail?** Credit-default prediction is an ideal testbed —
it has genuine regime shifts (financial crisis, COVID) and extreme class imbalance.

**Contributions.** (i) A controlled comparison of 6 curation strategies across 2 real credit datasets and 5
time-separated splits, with multi-seed paired-bootstrap significance and a *fair, tuned* full-data baseline.
(ii) A data-efficiency result: parity, not superiority, vs. tuned XGBoost. (iii) A characterization of two
predictable failure modes of popular heuristics under imbalance/drift, shown to be largely model-agnostic.

## 2. Setup
**Datasets & splits.** Lending Club (2007–2018; pre-crisis, post-crisis, recent splits) and Freddie Mac
Single-Family sample (2019–2021; pre-COVID→COVID `fm_early`, `fm_late`). Default label: charged-off/defaulted
(LC) or ever-180+-days-delinquent / credit-event zero-balance (Freddie). Preprocessing (median impute,
quantile-normal) is fit on each pool period only — no test leakage. Freddie default rates: 3.4%→0.9% across the
COVID window (forbearance-suppressed — see Limitations).

**Curation strategies (context budget n ∈ {128,256,512,1024}).** random, most_recent, class_balanced,
economically_similar (nearest pool-year by rate/default/underwriting proxies), high_confidence (LightGBM
decision-boundary rows), diverse (GPU k-means representatives).

**Models.** In-context FM: TabPFN v2. Trained-on-same-context: logistic regression, XGBoost. Full-pool
baselines: XGBoost fixed and tuned (cross-validated randomized search on the pool). A second in-context FM (Google TabFM v1, 2026) is
run on the *identical* exported contexts (`outputs/contexts/`) as a cross-model check. All curated models
see the identical context per (split, strategy, budget, seed).

**Protocol.** 5 seeds; metrics ROC-AUC, average precision, Brier, ECE. Significance via paired bootstrap
(1,000 draws) over shared test rows on the AUC gap. GPU-only model computation.

## 3. Results
### 3.1 Data efficiency: parity with a tuned full-data model (Fig. 06)
A curated 1,024-row context, with no training, matches a *tuned* full-pool XGBoost on the three viable
splits despite using orders of magnitude less data: fm_late 0.793 vs 0.791 (XGBoost trained on 100k rows),
fm_early 0.765 vs 0.755 (50k), lc_post_crisis 0.689 vs 0.685 (13.5k). It underperforms only on the largest,
most stationary pool (lc_recent 0.673 vs 0.687, 150k rows).

### 3.2 But no significant win over a fair baseline (gaps.csv)
Against the tuned baseline the best strategy's AUC gap is not significant on any split (paired bootstrap,
1,000 draws): fm_early +0.010 [-0.009,+0.031], fm_late +0.002 [-0.011,+0.012], lc_post_crisis +0.004
[-0.006,+0.018], and a significant *loss* on lc_recent (-0.015 [-0.023,-0.006]). The apparent wins over the
*untuned* baseline (fm_early +0.041, fm_late +0.022, lc_post_crisis +0.020, all significant) vanish once the
baseline is tuned. Honest headline: **parity, not superiority.**

### 3.3 The in-context prior beats training on the same context (Fig. 07)
On viable contexts (excluding near-chance lc_pre_crisis and degenerate high_confidence), TabPFN outperforms
a logistic regression fit on the *identical* 1,024-row context in 85% of (split, strategy) cells (+0.016 AUC
mean) and XGBoost-on-context by +0.046. The pretrained prior extracts more from a small curated context than
models trained on it — an in-context-specific advantage. **This is not a TabPFN artifact (Fig. 08):** a
second foundation model, Google TabFM, run on the identical contexts tracks TabPFN to within 0.016 mean
AUC per cell (correlation 0.98) and likewise beats logistic-regression-on-context on 65% of viable cells
(+0.003 mean), reproducing the in-context-prior advantage on a distinct FM.

### 3.4 Predictable, model-agnostic failure modes
**Uncertainty sampling inverts predictions.** `high_confidence` (context = LightGBM decision-boundary rows)
selects a high-risk slice (79.7% default rate vs 3.3% pool on fm_early) in which the sign of the feature–
label relationship flips (credit_score↔default correlation -0.125 globally vs +0.172 in context). Every
model inherits the inversion (mean AUC 0.35 across TabPFN/TabFM/LR/XGBoost on higher-signal splits;
TabFM 0.37 < 0.5 confirms it on a second FM) — selection bias, not a model quirk. **Diversity sampling collapses** to a single class on rare-event Freddie splits (0
positives in a 128-row k-means-representative context), yielding chance AUC. Otherwise class_balanced and
most_recent rank highest; economically_similar equals random when a split has one pool year.

## 4. Related Work
In-context tabular models: TabPFN(-v2), TabICL, ConTextTab. Context/support selection: per-query retrieval
(LocalPFN, TabDPT, MixturePFN), coreset/k-means/data-distillation compression, fairness-aware selection. Active
learning (uncertainty sampling) and coreset selection more broadly. We differ by studying *temporal drift*, a
*strategy comparison with failure modes*, and *in-context-FM vs. trained-on-context* behavior on real credit
data, rather than proposing a new selection method.

## 5. Limitations
Single label definition per dataset; two datasets, both US consumer/mortgage credit. The Freddie COVID shift is
forbearance-confounded (default rates fall due to CARES-Act deferrals, not pure credit improvement). economically_
similar degenerates to random when a split has one pool year. Post-hoc "best strategy" selection is optimistic yet
still yields no significant win. No mechanistic theory — failure modes are characterized empirically (selection
bias / Simpson's-style sign reversal), not proven. Single tuned-baseline budget.
