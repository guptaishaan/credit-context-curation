# Context Curation for In-Context Credit Risk Models

This project asks a practical question about credit-risk prediction: when TabPFN can inspect only a small labeled context rather than a full training set, which past loan applications should it see? It compares six ways of selecting that context across time-separated Lending Club and Home Credit data, including sharp economic shifts, and compares them with a conventional XGBoost model trained on every available historical row. The goal is to learn whether selecting examples that resemble the current economic environment can close the usual data advantage of a full trained model.

## Setup and data

1. Download the Lending Club Kaggle dataset at [wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club). Extract the CSV containing `issue_d` and `loan_status`, rename it `lending_club.csv`, and place it at `data/lending_club.csv`.
2. Join/download the [Home Credit Credit Risk Model Stability competition](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability), obtain `train_base.csv`, rename it `home_credit_base.csv`, and place it at `data/home_credit_base.csv`.
3. On a GPU compute node with Python 3.9 or newer, create and activate an environment, then install the pinned packages:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. Run the full two-dataset study on a GPU node:

   ```bash
   python src/run_experiment.py
   python src/visualize.py
   ```

   On Slurm clusters, submit the included GPU job instead:

   ```bash
   sbatch jobs/run_gpu_experiment.sbatch
   ```

5. Iterate on one dataset only with `python src/run_experiment.py --dataset lending_club` or `python src/run_experiment.py --dataset home_credit`.
6. Check the smallest end-to-end configuration with `python src/run_experiment.py --dry-run`. It runs Lending Club's pre-crisis split, random 128-row TabPFN context, and the XGBoost baseline. The supplied runner requires CUDA deliberately, so run this command on a GPU compute node rather than the login node.

The loaders never download data. They provide clear file-location errors instead, so data access remains explicit and reproducible. Their categorical conversion happens on load, while missing-value medians, sparse-column filtering, and quantile normalization are fitted separately on each pool period; this prevents future test years from influencing preprocessing.

## Context strategies

### Random

Random is the clean control group. It draws past loans uniformly from the available pool, so it shows what happens when we have no opinion about which examples matter. Any improvement from another method is measured against this ordinary sample. The same seed gives the same context every time.

### Most recent

Most recent takes the latest available applications before the test period. The idea is that lending rules, interest rates, and borrower behavior may change over time, so old examples can become stale. It is simple enough that a lender could use it without extra modeling. It may be especially useful when the economy has just changed direction.

### Class balanced

Class balanced asks TabPFN to see roughly equal numbers of defaults and non-defaults. Credit defaults are uncommon, so a random context can contain few positive examples. Seeing more defaults may help the model understand their pattern and may make probabilities more calibrated. The tradeoff is that the context's default rate no longer looks like the real population rate.

### Economically similar

Economically similar first compares each historical pool year with the held-out period using average interest-rate conditions, default rate, and an underwriting-quality proxy. It then draws examples only from the closest historical year. This is like choosing an old case study that has the most similar economic weather. Test-period statistics are used only to choose existing context rows, not to fit a predictive model, as defined by this experiment.

### High confidence

Despite its short name, this strategy keeps the examples on which a lightweight model is least sure. It trains a 100-tree LightGBM model on the historical pool, then chooses probabilities closest to 50%. Those borderline loans may teach TabPFN the decision boundary more clearly than very obvious approvals or defaults. The LightGBM selection model runs on the GPU.

### Diverse

Diverse uses GPU K-Means to spread selected loans across the feature space. It takes one real loan close to each cluster center, so repeated near-identical applicants do not crowd out unusual cases. This strategy does not try to guess which examples are recent or hard; it tries to cover many kinds of borrowers. It is a strong broad-coverage baseline for the more targeted methods.

## Outputs

`outputs/results.csv` is append-written after every baseline or TabPFN evaluation and contains dataset, split, strategy, context size, ROC-AUC, average precision, Brier score, expected calibration error, and model runtime. The runner removes the prior results file at the start of a new invocation, preventing accidental duplicate rows, and asserts the expected row count (125 for a full run) and finite metrics on completion. Compressed prediction arrays are retained under `outputs/predictions/` solely to make the calibration figure reproducible.

`python src/visualize.py` writes five 150-dpi PNGs to `outputs/figures/`: strategy AUC comparisons, Lending Club scaling curves, reliability diagrams, cross-split rank heatmap, and TabPFN's AUC gap to XGBoost.

## What to expect

The most plausible result is that `economically_similar` and `most_recent` improve over `random` on the sharpest regime shifts, especially Lending Club's pre-crisis split and Home Credit's pre-COVID-to-COVID split. `class_balanced` may improve calibration without necessarily improving AUC, while `diverse` should be a strong general-purpose baseline. XGBoost will likely lead in the smoother post-crisis splits, where using more data matters most. The interesting headline outcome would be a carefully selected TabPFN context matching or beating full-pool XGBoost under the strongest temporal shift.

## Limitations and future work

TabPFN is capped at 1,024 context rows, so it cannot directly consume all historical cases. XGBoost is intentionally fixed at reasonable default-like hyperparameters; a tuned XGBoost model would probably widen its advantage. The natural next step is to rerun every combination under several independent seeds and report uncertainty intervals rather than one deterministic estimate. The economic-similarity method also uses held-out aggregate labels to choose a context, which is valid for this specified context-selection experiment but would require care in a real-time deployment setting.
