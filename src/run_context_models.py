"""Trained-on-context baselines (logistic regression, XGBoost) over the SAME curated contexts the
in-context foundation models see. Reuses run_experiment's selection so contexts match exactly, and
writes rows with a ``model`` column for the cross-model comparison.
"""
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from splitter import get_split
from evaluate import compute_metrics, append_result
from run_experiment import _load_dataset, _macro_indices, _select

ROOT = Path(__file__).resolve().parents[1]


def _fit_predict(model_name, Xc, yc, Xt):
    """Fit one trained-on-context model and return test default probabilities and runtime."""
    start = time.perf_counter()
    if yc.nunique() < 2:  # degenerate single-class context: emit its constant probability (chance AUC)
        return np.full(len(Xt), float(yc.iloc[0] == 1)), time.perf_counter() - start
    if model_name == "lr_context":
        model = LogisticRegression(max_iter=1000).fit(Xc, yc)
    else:
        model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, tree_method="hist", device="cpu",
                              eval_metric="logloss", random_state=42, n_jobs=4).fit(Xc, yc)
    return model.predict_proba(Xt)[:, 1], time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--results-path", required=True)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config.yaml").open())
    results_path = Path(args.results_path)
    results_path.unlink(missing_ok=True)
    for dataset in config["datasets"]:
        raw_X, raw_y, raw_dates, preprocessor = _load_dataset(dataset, config["datasets"][dataset], args.seed)
        for split in config["datasets"][dataset]["splits"]:
            Xp_raw, yp, dp, Xt_raw, yt, _ = get_split(raw_X, raw_y, raw_dates,
                                                      config["datasets"][dataset]["splits"][split])
            Xp, Xt = preprocessor(Xp_raw, Xt_raw)
            macro_indices = _macro_indices(dataset, Xp.columns)
            for strategy in config["strategies"]:
                for n_context in config["n_context_values"]:
                    Xc, yc = _select(strategy, Xp, yp, dp, n_context, args.seed, Xt, yt, macro_indices)
                    if len(Xc) > 1024:
                        chosen = np.random.default_rng(args.seed).choice(len(Xc), 1024, replace=False)
                        Xc, yc = Xc.iloc[chosen], yc.iloc[chosen]
                    for model_name in ("lr_context", "xgb_context"):
                        probs, runtime = _fit_predict(model_name, Xc, yc, Xt)
                        row = {"dataset": dataset, "split": split, "strategy": strategy,
                               "n_context": int(len(Xc)), "n_requested": int(n_context), "seed": int(args.seed),
                               "model": model_name, **compute_metrics(yt, probs, runtime)}
                        append_result(results_path, row)
                    print(f"[{dataset}][{split}][{strategy}] n={len(Xc)} lr+xgb done", flush=True)


if __name__ == "__main__":
    main()
