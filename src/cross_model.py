"""Lean cross-model comparison at the headline budget (n=1024).

For each (seed, split, strategy) it selects the SAME context TabPFN saw, fits logistic-regression and
XGBoost on that context, reuses TabPFN's saved test predictions, and evaluates all three on a common
fixed test subsample (kept small so this stays fast on a contended node). It also exports each context
and the test subsample so TabFM (separate Py3.11 env) can be run on identical inputs and merged.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from splitter import get_split
from run_experiment import _load_dataset, _macro_indices, _select

ROOT = Path(__file__).resolve().parents[1]
N_CONTEXT = 1024
TEST_SUBSAMPLE = 20000
SEEDS = (42, 1, 2)


def _auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def _tabpfn_pred(seed, dataset, split, strategy):
    directory = ROOT / "outputs" / "seeds" / f"pred_s{seed}"
    matches = sorted(directory.glob(f"{dataset}_{split}_{strategy}_*.npz"),
                     key=lambda p: int(p.stem.split("_")[-1]))
    if not matches:
        return None
    d = np.load(matches[-1])
    return d["y_true"].astype(int), d["y_prob"].astype(float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", default=str(ROOT / "outputs" / "contexts"))
    parser.add_argument("--results-path", default=str(ROOT / "outputs" / "stats" / "cross_model.csv"))
    args = parser.parse_args()
    export = Path(args.export_dir); export.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "config.yaml").open())
    rng = np.random.default_rng(0)
    rows = []
    for seed in SEEDS:
        for dataset in config["datasets"]:
            raw_X, raw_y, raw_dates, preprocessor = _load_dataset(dataset, config["datasets"][dataset], seed)
            for split in config["datasets"][dataset]["splits"]:
                Xp_raw, yp, dp, Xt_raw, yt, _ = get_split(raw_X, raw_y, raw_dates,
                                                          config["datasets"][dataset]["splits"][split])
                Xp, Xt = preprocessor(Xp_raw, Xt_raw)
                mi = _macro_indices(dataset, Xp.columns)
                idx = rng.choice(len(yt), min(TEST_SUBSAMPLE, len(yt)), replace=False)
                yt_sub = yt.iloc[idx].to_numpy()
                for strategy in config["strategies"]:
                    Xc, yc = _select(strategy, Xp, yp, dp, N_CONTEXT, seed, Xt, yt, mi)
                    if len(Xc) > 1024:
                        chosen = np.random.default_rng(seed).choice(len(Xc), 1024, replace=False)
                        Xc, yc = Xc.iloc[chosen], yc.iloc[chosen]
                    Xt_sub = Xt.iloc[idx]
                    # trained-on-context models
                    if yc.nunique() < 2:
                        lr_p = xgb_p = np.full(len(idx), float(yc.iloc[0] == 1))
                    else:
                        lr_p = LogisticRegression(max_iter=1000).fit(Xc, yc).predict_proba(Xt_sub)[:, 1]
                        xgb_p = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                                              colsample_bytree=0.8, tree_method="hist", device="cpu",
                                              eval_metric="logloss", random_state=42, n_jobs=4).fit(Xc, yc).predict_proba(Xt_sub)[:, 1]
                    tab = _tabpfn_pred(seed, dataset, split, strategy)
                    tab_auc = _auc(yt_sub, tab[1][idx]) if tab is not None else np.nan
                    for model, auc in (("tabpfn", tab_auc), ("lr_context", _auc(yt_sub, lr_p)), ("xgb_context", _auc(yt_sub, xgb_p))):
                        rows.append({"seed": seed, "dataset": dataset, "split": split, "strategy": strategy,
                                     "n_context": int(len(Xc)), "model": model, "roc_auc": auc})
                    np.savez_compressed(export / f"{dataset}_{split}_{strategy}_s{seed}.npz",
                                        Xc=Xc.to_numpy(float), yc=yc.to_numpy(int),
                                        Xt=Xt_sub.to_numpy(float), yt=yt_sub, cols=np.array(Xc.columns))
                    print(f"[s{seed}][{dataset}][{split}][{strategy}] n={len(Xc)} tabpfn={tab_auc:.3f} lr={_auc(yt_sub,lr_p):.3f} xgb={_auc(yt_sub,xgb_p):.3f}", flush=True)
    pd.DataFrame(rows).to_csv(args.results_path, index=False)
    print("wrote", args.results_path, "rows", len(rows))


if __name__ == "__main__":
    main()
