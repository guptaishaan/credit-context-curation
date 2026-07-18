"""Run TabFM on the exported curated contexts and write its AUCs for the cross-model comparison.

Reads outputs/contexts/*.npz (Xc, yc, Xt, yt, cols) produced by the credit project's cross_model.py,
fits TabFM on each context, predicts on the exported test subsample, and appends a row per context.
Runs in the isolated TabFM (Py3.11) env.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import tabfm
from tabfm import TabFMClassifier

CTX = Path("/ccn2/u/ishaangp/projects/credit_context_curation/outputs/contexts")
OUT = Path("/ccn2/u/ishaangp/projects/credit_context_curation/outputs/stats/tabfm_contexts.csv")
DATASETS = ["freddie_mac", "lending_club"]
SPLITS = ["fm_early", "fm_late", "lc_pre_crisis", "lc_post_crisis", "lc_recent"]


def _parse(stem):
    seed = int(stem.split("_s")[-1])
    rest = stem[: stem.rfind("_s")]
    dataset = next(d for d in DATASETS if rest.startswith(d))
    rest = rest[len(dataset) + 1:]
    split = next(s for s in SPLITS if rest.startswith(s))
    strategy = rest[len(split) + 1:]
    return dataset, split, strategy, seed


def _positive_proba(clf, Xt):
    proba = np.asarray(clf.predict_proba(Xt))
    if proba.ndim == 2 and proba.shape[1] == 2:
        classes = list(getattr(clf, "classes_", [0, 1]))
        return proba[:, classes.index(1)] if 1 in classes else proba[:, -1]
    return proba.ravel()


def main():
    import csv
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = tabfm.tabfm_v1_0_0_pytorch.load(device=device)
    print(f"MODEL LOADED on {device}", flush=True)
    clf = TabFMClassifier(model=model)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    handle = OUT.open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=["dataset", "split", "strategy", "seed", "model", "roc_auc"])
    writer.writeheader()
    n = 0
    for path in sorted(CTX.glob("*.npz")):  # all 90 exported contexts (3 seeds x 5 splits x 6 strategies)
        dataset, split, strategy, seed = _parse(path.stem)
        d = np.load(path, allow_pickle=True)
        cols = [str(c) for c in d["cols"]]
        Xc = pd.DataFrame(d["Xc"], columns=cols)
        yc = d["yc"].astype(int)
        Xt = pd.DataFrame(d["Xt"], columns=cols)
        yt = d["yt"].astype(int)
        if len(np.unique(yc)) < 2:
            p = np.full(len(yt), float(yc[0] == 1))
        else:
            clf.fit(Xc, yc)
            p = _positive_proba(clf, Xt)
        auc = roc_auc_score(yt, p) if len(np.unique(yt)) == 2 else np.nan
        writer.writerow({"dataset": dataset, "split": split, "strategy": strategy, "seed": seed,
                         "model": "tabfm", "roc_auc": auc})
        handle.flush()
        n += 1
        print(f"[s{seed}][{dataset}][{split}][{strategy}] tabfm_auc={auc:.3f}", flush=True)
    handle.close()
    print("wrote", OUT, "rows", n)


if __name__ == "__main__":
    main()
