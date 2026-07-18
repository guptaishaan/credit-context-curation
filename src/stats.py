"""Aggregate the multi-seed sweep: mean +/- 95% CI per config and paired-bootstrap gap CIs.

Reads per-seed result CSVs and saved prediction vectors written by run_experiment.py --seed ... and
produces (1) a per-configuration aggregate table across seeds and (2) paired bootstrap confidence
intervals, over shared test instances, for the AUC gap between the best TabPFN strategy and each
XGBoost baseline. All comparisons are reported regardless of sign.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, t
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = ROOT / "outputs" / "seeds"
OUT_DIR = ROOT / "outputs" / "stats"
BASELINES = ("xgboost_tuned", "xgboost_full_pool")
BOOTSTRAP = 1000
MAX_BOOTSTRAP_N = 40000  # cap per-draw size on huge test sets to keep the bootstrap fast


def _load_results():
    """Concatenate every per-seed results CSV into one long frame."""
    frames = [pd.read_csv(path) for path in sorted(SEED_DIR.glob("results_s*.csv"))]
    if not frames:
        raise FileNotFoundError("No per-seed results found under outputs/seeds/results_s*.csv")
    return pd.concat(frames, ignore_index=True)


def _ci95(values):
    """Return the half-width of a two-sided 95% t-confidence interval for a small sample mean."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return np.nan
    return float(t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))


def aggregate(results):
    """Mean, std, and 95% CI of each metric per (dataset, split, strategy, requested context)."""
    metrics = ["roc_auc", "average_precision", "brier_score", "ece"]
    rows = []
    for keys, group in results.groupby(["dataset", "split", "strategy", "n_requested"]):
        row = dict(zip(["dataset", "split", "strategy", "n_requested"], keys), n_seeds=len(group))
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_ci95"] = _ci95(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def _fast_auc(y, p):
    """AUC via the rank-sum identity; fast enough to call inside a bootstrap loop."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    return (rankdata(p)[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _pred(seed, dataset, split, strategy):
    """Load the largest-context saved prediction vector for one strategy in one seed."""
    directory = SEED_DIR / f"pred_s{seed}"
    matches = sorted(directory.glob(f"{dataset}_{split}_{strategy}_*.npz"),
                     key=lambda path: int(path.stem.split("_")[-1]))
    if not matches:
        return None
    data = np.load(matches[-1])
    return data["y_true"].astype(int), data["y_prob"].astype(float)


def _bootstrap_gap(seeds, dataset, split, strategy, baseline, rng):
    """Pooled paired bootstrap of the AUC gap (strategy - baseline) over shared test rows."""
    deltas = []
    for seed in seeds:
        a = _pred(seed, dataset, split, strategy)
        b = _pred(seed, dataset, split, baseline)
        if a is None or b is None:
            continue
        y, pa = a
        _, pb = b
        n = len(y)
        size = min(n, MAX_BOOTSTRAP_N)
        for _ in range(BOOTSTRAP // len(seeds)):
            idx = rng.integers(0, n, size)
            deltas.append(_fast_auc(y[idx], pa[idx]) - _fast_auc(y[idx], pb[idx]))
    deltas = np.array(deltas, dtype=float)
    return float(np.nanmean(deltas)), float(np.nanpercentile(deltas, 2.5)), float(np.nanpercentile(deltas, 97.5))


def gaps(results):
    """Per split, bootstrap the best TabPFN strategy's AUC gap to each XGBoost baseline."""
    rng = np.random.default_rng(0)
    seeds = sorted(results["seed"].unique())
    strategies = [s for s in results["strategy"].unique() if not s.startswith("xgboost")]
    peak = results[results["strategy"].isin(strategies)]["n_requested"].max()
    rows = []
    for (dataset, split), group in results.groupby(["dataset", "split"]):
        at_peak = group[(group.n_requested == peak) & (group.strategy.isin(strategies))]
        best = at_peak.groupby("strategy").roc_auc.mean().idxmax()
        for baseline in BASELINES:
            mean, lo, hi = _bootstrap_gap(seeds, dataset, split, best, baseline, rng)
            rows.append({"dataset": dataset, "split": split, "best_strategy": best, "baseline": baseline,
                         "gap_auc_mean": mean, "gap_ci95_lo": lo, "gap_ci95_hi": hi,
                         "significant": (lo > 0) or (hi < 0)})
    return pd.DataFrame(rows)


def main():
    """Write the aggregate table and the gap-significance table, and print a readable summary."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = _load_results()
    agg = aggregate(results)
    agg.to_csv(OUT_DIR / "aggregate.csv", index=False)
    gap = gaps(results)
    gap.to_csv(OUT_DIR / "gaps.csv", index=False)
    print(f"seeds: {sorted(results['seed'].unique())} | rows: {len(results)}")
    print("\n=== Best TabPFN strategy vs XGBoost baselines (paired bootstrap AUC gap, 95% CI) ===")
    for _, r in gap.iterrows():
        star = "  *" if r["significant"] else ""
        print(f"{r.dataset:12s} {r.split:15s} {r.best_strategy:20s} vs {r.baseline:18s} "
              f"gap={r.gap_auc_mean:+.4f} [{r.gap_ci95_lo:+.4f}, {r.gap_ci95_hi:+.4f}]{star}")


if __name__ == "__main__":
    main()
