"""Matplotlib figures for completed context-curation experiments."""
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def _results():
    """Load completed results or fail clearly when experiments have not run."""
    path = ROOT / "outputs" / "results.csv"
    if not path.exists():
        raise FileNotFoundError("Run src/run_experiment.py before generating figures.")
    return pd.read_csv(path)


def _save(fig, name):
    """Save a labeled Matplotlib figure at 150 dpi and release its memory."""
    directory = ROOT / "outputs" / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(directory / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def strategy_comparison(frame):
    """Create per-dataset n=512 strategy AUC bars with XGBoost reference lines."""
    datasets = list(frame.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        subset = frame[(frame.dataset == dataset) & (frame.n_context == 512) & (frame.strategy != "xgboost_full_pool")]
        values = subset.groupby("strategy").roc_auc.mean().sort_values(ascending=False)
        baseline = frame[(frame.dataset == dataset) & (frame.strategy == "xgboost_full_pool")].roc_auc.mean()
        axis.bar(range(len(values)), values, label="TabPFN")
        axis.axhline(baseline, color="black", linestyle="--", label="XGBoost full pool")
        axis.set_xticks(range(len(values)), values.index, rotation=35, ha="right")
        axis.set(title=f"{dataset}: strategy AUC (n=512)", ylabel="ROC-AUC", xlabel="Context strategy")
        axis.legend()
    _save(fig, "01_strategy_comparison.png")


def scaling_curves(frame):
    """Plot lc_pre_crisis AUC scaling for every TabPFN strategy and XGBoost."""
    subset = frame[(frame.split == "lc_pre_crisis") & (frame.strategy != "xgboost_full_pool")]
    fig, axis = plt.subplots(figsize=(8, 5))
    for strategy, group in subset.groupby("strategy"):
        group = group.sort_values("n_context")
        axis.plot(group.n_context, group.roc_auc, marker="o", label=strategy)
    baseline = frame[(frame.split == "lc_pre_crisis") & (frame.strategy == "xgboost_full_pool")].roc_auc.mean()
    axis.axhline(baseline, color="black", linestyle="--", label="XGBoost (full pool)")
    axis.set(title="Scaling on Lending Club pre-crisis split", xlabel="Context rows", ylabel="ROC-AUC")
    axis.legend(fontsize=8)
    _save(fig, "02_scaling_curves.png")


def _reliability(axis, y_true, y_prob, title):
    """Draw one ten-bin observed-frequency reliability curve with a perfect reference."""
    bins = np.linspace(0, 1, 11)
    centers, observed = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & ((y_prob < hi) if hi < 1 else (y_prob <= hi))
        if mask.any():
            centers.append(y_prob[mask].mean())
            observed.append(y_true[mask].mean())
    axis.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axis.plot(centers, observed, marker="o", label="Model")
    axis.set(title=title, xlabel="Mean predicted probability", ylabel="Observed default rate", xlim=(0, 1), ylim=(0, 1))
    axis.legend()


def calibration_diagram(frame):
    """Create random-versus-economic reliability panels for Lending Club pre-crisis n=512."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    directory = ROOT / "outputs" / "predictions"
    for axis, strategy in zip(axes, ["random", "economically_similar"]):
        # The pre-crisis pool is small, so the 512-row context is clamped; use the largest
        # available context artifact for this strategy instead of assuming an exact size.
        matches = sorted(directory.glob(f"lending_club_lc_pre_crisis_{strategy}_*.npz"),
                         key=lambda p: int(p.stem.split("_")[-1]))
        if not matches:
            raise FileNotFoundError("Missing prediction artifact for Figure 3: " + strategy)
        values = np.load(matches[-1])
        _reliability(axis, values["y_true"], values["y_prob"], strategy.replace("_", " ").title())
    _save(fig, "03_calibration_reliability.png")


def ranking_heatmap(frame):
    """Plot rank-within-split AUC heatmap at n=512, including XGBoost."""
    tabular = frame[(frame.n_context == 512) & (frame.strategy != "xgboost_full_pool")]
    xgb = frame[frame.strategy == "xgboost_full_pool"]
    view = pd.concat([tabular, xgb], ignore_index=True)
    pivot = view.pivot_table(index="strategy", columns="split", values="roc_auc", aggfunc="mean")
    ranks = pivot.rank(axis=0, ascending=False, method="min")
    display = ranks.max().max() + 1 - ranks
    fig, axis = plt.subplots(figsize=(10, 5))
    image = axis.imshow(display, cmap="Blues", aspect="auto")
    axis.set(xticks=range(len(pivot.columns)), xticklabels=pivot.columns, yticks=range(len(pivot.index)), yticklabels=pivot.index,
             title="ROC-AUC rank within each temporal split (1 = best)", xlabel="Split", ylabel="Strategy")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for i in range(ranks.shape[0]):
        for j in range(ranks.shape[1]):
            if np.isfinite(ranks.iloc[i, j]):
                axis.text(j, i, str(int(ranks.iloc[i, j])), ha="center", va="center")
    fig.colorbar(image, ax=axis, label="Better rank (darker)")
    _save(fig, "04_strategy_ranking_heatmap.png")


def auc_gap(frame):
    """Plot each n=512 TabPFN strategy's AUC difference from full-pool XGBoost."""
    tabular = frame[(frame.n_context == 512) & (frame.strategy != "xgboost_full_pool")].copy()
    baselines = frame[frame.strategy == "xgboost_full_pool"].set_index(["dataset", "split"]).roc_auc
    tabular["gap"] = [row.roc_auc - baselines.loc[(row.dataset, row.split)] for _, row in tabular.iterrows()]
    groups, strategies = tabular[["dataset", "split"]].drop_duplicates().apply(lambda r: f"{r.dataset}\n{r.split}", axis=1).tolist(), sorted(tabular.strategy.unique())
    fig, axis = plt.subplots(figsize=(12, 5))
    width, positions = 0.12, np.arange(len(groups))
    for index, strategy in enumerate(strategies):
        values = [tabular[(tabular.dataset == pair[0]) & (tabular.split == pair[1]) & (tabular.strategy == strategy)].gap.mean()
                  for pair in tabular[["dataset", "split"]].drop_duplicates().itertuples(index=False, name=None)]
        axis.bar(positions + (index - len(strategies) / 2) * width, values, width, label=strategy)
    axis.axhline(0, color="black", linewidth=1)
    axis.set(title="AUC gap to XGBoost at n=512", xlabel="Dataset and split", ylabel="TabPFN AUC − XGBoost AUC",
             xticks=positions, xticklabels=groups)
    axis.legend(fontsize=8, ncol=2)
    _save(fig, "05_auc_gap_to_xgboost.png")


def main():
    """Build all five requested figures from one completed results CSV."""
    frame = _results()
    strategy_comparison(frame)
    scaling_curves(frame)
    calibration_diagram(frame)
    ranking_heatmap(frame)
    auc_gap(frame)
    LOGGER.info("Saved five figures under outputs/figures/.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
