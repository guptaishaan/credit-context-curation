"""Figure 07: in-context FM vs trained-on-the-same-context, at n=1024.

Scatter of TabPFN AUC against logistic-regression-on-context AUC across every (split, strategy) cell.
Points above the diagonal = the pretrained in-context prior beats training on the identical context;
the cluster below 0.5 is the model-agnostic high_confidence inversion.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = pd.read_csv(ROOT / "outputs" / "stats" / "cross_model.csv")
    g = d.groupby(["dataset", "split", "strategy", "model"]).roc_auc.mean().unstack("model").reset_index()
    strategies = sorted(g.strategy.unique())
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7.2, 7))
    ax.plot([0.2, 0.85], [0.2, 0.85], "k--", lw=1, label="y = x (tie)")
    ax.axhline(0.5, color="grey", ls=":", lw=0.8); ax.axvline(0.5, color="grey", ls=":", lw=0.8)
    for i, s in enumerate(strategies):
        sub = g[g.strategy == s]
        ax.scatter(sub["lr_context"], sub["tabpfn"], s=70, color=cmap(i), label=s, edgecolor="white", zorder=3)
    ax.set(xlabel="Logistic regression trained on the same 1,024-row context (AUC)",
           ylabel="TabPFN in-context (AUC)", xlim=(0.2, 0.85), ylim=(0.2, 0.85),
           title="In-context prior vs. training on the identical context\n(above diagonal = FM prior wins; lower-left cluster = high_confidence inversion)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    out = ROOT / "outputs" / "figures" / "07_incontext_vs_trained.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    # summary numbers for the writeup
    viable = g[(g.strategy != "high_confidence") & (g.split != "lc_pre_crisis")]
    print(f"viable cells: TabPFN-LR mean gap = {(viable.tabpfn - viable.lr_context).mean():+.3f} "
          f"(TabPFN>LR in {(viable.tabpfn>viable.lr_context).mean()*100:.0f}% of cells)")
    print(f"TabPFN-XGB mean gap = {(viable.tabpfn - viable.xgb_context).mean():+.3f}")
    hc = g[g.strategy == "high_confidence"]
    print(f"high_confidence: all-model mean AUC = {hc[['tabpfn','lr_context','xgb_context']].mean().mean():.3f} (inverted)")


if __name__ == "__main__":
    main()
