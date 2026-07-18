"""Merge TabFM results into the cross-model comparison and report the 4-model numbers.

Run AFTER run_tabfm_contexts.py has produced outputs/stats/tabfm_contexts.csv. Combines it with the
TabPFN/LR/XGBoost cross_model.csv, writes a merged table, a TabPFN-vs-TabFM agreement figure, and
prints the summary statistics needed to update the paper (§3.3 and the abstract).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "outputs" / "stats"


def main():
    cm = pd.read_csv(STATS / "cross_model.csv")                    # tabpfn, lr_context, xgb_context
    tf = pd.read_csv(STATS / "tabfm_contexts.csv")                 # tabfm
    combined = pd.concat([cm, tf], ignore_index=True)
    combined.to_csv(STATS / "cross_model_4way.csv", index=False)

    piv = combined.groupby(["dataset", "split", "strategy", "model"]).roc_auc.mean().unstack("model")
    order = ["tabpfn", "tabfm", "lr_context", "xgb_context"]
    piv = piv[[c for c in order if c in piv.columns]].round(3)
    print("=== 4-model AUC (mean over seeds, n=1024) ===")
    print(piv.to_string())

    g = piv.reset_index()
    viable = g[(g.strategy != "high_confidence") & (g.split != "lc_pre_crisis")]
    print("\n=== summary ===")
    if "tabfm" in g.columns:
        print(f"TabFM vs LR-context (viable): mean gap {(viable.tabfm - viable.lr_context).mean():+.3f}, "
              f"TabFM>LR in {(viable.tabfm > viable.lr_context).mean() * 100:.0f}% of cells")
        print(f"TabPFN vs TabFM agreement: corr={g.tabpfn.corr(g.tabfm):.3f}, "
              f"mean |diff|={(g.tabpfn - g.tabfm).abs().mean():.3f}")
        hc = g[g.strategy == "high_confidence"]
        print(f"high_confidence mean AUC: tabpfn={hc.tabpfn.mean():.3f} tabfm={hc.tabfm.mean():.3f} "
              f"(both < 0.5 => model-agnostic inversion confirmed on a 2nd FM)")

        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.plot([0.2, 0.85], [0.2, 0.85], "k--", lw=1, label="agree")
        ax.axhline(0.5, color="grey", ls=":", lw=0.8); ax.axvline(0.5, color="grey", ls=":", lw=0.8)
        for i, s in enumerate(sorted(g.strategy.unique())):
            sub = g[g.strategy == s]
            ax.scatter(sub.tabpfn, sub.tabfm, s=70, color=plt.get_cmap("tab10")(i), label=s, edgecolor="white", zorder=3)
        ax.set(xlabel="TabPFN AUC", ylabel="TabFM AUC", xlim=(0.2, 0.85), ylim=(0.2, 0.85), aspect="equal",
               title="Cross-FM agreement: TabPFN vs TabFM on identical curated contexts")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(ROOT / "outputs" / "figures" / "08_tabpfn_vs_tabfm.png", dpi=150, bbox_inches="tight")
        print("\nwrote outputs/figures/08_tabpfn_vs_tabfm.png and outputs/stats/cross_model_4way.csv")


if __name__ == "__main__":
    main()
