"""Figure 06: data-efficiency view of the multi-seed sweep.

Per temporal split, compares the best TabPFN strategy at a 1,024-row in-context budget against the
tuned and fixed full-pool XGBoost baselines (95% CI error bars across seeds), annotated with how many
training rows each XGBoost used. The point of the study: parity at ~10^2-10^5x less data, no training.
"""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["lc_pre_crisis", "lc_post_crisis", "lc_recent", "fm_early", "fm_late"]


def _mean_ci(values):
    values = np.asarray(values, dtype=float)
    half = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return values.mean(), half


def main():
    res = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / "outputs/seeds/results_s*.csv"))], ignore_index=True)
    labels, tab_m, tab_e, tun_m, tun_e, fix_m, fix_e, ratios = [], [], [], [], [], [], [], []
    for split in ORDER:
        g = res[res.split == split]
        peak = g[~g.strategy.str.startswith("xgboost")]
        peak = peak[peak.n_requested == 1024]
        best = peak.groupby("strategy").roc_auc.mean().idxmax()
        bm, be = _mean_ci(peak[peak.strategy == best].roc_auc)
        um, ue = _mean_ci(g[g.strategy == "xgboost_tuned"].roc_auc)
        fm, fe = _mean_ci(g[g.strategy == "xgboost_full_pool"].roc_auc)
        labels.append(f"{split}\n(best: {best})")
        tab_m.append(bm); tab_e.append(be); tun_m.append(um); tun_e.append(ue); fix_m.append(fm); fix_e.append(fe)
        ratios.append(int(round(g[g.strategy == "xgboost_tuned"].n_context.mean() / 1024)))

    x = np.arange(len(ORDER)); w = 0.26
    fig, axis = plt.subplots(figsize=(12, 5.5))
    axis.bar(x - w, tab_m, w, yerr=tab_e, capsize=3, label="TabPFN (1,024 in-context rows, no training)")
    axis.bar(x, tun_m, w, yerr=tun_e, capsize=3, label="XGBoost tuned (full pool)")
    axis.bar(x + w, fix_m, w, yerr=fix_e, capsize=3, label="XGBoost fixed (full pool)")
    axis.axhline(0.5, color="grey", linestyle=":", linewidth=1, label="chance")
    for i, r in enumerate(ratios):
        top = max(tab_m[i] + tab_e[i], tun_m[i] + tun_e[i])
        note = f"{r}x less\ndata" if r >= 2 else "same-size\npool"
        axis.text(i, top + 0.012, note, ha="center", va="bottom", fontsize=8, color="dimgray")
    axis.set_xticks(x, labels, fontsize=8)
    axis.set(ylabel="ROC-AUC (mean ± 95% CI, 5 seeds)", ylim=(0.45, max(tab_m + tun_m) + 0.09),
             title="Data efficiency: 1,024 curated in-context rows vs full-pool XGBoost")
    axis.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = ROOT / "outputs" / "figures" / "06_data_efficiency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
