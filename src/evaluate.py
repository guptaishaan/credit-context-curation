"""Metrics and durable result writing for credit-risk experiments."""
from pathlib import Path
import csv
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def ece(y_true, y_prob, n_bins=10):
    """Compute expected calibration error from equally spaced probability bins.

    Inputs are one-dimensional labels and predicted default probabilities. Returns a scalar
    weighted average of the absolute observed-versus-predicted gap in nonempty bins.
    """
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    bins, value = np.linspace(0, 1, n_bins + 1), 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & ((y_prob < bins[i + 1]) if i < n_bins - 1 else (y_prob <= bins[i + 1]))
        if mask.sum():
            value += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return value / len(y_true)


def compute_metrics(y_true, y_prob, runtime_seconds):
    """Calculate ranking, calibration, and runtime metrics for one prediction vector.

    Returns a flat serializable dictionary; a single-class test set is rejected because AUC and
    average precision would not be meaningful for the proposed research question.
    """
    if len(np.unique(y_true)) < 2:
        raise ValueError("Test labels need both classes to compute all requested metrics.")
    return {"roc_auc": roc_auc_score(y_true, y_prob), "average_precision": average_precision_score(y_true, y_prob),
            "brier_score": brier_score_loss(y_true, y_prob), "ece": ece(y_true, y_prob), "runtime_seconds": runtime_seconds}


def append_result(path, row):
    """Append one flat result dictionary to CSV immediately, creating its header if needed.

    The function opens and closes the file for every row so previously completed experimental
    work remains available if an individual later combination fails.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
