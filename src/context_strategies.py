"""Deterministic strategies for choosing TabPFN's labeled context."""
import logging
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _take(X, y, indices):
    """Return reset-index context rows selected by integer positions."""
    return X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)


def _size(n_context, available):
    """Validate requested context size and clamp it to available pool rows with a warning."""
    if not available:
        raise ValueError("Cannot select a context from an empty pool.")
    if n_context > available:
        LOGGER.warning("Requested context %d exceeds pool size %d; using full pool.", n_context, available)
    return min(n_context, available)


def random(X_pool, y_pool, pool_dates, n_context, seed):
    """Uniformly sample labeled pool rows for a reference context.

    The pool dates are accepted for a common strategy interface but are not used. Returns a
    deterministic context DataFrame and aligned binary label Series.
    """
    n = _size(n_context, len(X_pool))
    indices = np.random.default_rng(seed).choice(len(X_pool), n, replace=False)
    return _take(X_pool, y_pool, indices)


def most_recent(X_pool, y_pool, pool_dates, n_context, seed):
    """Choose the most recently dated pool rows as the context.

    This measures whether examples closest in time are most useful under temporal drift.
    ``seed`` is unused but retained for the common public strategy signature.
    """
    n = _size(n_context, len(X_pool))
    indices = np.argsort(pd.to_datetime(pool_dates).to_numpy())[-n:]
    return _take(X_pool, y_pool, indices)


def class_balanced(X_pool, y_pool, pool_dates, n_context, seed):
    """Sample an as-even-as-possible mix of positive and negative labels.

    If a class is scarce, it contributes all available rows and the other class fills the
    remaining slots. The returned context is shuffled deterministically.
    """
    n = _size(n_context, len(X_pool))
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(y_pool.to_numpy() == 1), np.flatnonzero(y_pool.to_numpy() == 0)
    p_n = min(len(positive), (n + 1) // 2)
    n_n = min(len(negative), n - p_n)
    p_n = min(len(positive), n - n_n)
    chosen = np.r_[rng.choice(positive, p_n, replace=False), rng.choice(negative, n_n, replace=False)]
    return _take(X_pool, y_pool, rng.permutation(chosen))


def economically_similar(X_pool, y_pool, pool_dates, n_context, seed, X_test=None, y_test=None, macro_indices=(1, 6)):
    """Sample from the pool year whose observable credit environment best matches test.

    Distances use a rate-like feature, default rate, and underwriting-like feature. Test labels
    are intentionally used only to choose existing context examples, as specified in the brief.
    """
    if X_test is None or y_test is None:
        raise ValueError("economically_similar requires held-out X_test and y_test.")
    years = pd.to_datetime(pool_dates).dt.year
    indices = tuple(i for i in macro_indices if i < X_pool.shape[1])
    if len(indices) < 2:
        raise ValueError("Need two numeric macro proxy columns for economically_similar.")
    target = np.array([X_test.iloc[:, indices[0]].mean(), y_test.mean(), X_test.iloc[:, indices[1]].mean()])
    stats = {year: np.array([X_pool.loc[years == year].iloc[:, indices[0]].mean(), y_pool.loc[years == year].mean(),
                             X_pool.loc[years == year].iloc[:, indices[1]].mean()]) for year in sorted(years.unique())}
    best_year = min(stats, key=lambda year: np.linalg.norm(stats[year] - target))
    positions = np.flatnonzero((years == best_year).to_numpy())
    chosen = np.random.default_rng(seed).choice(positions, _size(n_context, len(positions)), replace=False)
    return _take(X_pool, y_pool, chosen)


def high_confidence(X_pool, y_pool, pool_dates, n_context, seed):
    """Choose pool examples closest to a GPU LightGBM decision boundary.

    A 100-tree classifier is fitted on the entire pool, then rows with probabilities nearest
    0.5 are returned because they are the most informative borderline examples.
    """
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(n_estimators=100, random_state=seed, device_type="gpu", gpu_use_dp=False, verbosity=-1)
    model.fit(X_pool, y_pool)
    uncertainty = np.abs(model.predict_proba(X_pool)[:, 1] - 0.5)
    return _take(X_pool, y_pool, np.argsort(uncertainty)[:_size(n_context, len(X_pool))])


def _nearest_centroids_gpu(X, n_context, seed, iterations=20):
    """Run deterministic GPU K-Means and return one nearest row index for each centroid."""
    import torch
    rng = np.random.default_rng(seed)
    device = torch.device("cuda")
    values = torch.as_tensor(X.to_numpy(dtype=np.float32), device=device)
    init = torch.as_tensor(rng.choice(len(X), n_context, replace=False), device=device)
    centers = values[init].clone()
    for _ in range(iterations):
        sums, counts = torch.zeros_like(centers), torch.zeros(n_context, device=device)
        for batch in values.split(2048):
            labels = torch.cdist(batch, centers).argmin(dim=1)
            sums.index_add_(0, labels, batch)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
        centers = torch.where((counts > 0)[:, None], sums / counts.clamp_min(1)[:, None], centers)
    nearest = []
    for center in centers:
        best_distance, best_index, offset = None, 0, 0
        for batch in values.split(4096):
            distance, position = torch.sum((batch - center) ** 2, dim=1).min(dim=0)
            if best_distance is None or distance < best_distance:
                best_distance, best_index = distance, offset + int(position)
            offset += len(batch)
        nearest.append(best_index)
    return np.unique(nearest)


def diverse(X_pool, y_pool, pool_dates, n_context, seed):
    """Choose feature-space representatives using GPU K-Means centroids.

    The row closest to every centroid is retained. Duplicate nearest rows are filled with a
    deterministic random sample so the requested context size is preserved whenever possible.
    """
    n = _size(n_context, len(X_pool))
    selected = _nearest_centroids_gpu(X_pool, n, seed)
    if len(selected) < n:
        rest = np.setdiff1d(np.arange(len(X_pool)), selected)
        selected = np.r_[selected, np.random.default_rng(seed).choice(rest, n - len(selected), replace=False)]
    return _take(X_pool, y_pool, selected)
