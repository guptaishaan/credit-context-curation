"""Dataset-agnostic chronological splitting helpers."""
import pandas as pd


def get_split(X, y, dates, split_config):
    """Split aligned features, labels, and dates into configured pool and test years.

    Args are full aligned datasets and a mapping with ``pool_years`` and ``test_years``.
    Returns six aligned objects in pool-then-test order.
    """
    dates = pd.to_datetime(dates)
    pool_mask = dates.dt.year.isin(split_config["pool_years"])
    test_mask = dates.dt.year.isin(split_config["test_years"])
    if not pool_mask.any() or not test_mask.any():
        raise ValueError("Split has no pool or test rows; check dataset date coverage.")
    return (X.loc[pool_mask].reset_index(drop=True), y.loc[pool_mask].reset_index(drop=True),
            dates.loc[pool_mask].reset_index(drop=True), X.loc[test_mask].reset_index(drop=True),
            y.loc[test_mask].reset_index(drop=True), dates.loc[test_mask].reset_index(drop=True))
