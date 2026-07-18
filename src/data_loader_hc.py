"""Home Credit stability data loader and leak-free split preprocessing."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from data_loader import _sample_years

HC_COLUMNS = ["amtannuity", "amtcredit", "amtdownpayment", "amtgoodsprice", "amtinstprincipal",
              "amtprincipal", "credittypedetailed", "datefirstdue", "datelastdue", "datelastpaid",
              "numberofinstallments", "maxofdpd"]


def _to_number(series):
    """Convert numeric, date, or categorical Home Credit fields into deterministic numbers."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.5:
        return numeric
    dates = pd.to_datetime(series, errors="coerce")
    if dates.notna().mean() > 0.5:
        return (dates - pd.Timestamp("1970-01-01")).dt.days.astype(float)
    codes, _ = pd.factorize(series.astype("string"), sort=True)
    return pd.Series(codes, index=series.index).replace(-1, np.nan).astype(float)


def load(path="data/home_credit_base.csv", max_rows=200000, seed=42):
    """Load Home Credit rows and return raw features, labels, and decision dates.

    Feature typing happens here; pool-only imputation and scaling happen in ``preprocess_split``
    so the returned held-out period cannot affect fitted preprocessing statistics.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Missing data/home_credit_base.csv. Download competition "
                                "home-credit-credit-risk-model-stability train_base.csv and rename it here.")
    raw = pd.read_csv(path, low_memory=False)
    required = set(HC_COLUMNS + ["date_decision", "target"])
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError("Home Credit CSV is missing columns: " + ", ".join(sorted(missing)))
    raw["_date"] = pd.to_datetime(raw.date_decision, format="%Y-%m-%d", errors="coerce")
    raw = raw.loc[raw["_date"].notna() & raw.target.notna()].copy().sort_values("_date").reset_index(drop=True)
    raw = _sample_years(raw, raw["_date"], max_rows, seed).sort_values("_date").reset_index(drop=True)
    X = pd.DataFrame({column: _to_number(raw[column]) for column in HC_COLUMNS})
    return X, raw.target.astype(int), raw["_date"]


def preprocess_split(X_pool, X_test):
    """Drop sparse pool features, median-impute, and quantile-scale without test leakage.

    Inputs are raw aligned DataFrames. Returns transformed pool and test DataFrames with the
    same retained columns and transformations learned exclusively from the pool.
    """
    keep = X_pool.columns[X_pool.isna().mean() <= 0.5]
    if not len(keep):
        raise ValueError("All Home Credit features exceed the 50% pool missingness limit.")
    medians = X_pool[keep].median(numeric_only=True)
    pool = X_pool[keep].fillna(medians).fillna(0.0).astype(float)
    test = X_test[keep].fillna(medians).fillna(0.0).astype(float)
    transformer = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(pool)), random_state=42)
    return (pd.DataFrame(transformer.fit_transform(pool), columns=pool.columns),
            pd.DataFrame(transformer.transform(test), columns=test.columns))
