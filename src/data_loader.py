"""Lending Club data loading and leak-free split preprocessing."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

LC_NUMERIC = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
              "fico_range_low", "fico_range_high", "open_acc", "pub_rec", "revol_bal",
              "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies"]
IN_PROGRESS = {"Current", "Late", "In Grace Period"}


def _sample_years(frame, dates, max_rows, seed):
    """Cap data with deterministic proportional samples inside each calendar year."""
    if len(frame) <= max_rows:
        return frame
    rng = np.random.default_rng(seed)
    sizes = dates.dt.year.value_counts().sort_index()
    quotas = np.maximum(1, np.floor(sizes / sizes.sum() * max_rows).astype(int))
    quotas.iloc[:max_rows - int(quotas.sum())] += 1
    pieces = [g.iloc[rng.choice(len(g), min(len(g), int(quotas.loc[year])), replace=False)]
              for year, g in frame.groupby(dates.dt.year, sort=True)]
    return pd.concat(pieces, ignore_index=True)


def _numeric(series):
    """Convert ordinary numeric or percent-formatted Lending Club fields to floats."""
    return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False), errors="coerce")


def load(path="data/lending_club.csv", max_rows=200000, seed=42):
    """Load Lending Club rows and return raw model features, binary labels, and dates.

    Categorical encoding is performed here; median imputation and scaling are deliberately
    fitted later on the temporal pool so no test-period statistics can leak into features.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Missing data/lending_club.csv. Download Kaggle dataset "
                                "wordsforthewise/lending-club and save its CSV at this path.")
    raw = pd.read_csv(path, low_memory=False)
    required = set(LC_NUMERIC + ["issue_d", "loan_status", "home_ownership", "purpose"])
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError("Lending Club CSV is missing columns: " + ", ".join(sorted(missing)))
    raw["_date"] = pd.to_datetime(raw["issue_d"], format="%b-%Y", errors="coerce")
    raw = raw.loc[raw["_date"].notna() & ~raw.loan_status.isin(IN_PROGRESS)].copy()
    raw["_target"] = raw.loan_status.isin(["Charged Off", "Default"]).astype(int)
    raw = raw.sort_values("_date").reset_index(drop=True)
    raw = _sample_years(raw, raw["_date"], max_rows, seed).sort_values("_date").reset_index(drop=True)
    numeric = pd.DataFrame({col: _numeric(raw[col]) for col in LC_NUMERIC})
    homes = raw.home_ownership.where(raw.home_ownership.isin(["RENT", "OWN", "MORTGAGE"]), "OTHER")
    home_x = pd.get_dummies(homes, prefix="home_ownership").reindex(
        columns=["home_ownership_RENT", "home_ownership_OWN", "home_ownership_MORTGAGE", "home_ownership_OTHER"], fill_value=0)
    top_purpose = raw.purpose.value_counts().head(8).index
    purposes = raw.purpose.where(raw.purpose.isin(top_purpose), "other")
    purpose_x = pd.get_dummies(purposes, prefix="purpose")
    return pd.concat([numeric, home_x, purpose_x], axis=1), raw["_target"], raw["_date"]


def preprocess_split(X_pool, X_test):
    """Median-impute and quantile-scale a split using pool statistics only.

    Takes raw feature DataFrames and returns numeric pool and test DataFrames with identical
    columns. The fitted transformations never inspect the held-out test values.
    """
    medians = X_pool.median(numeric_only=True)
    pool = X_pool.fillna(medians).fillna(0.0).astype(float)
    test = X_test.fillna(medians).fillna(0.0).astype(float)
    transformer = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(pool)), random_state=42)
    return (pd.DataFrame(transformer.fit_transform(pool), columns=pool.columns),
            pd.DataFrame(transformer.transform(test), columns=test.columns))
