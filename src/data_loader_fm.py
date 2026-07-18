"""Freddie Mac Single-Family Loan-Level (sample) loader and leak-free split preprocessing.

Reads the per-vintage ``sample_orig_YYYY.txt`` / ``sample_svcg_YYYY.txt`` pairs, derives a binary
default label from the monthly performance file (ever 180+ days delinquent, or a credit-event zero
balance code), and returns the same ``(X, y, dates)`` triple the other loaders provide.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from data_loader import _sample_years

# Zero-indexed field positions in the 32-column standard origination file.
ORIG_NUMERIC = {"credit_score": 0, "mi_pct": 5, "num_units": 6, "cltv": 8, "dti": 9,
                "orig_upb": 10, "ltv": 11, "orig_interest_rate": 12, "orig_loan_term": 21,
                "num_borrowers": 22}
ORIG_CATEGORICAL = {"first_time_homebuyer": 2, "occupancy": 7, "channel": 13,
                    "property_type": 17, "loan_purpose": 20}
ORIG_LOAN_SEQ = 19
# "Not available" sentinels Freddie uses for out-of-range values (per the User Guide / FAQ).
SENTINELS = {"credit_score": 9999, "mi_pct": 999, "cltv": 999, "dti": 999, "ltv": 999}
# Monthly performance file: loan id, delinquency status, zero balance code.
SVCG_SEQ, SVCG_DELINQUENCY, SVCG_ZERO_BALANCE = 0, 3, 8
DEFAULT_ZERO_BALANCE = {"03", "09"}  # short sale / charge-off and REO disposition


def _factorize(series):
    """Encode a categorical Freddie field to deterministic float codes with NaN for missing."""
    codes, _ = pd.factorize(series.astype("string").str.strip(), sort=True)
    return pd.Series(codes, index=series.index).replace(-1, np.nan).astype(float)


def _loan_default(svcg_path):
    """Return a per-loan default flag from one monthly performance file.

    A loan defaults if it ever reaches 180+ days delinquent (status >= 6) or terminates with a
    credit-event zero balance code. Reads only the three columns needed to keep memory bounded.
    """
    perf = pd.read_csv(svcg_path, sep="|", header=None, low_memory=False,
                       usecols=[SVCG_SEQ, SVCG_DELINQUENCY, SVCG_ZERO_BALANCE],
                       dtype={SVCG_SEQ: "string", SVCG_ZERO_BALANCE: "string"})
    dlq = pd.to_numeric(perf[SVCG_DELINQUENCY], errors="coerce")
    event = (dlq >= 6) | perf[SVCG_ZERO_BALANCE].str.strip().isin(DEFAULT_ZERO_BALANCE)
    return event.groupby(perf[SVCG_SEQ]).any()


def _load_year(orig_path, svcg_path):
    """Build features, labels, and origination dates for one vintage year."""
    usecols = sorted({*ORIG_NUMERIC.values(), *ORIG_CATEGORICAL.values(), ORIG_LOAN_SEQ})
    orig = pd.read_csv(orig_path, sep="|", header=None, low_memory=False,
                       usecols=usecols, dtype={ORIG_LOAN_SEQ: "string"})
    X = pd.DataFrame({name: pd.to_numeric(orig[index], errors="coerce") for name, index in ORIG_NUMERIC.items()})
    for name, sentinel in SENTINELS.items():
        X[name] = X[name].replace(sentinel, np.nan)
    for name, index in ORIG_CATEGORICAL.items():
        X[name] = _factorize(orig[index])
    # Origination year/quarter is encoded in the loan sequence number, e.g. F19Q3... -> 2019 Q3.
    loan_seq = orig[ORIG_LOAN_SEQ]
    year = 2000 + loan_seq.str[1:3].astype(int)
    quarter = loan_seq.str[4].astype(int)
    dates = pd.to_datetime(dict(year=year, month=(quarter - 1) * 3 + 1, day=1))
    default = _loan_default(svcg_path).reindex(loan_seq).fillna(False).astype(int)
    return X, pd.Series(default.to_numpy(), name="target"), dates


def load(path="data", max_rows=200000, seed=42):
    """Load every available Freddie Mac sample vintage and return raw features, labels, and dates.

    Feature typing happens here; pool-only imputation and scaling happen in ``preprocess_split`` so
    the returned held-out period cannot affect fitted preprocessing statistics.
    """
    root = Path(path)
    origins = sorted(root.glob("Sample */sample_orig_*.txt"))
    if not origins:
        raise FileNotFoundError("No Freddie Mac sample files found under 'Sample <year>/' in " + str(root))
    frames, targets, dates = [], [], []
    for orig_path in origins:
        svcg_path = orig_path.with_name(orig_path.name.replace("sample_orig_", "sample_svcg_"))
        if not svcg_path.exists():
            raise FileNotFoundError("Missing performance file for " + orig_path.name)
        X, y, d = _load_year(orig_path, svcg_path)
        frames.append(X)
        targets.append(y)
        dates.append(d)
    X = pd.concat(frames, ignore_index=True)
    y = pd.concat(targets, ignore_index=True)
    d = pd.concat(dates, ignore_index=True)
    combined = pd.concat([X, y.rename("target"), d.rename("_date")], axis=1)
    combined = _sample_years(combined, combined["_date"], max_rows, seed).sort_values("_date").reset_index(drop=True)
    return combined[X.columns], combined["target"].astype(int), combined["_date"]


def preprocess_split(X_pool, X_test):
    """Drop sparse pool features, median-impute, and quantile-scale without test leakage.

    Inputs are raw aligned DataFrames. Returns transformed pool and test DataFrames with the same
    retained columns and transformations learned exclusively from the pool.
    """
    keep = X_pool.columns[X_pool.isna().mean() <= 0.5]
    if not len(keep):
        raise ValueError("All Freddie Mac features exceed the 50% pool missingness limit.")
    medians = X_pool[keep].median(numeric_only=True)
    pool = X_pool[keep].fillna(medians).fillna(0.0).astype(float)
    test = X_test[keep].fillna(medians).fillna(0.0).astype(float)
    transformer = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(pool)), random_state=42)
    return (pd.DataFrame(transformer.fit_transform(pool), columns=pool.columns),
            pd.DataFrame(transformer.transform(test), columns=test.columns))
