"""Conventional full-pool GPU XGBoost reference models (fixed and lightly tuned)."""
import time
from xgboost import XGBClassifier
from evaluate import compute_metrics


def run_xgboost_baseline(X_pool, y_pool, X_test, y_test, seed):
    """Fit the specified GPU XGBoost model on every pool row and evaluate on test.

    Inputs are already pool-fitted, numeric feature DataFrames and aligned labels. Returns the
    same flat metrics dictionary produced for TabPFN strategies, plus the test probability vector.
    """
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, use_label_encoder=False, eval_metric="logloss",
                          random_state=seed, n_jobs=-1, tree_method="hist", device="cuda")
    start = time.perf_counter()
    model.fit(X_pool, y_pool)
    probabilities = model.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test, probabilities, time.perf_counter() - start), probabilities


def run_xgboost_tuned(X_pool, y_pool, X_test, y_test, seed):
    """Fit a lightly tuned GPU XGBoost baseline (randomized search on the pool) and evaluate.

    A fair, non-strawman reference: hyperparameters are selected by cross-validated randomized
    search on the historical pool only, so no test-period information leaks. Returns the flat
    metrics dictionary and the test probability vector.
    """
    from sklearn.model_selection import RandomizedSearchCV
    # CPU tree method for the CV search: 30 small fits are fast on CPU and avoid contending for
    # shared GPUs. Model quality is device-independent; only the fixed baseline stays on GPU.
    base = XGBClassifier(use_label_encoder=False, eval_metric="logloss", tree_method="hist",
                         device="cpu", random_state=seed, n_jobs=4)
    grid = {"n_estimators": [200, 400, 600], "max_depth": [4, 6, 8], "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.9], "colsample_bytree": [0.7, 0.9], "reg_lambda": [1.0, 5.0]}
    search = RandomizedSearchCV(base, grid, n_iter=10, scoring="roc_auc", cv=3,
                                random_state=seed, n_jobs=2, refit=True)
    start = time.perf_counter()
    search.fit(X_pool, y_pool)
    probabilities = search.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test, probabilities, time.perf_counter() - start), probabilities
