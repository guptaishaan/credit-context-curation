"""GPU experiment entry point for TabPFN context-curation research."""
import argparse
import importlib.metadata
import inspect
import logging
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
import yaml

from baselines import run_xgboost_baseline, run_xgboost_tuned
from evaluate import append_result, compute_metrics
from splitter import get_split
import context_strategies

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def _assert_tabpfn_and_cuda():
    """Verify TabPFN v2 and CUDA before any model computation begins.

    This project is intentionally GPU-only so training and inference do not silently run on a
    login-node CPU. Raises a clear error when submitted without an allocated GPU.
    """
    try:
        version = importlib.metadata.version("tabpfn")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("TabPFN is not installed; run pip install -r requirements.txt.") from error
    if tuple(map(int, version.split(".")[:2])) < (2, 0):
        raise RuntimeError("TabPFN >= 2.0.0 is required; found " + version)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is visible. Submit jobs/run_gpu_experiment.sbatch to a GPU node.")


def _load_dataset(name, settings, seed):
    """Load one named dataset and return its raw rows plus split preprocessor function."""
    if name == "lending_club":
        from data_loader import load, preprocess_split
    elif name == "freddie_mac":
        from data_loader_fm import load, preprocess_split
    else:
        raise ValueError("Unknown dataset: " + name)
    path = ROOT / settings["path"]
    return (*load(path, settings["max_rows"], seed), preprocess_split)


def _tabpfn_predict(X_ctx, y_ctx, X_test, ensembles, seed):
    """Fit GPU TabPFN on a context and return held-out default probabilities and runtime."""
    from tabpfn import TabPFNClassifier
    arguments = {"device": "cuda"}
    parameters = inspect.signature(TabPFNClassifier).parameters
    if "n_estimators" in parameters:
        arguments["n_estimators"] = ensembles
    else:
        arguments["N_ensemble_configurations"] = ensembles
    if "random_state" in parameters:
        arguments["random_state"] = seed
    classifier = TabPFNClassifier(**arguments)
    start = time.perf_counter()
    classifier.fit(X_ctx, y_ctx)
    if len(classifier.classes_) < 2:
        # A single-class curated context (can happen when a strategy selects no defaults on a rare-event
        # split) cannot rank the absent class; emit its constant probability, giving chance-level AUC.
        return np.full(len(X_test), float(classifier.classes_[0] == 1)), time.perf_counter() - start
    # A single large test batch overflows TabPFN's attention CUDA kernel (cudaErrorInvalidConfiguration).
    # Per-row scoring is independent given the fixed context, so chunked prediction is numerically exact.
    chunks = [classifier.predict_proba(X_test.iloc[i:i + 4096])[:, 1] for i in range(0, len(X_test), 4096)]
    return np.concatenate(chunks), time.perf_counter() - start


def _macro_indices(dataset, columns):
    """Return rate/underwriting proxy column positions for economic similarity selection."""
    names = ("int_rate", "fico_range_low") if dataset == "lending_club" else ("orig_interest_rate", "credit_score")
    missing = [name for name in names if name not in columns]
    if missing:
        raise ValueError("Economic proxy columns missing after preprocessing: " + ", ".join(missing))
    return tuple(columns.get_loc(name) for name in names)


def _select(strategy, X_pool, y_pool, dates, n_context, seed, X_test, y_test, macro_indices):
    """Invoke one strategy with the shared public API and its required optional evidence."""
    chooser = getattr(context_strategies, strategy)
    if strategy == "economically_similar":
        return chooser(X_pool, y_pool, dates, n_context, seed, X_test, y_test, macro_indices)
    return chooser(X_pool, y_pool, dates, n_context, seed)


def _write_predictions(predictions_dir, dataset, split, strategy, n_context, y_true, probabilities):
    """Persist prediction vectors needed for the reliability plot without bloating results CSV."""
    predictions_dir.mkdir(parents=True, exist_ok=True)
    name = f"{dataset}_{split}_{strategy}_{n_context}.npz"
    np.savez_compressed(predictions_dir / name, y_true=np.asarray(y_true), y_prob=np.asarray(probabilities))


def _row(dataset, split, strategy, n_context, n_requested, metrics, seed):
    """Attach experimental identifiers to a metric dictionary for CSV serialization.

    ``n_context`` is the achieved context size (clamped to the pool); ``n_requested`` is the
    configured size, kept stable across seeds so multi-seed results group cleanly.
    """
    return {"dataset": dataset, "split": split, "strategy": strategy, "n_context": int(n_context),
            "n_requested": int(n_requested), "seed": int(seed), **metrics}


def _run_split(dataset, split, config, raw_X, raw_y, raw_dates, preprocessor, results_path, predictions_dir, tuned=False):
    """Preprocess one temporal split and execute its baseline and TabPFN combinations."""
    Xp_raw, yp, dp, Xt_raw, yt, _ = get_split(raw_X, raw_y, raw_dates, config["datasets"][dataset]["splits"][split])
    Xp, Xt = preprocessor(Xp_raw, Xt_raw)
    seed, ensembles = config["seed"], config["tabpfn"]["n_ensemble_configurations"]
    base_metrics, base_probs = run_xgboost_baseline(Xp, yp, Xt, yt, seed)
    baseline = _row(dataset, split, "xgboost_full_pool", len(Xp), len(Xp), base_metrics, seed)
    append_result(results_path, baseline)
    _write_predictions(predictions_dir, dataset, split, "xgboost_full_pool", len(Xp), yt, base_probs)
    print(f"[{dataset}] [{split}] [xgboost_full_pool] n={len(Xp)} | AUC={baseline['roc_auc']:.4f} | AP={baseline['average_precision']:.4f} | runtime={baseline['runtime_seconds']:.1f}s")
    if tuned:
        tuned_metrics, tuned_probs = run_xgboost_tuned(Xp, yp, Xt, yt, seed)
        tuned_row = _row(dataset, split, "xgboost_tuned", len(Xp), len(Xp), tuned_metrics, seed)
        append_result(results_path, tuned_row)
        _write_predictions(predictions_dir, dataset, split, "xgboost_tuned", len(Xp), yt, tuned_probs)
        print(f"[{dataset}] [{split}] [xgboost_tuned] n={len(Xp)} | AUC={tuned_row['roc_auc']:.4f} | AP={tuned_row['average_precision']:.4f} | runtime={tuned_row['runtime_seconds']:.1f}s")
    macro_indices = _macro_indices(dataset, Xp.columns)
    for strategy in config["strategies"]:
        for n_context in config["n_context_values"]:
            Xc, yc = _select(strategy, Xp, yp, dp, n_context, seed, Xt, yt, macro_indices)
            if len(Xc) > 1024:
                LOGGER.warning("Strategy %s returned %d rows; subsampling to TabPFN cap of 1024.", strategy, len(Xc))
                chosen = np.random.default_rng(seed).choice(len(Xc), 1024, replace=False)
                Xc, yc = Xc.iloc[chosen], yc.iloc[chosen]
            probabilities, runtime = _tabpfn_predict(Xc, yc, Xt, ensembles, seed)
            result = _row(dataset, split, strategy, len(Xc), n_context, compute_metrics(yt, probabilities, runtime), seed)
            append_result(results_path, result)
            _write_predictions(predictions_dir, dataset, split, strategy, len(Xc), yt, probabilities)
            print(f"[{dataset}] [{split}] [{strategy}] n={len(Xc)} | AUC={result['roc_auc']:.4f} | AP={result['average_precision']:.4f} | runtime={runtime:.1f}s")


def _check_results(results_path, expected_rows):
    """Assert exact completion count and finite metrics, then print a clear PASS message."""
    frame = pd.read_csv(results_path)
    metrics = ["roc_auc", "average_precision", "brier_score", "ece", "runtime_seconds"]
    if len(frame) != expected_rows or not np.isfinite(frame[metrics].to_numpy(dtype=float)).all():
        raise AssertionError(f"FAIL: expected {expected_rows} finite rows, found {len(frame)} rows.")
    print(f"PASS: results.csv contains exactly {expected_rows} rows with no NaN metrics.")


def _print_hard_split_summary(results_path):
    """Print mean strategy metrics for the two most severe temporal-shift splits present."""
    frame = pd.read_csv(results_path)
    subset = frame[frame["split"].isin(["lc_pre_crisis", "fm_early"])]
    if len(subset):
        summary = subset.groupby(["dataset", "split", "strategy"])[["roc_auc", "average_precision"]].mean().round(4)
        print("\nHard-drift summary:\n" + summary.to_string())


def main():
    """Parse arguments, run selected datasets on GPU, and validate durable result output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run Lending Club pre-crisis random n=128 plus XGBoost.")
    parser.add_argument("--dataset", choices=["lending_club", "freddie_mac"], help="Run just one dataset.")
    parser.add_argument("--seed", type=int, help="Override the config random seed (for multi-seed sweeps).")
    parser.add_argument("--results-path", help="Override the results CSV output path.")
    parser.add_argument("--predictions-dir", help="Override the predictions output directory.")
    parser.add_argument("--tuned-xgb", action="store_true", help="Also run a cross-validated tuned XGBoost baseline per split.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _assert_tabpfn_and_cuda()
    with (ROOT / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    if args.seed is not None:
        config["seed"] = args.seed
    selected = [args.dataset] if args.dataset else list(config["datasets"])
    tuned = args.tuned_xgb and not args.dry_run
    if args.dry_run:
        selected, config["strategies"], config["n_context_values"] = ["lending_club"], ["random"], [128]
        config["datasets"]["lending_club"]["splits"] = {"lc_pre_crisis": config["datasets"]["lending_club"]["splits"]["lc_pre_crisis"]}
    results_path = Path(args.results_path) if args.results_path else ROOT / config["output"]["results_path"]
    predictions_dir = Path(args.predictions_dir) if args.predictions_dir else ROOT / "outputs" / "predictions"
    results_path.unlink(missing_ok=True)
    shutil.rmtree(predictions_dir, ignore_errors=True)
    for dataset in selected:
        raw_X, raw_y, raw_dates, preprocessor = _load_dataset(dataset, config["datasets"][dataset], config["seed"])
        for split in config["datasets"][dataset]["splits"]:
            _run_split(dataset, split, config, raw_X, raw_y, raw_dates, preprocessor, results_path, predictions_dir, tuned)
    baselines = 2 if tuned else 1
    expected = 2 if args.dry_run else sum(baselines + len(config["strategies"]) * len(config["n_context_values"])
                                         for name in selected for _ in config["datasets"][name]["splits"])
    _check_results(results_path, expected)
    _print_hard_split_summary(results_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        LOGGER.exception("Experiment failed: %s", error)
        sys.exit(1)
