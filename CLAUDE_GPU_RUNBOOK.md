# Claude Code GPU Runbook

Use this runbook to execute the complete credit-context-curation study on a cluster GPU node. Work only inside this project directory:

```bash
cd /ccn2/u/ishaangp/projects/credit_context_curation
```

## Objective

Run the full two-dataset experiment with GPU-only model computation, verify that `outputs/results.csv` has exactly 125 finite metric rows, and generate the five requested figures under `outputs/figures/`. Do not invent data, results, or figures if a prerequisite is missing.

## Required inputs

Before submitting a job, verify these Kaggle source files exist:

```bash
test -s data/lending_club.csv
ls data/Sample\ */sample_orig_*.txt >/dev/null
```

If either command fails, stop and report the missing file. Do not add data downloading logic to the Python pipeline. The required sources and placement instructions are in `README.md`.

## GPU submission

The project-local environment is `.venv`, and the job script requests one GPU, 16 CPUs, 64 GB RAM, and 24 hours:

```bash
sbatch jobs/run_gpu_experiment.sbatch
```

Record the submitted job ID. If this cluster uses a scheduler other than Slurm, request an interactive GPU allocation or adapt the resource directives only; retain the two commands executed by the script:

```bash
source .venv/bin/activate
python src/run_experiment.py
python src/visualize.py
```

Never run model training on a login node or use a CPU fallback. `src/run_experiment.py` rejects execution unless CUDA is visible.

## GPU-node preflight

At the start of the allocated node session, run:

```bash
nvidia-smi
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.cuda.get_device_name(0))"
python -c "import importlib.metadata; print(importlib.metadata.version('tabpfn'))"
python jobs/preflight_gpu.py
```

Continue only if `nvidia-smi` works, PyTorch reports a GPU, and TabPFN reports version 2.0.3 (or another version >= 2.0.0). If model weights need to be fetched on first use, ensure the compute node has the permitted access needed by TabPFN; do not substitute a different model.

`jobs/preflight_gpu.py` is stricter than the short commands above: it requires both data files,
the exact pinned TabPFN/LightGBM/XGBoost versions, and a CUDA-visible GPU. The Slurm job invokes
it automatically before any training. It also stores Hugging Face/TabPFN model cache files under
`outputs/huggingface-cache/` rather than a home-directory cache.

## Recommended staged execution

First run the GPU dry run. It should produce exactly two rows (one XGBoost and one TabPFN random-context result) and complete in under three minutes:

```bash
source .venv/bin/activate
python src/run_experiment.py --dry-run
```

Then run the full experiment and figures:

```bash
python src/run_experiment.py
python src/visualize.py
```

The full runner intentionally clears a prior `outputs/results.csv` at its start so the final file cannot contain duplicate rows. It appends each new result immediately as it completes, allowing partial results to survive an interrupted job.

## Monitoring

For a submitted Slurm job, inspect status and logs with:

```bash
squeue -u "$USER"
tail -f outputs/slurm-<JOB_ID>.out
tail -f outputs/slurm-<JOB_ID>.err
```

Progress lines have this form:

```text
[dataset] [split] [strategy] n=<context> | AUC=<value> | AP=<value> | runtime=<seconds>s
```

Do not interrupt a normally progressing job. If it fails, preserve `outputs/results.csv` and logs for diagnosis. Correct a code or environment issue only within this project directory, then rerun the full experiment so its final row-count assertion is meaningful.

## Completion checks

After the full run, execute:

```bash
source .venv/bin/activate
python - <<'PY'
import numpy as np
import pandas as pd

results = pd.read_csv('outputs/results.csv')
metrics = ['roc_auc', 'average_precision', 'brier_score', 'ece', 'runtime_seconds']
assert len(results) == 125, f'expected 125 rows, got {len(results)}'
assert np.isfinite(results[metrics].to_numpy(dtype=float)).all(), 'non-finite metric found'
assert set(results['dataset']) == {'lending_club', 'freddie_mac'}
print('PASS: 125 finite result rows across both datasets')
PY
find outputs/figures -maxdepth 1 -type f -name '*.png' -printf '%f\n' | sort
```

The figures directory must contain these five files:

```text
01_strategy_comparison.png
02_scaling_curves.png
03_calibration_reliability.png
04_strategy_ranking_heatmap.png
05_auc_gap_to_xgboost.png
```

## Final handoff

Report the GPU model name, job ID, whether the dry run passed, the full-run completion assertion, the locations of `outputs/results.csv` and `outputs/figures/`, and any deviation from the planned configuration. Do not claim the empirical study completed unless all 125 rows and all five figures are present.
