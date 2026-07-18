#!/bin/bash
#SBATCH --job-name=credit-context
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err

set -euo pipefail
cd "$(dirname "$0")/.."
export MPLCONFIGDIR="$PWD/outputs/.mplconfig"
export HF_HOME="$PWD/outputs/huggingface-cache"
export XDG_CACHE_HOME="$PWD/outputs/cache"
export PYTHONNOUSERSITE=1
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"

# Activate the cluster Python environment where requirements.txt was installed.
source .venv/bin/activate
python jobs/preflight_gpu.py
python src/run_experiment.py
python src/visualize.py
