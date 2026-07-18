# Run TabFM cross-model comparison on a free node

Everything is prepared. TabFM installed cleanly and **loads + runs** — the only blocker was that the
current node is saturated (load avg ~50, all GPUs 100%), so CPU inference crawled. On a node with free
compute this finishes quickly. All paths are on the shared `/ccn2` filesystem, so no copying is needed.

## What's already done (do NOT redo)
- Isolated conda env **`tabfm`** (Python 3.11) with TabFM 1.0.1 + `safetensors` installed.
- Repo + runner at `/ccn2/u/ishaangp/projects/tabfm_src/` (`run_tabfm_contexts.py`).
- **90 curated contexts exported** at `/ccn2/u/ishaangp/projects/credit_context_curation/outputs/contexts/`
  (`Xc, yc, Xt, yt` per (dataset, split, strategy, seed) — the *identical* contexts TabPFN/LR/XGBoost saw).
- Merge/figure script `src/finalize_tabfm.py`.

## Step 1 — run TabFM (in the `tabfm` env)
```bash
ssh <free-node>
source /ccn2/u/ishaangp/miniconda3/etc/profile.d/conda.sh
conda activate tabfm
cd /ccn2/u/ishaangp/projects/tabfm_src
export HF_HOME=/ccn2/u/ishaangp/projects/credit_context_curation/outputs/huggingface-cache
python run_tabfm_contexts.py
```
- Writes `outputs/stats/tabfm_contexts.csv` (one row per context, written incrementally — safe to resume-by-rerun).
- Prints `MODEL LOADED` then a line per context. 90 contexts total.
- **CPU is fine on an idle node.** For GPU speed, see note below.

### (Optional) GPU speedup
The `tabfm` env currently has `torch 2.13.0+cu130`. It only uses the GPU if the node's NVIDIA driver
supports CUDA 13 (`nvidia-smi` shows Driver ≥ 580 / "CUDA Version: 13.x"). If the driver is 12.x, either
run on CPU (above) or install a matching build **into the `tabfm` env only** (safe — it's isolated):
```bash
pip install "torch==2.13.0+cu126" --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
```
Then verify `python -c "import torch; print(torch.cuda.is_available())"` is `True` before rerunning.
(If TabFM does not auto-place on GPU, run on CPU — correctness is identical, only speed differs.)

## Step 2 — merge + figure (in the project `.venv`)
```bash
cd /ccn2/u/ishaangp/projects/credit_context_curation
export PYTHONNOUSERSITE=1
source .venv/bin/activate
python src/finalize_tabfm.py
```
This prints the **4-model AUC table** (TabPFN / TabFM / LR-context / XGB-context) and writes:
- `outputs/stats/cross_model_4way.csv`
- `outputs/figures/08_tabpfn_vs_tabfm.png` (cross-FM agreement scatter)

It also prints the three numbers to drop into the paper:
1. TabFM vs LR-context gap (does the in-context prior help a *second* FM?).
2. TabPFN↔TabFM agreement (correlation, mean |diff|).
3. `high_confidence` mean AUC for both FMs (does the failure mode reproduce on a 2nd FM?).

## Step 3 — update the paper (`paper/draft.md`)
The draft currently states TabFM as a planned extension. After Step 2, update:
- **§2 Models**: change TabFM from "planned" to included.
- **Abstract + §3.3**: add the cross-FM sentence using the printed numbers, e.g.
  *"The failure modes and the in-context-prior advantage reproduce on a second foundation model (TabFM):
  TabPFN and TabFM agree to within <X> AUC across cells, and high_confidence inverts for both."*
- Add Fig. 08.

## Sanity expectations
- `high_confidence` should be **< 0.5 for TabFM too** (model-agnostic inversion).
- On viable contexts TabFM should track TabPFN closely (both are in-context FMs) and generally beat
  LR/XGBoost-on-context. If TabFM sharply disagrees with TabPFN, that is itself a reportable finding —
  check `outputs/stats/cross_model_4way.csv` before concluding.
