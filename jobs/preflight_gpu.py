"""Fail-fast validation for a GPU node before the credit-risk experiment starts."""
import importlib.metadata
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"tabpfn": "2.0.3", "lightgbm": "4.3.0", "xgboost": "2.0.3"}


def _require(condition, message):
    """Raise a concise runtime error when one required execution condition is false.

    Args are a boolean condition and its user-facing remediation message. Returns nothing when
    the condition is true, otherwise terminates the preflight before model work starts.
    """
    if not condition:
        raise RuntimeError(message)


def _check_packages():
    """Verify the three GPU-model package versions required by the experiment.

    The function reads installed package metadata and returns nothing. Exact pins prevent a
    node-local package upgrade from changing the experimental software stack silently.
    """
    for package, expected in REQUIRED.items():
        found = importlib.metadata.version(package)
        _require(found == expected, f"{package} must be {expected}; found {found}.")
        print(f"{package}={found}")


def _check_gpu():
    """Confirm NVIDIA tooling and PyTorch can access at least one CUDA device.

    Returns nothing after printing the GPU name and CUDA runtime. Raises before any training if
    the allocation is missing, the driver is incompatible, or PyTorch sees no device.
    """
    command = ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    report = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    _require(bool(report), "nvidia-smi returned no GPU; request a GPU allocation.")
    import torch
    _require(torch.cuda.is_available(), "PyTorch cannot use CUDA; choose a compatible GPU node/environment.")
    print("GPU:", report.replace("\n", "; "))
    print("PyTorch CUDA:", torch.version.cuda, "| device:", torch.cuda.get_device_name(0))


def _check_data():
    """Require both user-supplied Kaggle files before a full study starts.

    This prevents an allocated GPU job from failing after setup because a source file was not
    copied into the project. Returns nothing when both nonempty paths exist.
    """
    lending_club = ROOT / "data/lending_club.csv"
    _require(lending_club.is_file() and lending_club.stat().st_size > 0,
             "Missing required data file: " + str(lending_club))
    freddie = list((ROOT / "data").glob("Sample */sample_orig_*.txt"))
    _require(bool(freddie), "Missing Freddie Mac sample origination files under data/Sample <year>/.")


def main():
    """Run all GPU-node, package, and source-data checks in fail-fast order."""
    _check_data()
    _check_packages()
    _check_gpu()
    print("PASS: GPU node and full-experiment inputs are ready.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("GPU preflight failed:", error, file=sys.stderr)
        sys.exit(1)
