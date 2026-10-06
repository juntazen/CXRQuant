"""Repository-relative path resolution for CXRQuant.

Every script in this repository resolves its inputs and outputs through this
module, so the toolkit runs unchanged on a laptop, in CI, inside Docker, or on a
cluster — no absolute paths, no machine-specific configuration.

Resolution order for the repository root:

1. ``CXRQUANT_ROOT`` environment variable, if set (useful when the package is
   pip-installed and the data/results live elsewhere);
2. the directory containing an installed source tree (``src/cxrquant`` layout);
3. the current working directory, as a last resort.

Each directory constant can also be overridden individually by environment
variable, which is what the Docker image and the CI workflow do.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "DATA_DIR",
    "DATA_JSON",
    "RESULTS_DIR",
    "FIGURES_DIR",
    "TABLES_DIR",
    "MODELS_DIR",
    "ensure_dirs",
]


def _detect_root() -> Path:
    env = os.environ.get("CXRQUANT_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    # .../<repo>/src/cxrquant/paths.py  ->  <repo>
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src").is_dir():
            return parent

    # installed into site-packages without the repo: fall back to cwd
    return Path.cwd().resolve()


REPO_ROOT: Path = _detect_root()


def _dir(env_var: str, default: Path) -> Path:
    value = os.environ.get(env_var)
    return Path(value).expanduser().resolve() if value else default


DATA_DIR: Path = _dir("CXRQUANT_DATA_DIR", REPO_ROOT / "data")
DATA_JSON: Path = _dir("CXRQUANT_DATA_JSON", DATA_DIR / "iuxray_paired.json")
RESULTS_DIR: Path = _dir("CXRQUANT_RESULTS_DIR", REPO_ROOT / "results")
FIGURES_DIR: Path = _dir("CXRQUANT_FIGURES_DIR", REPO_ROOT / "figures")
TABLES_DIR: Path = _dir("CXRQUANT_TABLES_DIR", REPO_ROOT / "tables")

# Local fine-tuned checkpoints are NOT shipped (multi-GB, regenerable).
# Only the GPU re-training path needs this; the CPU audit path does not.
MODELS_DIR: Path = _dir("CXRQUANT_MODELS_DIR", REPO_ROOT / "models")


def ensure_dirs(*dirs: Path) -> None:
    """Create output directories on demand (no-op if they already exist)."""
    for d in dirs or (RESULTS_DIR, FIGURES_DIR, TABLES_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
