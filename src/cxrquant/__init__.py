"""CXRQuant — finding-level stability auditing of reduced-precision inference for radiology impression generation.

The extraction and metric core (:mod:`cxrquant.clinical_safety`) is pure standard-library Python and
re-derives the reported numbers from the archived predictions in ``runs/`` and ``results_r2/`` on a CPU.
Training and generation (:mod:`cxrquant.pipeline`) require PyTorch, Transformers and BitsAndBytes.
"""

from __future__ import annotations

__version__ = "2.3.0"
__license__ = "MIT"

from cxrquant.clinical_safety.fact_extractor import (  # noqa: F401
    CHEXPERT_CATEGORIES,
    extract_labels,
    labels_to_binary,
)
from cxrquant.clinical_safety.safety_metrics import (  # noqa: F401
    clinical_efficacy_f1,
    fact_preservation,
)

__all__ = [
    "__version__",
    "CHEXPERT_CATEGORIES",
    "extract_labels",
    "labels_to_binary",
    "fact_preservation",
    "clinical_efficacy_f1",
]
