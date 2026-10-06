"""Optional CheXbert labeller (Smit et al., EMNLP 2020) as an independent extractor.

Maps free text to the same 14 CHEXPERT_CATEGORIES and the same {1, 0, -1, None}
encoding as ``fact_extractor.extract_labels`` so both extractors can feed the
unchanged metric layer. The checkpoint is not bundled; it is resolved from the
Hugging Face Hub (``StanfordAIMI/RRG_scorers``, file ``chexbert.pth``) at a
pinned revision, or from ``CXRQUANT_CHEXBERT_CHECKPOINT``. The architecture and
class order follow the reference implementation (stanfordmlgroup/CheXbert,
``src/models/bert_labeler.py`` and ``src/label.py``): BERT-base CLS token,
13 four-class heads (blank, positive, negative, uncertain) and one two-class
No Finding head (blank, positive).

Unlike the reference loader, weights are loaded with ``strict=True`` so that a
checkpoint/architecture mismatch fails instead of leaving heads at random
initialisation. torch/transformers are imported lazily; this module is an
optional dependency and the primary extractor never imports it.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

from .fact_extractor import CHEXPERT_CATEGORIES

CHECKPOINT_REPO = "StanfordAIMI/RRG_scorers"
CHECKPOINT_FILE = "chexbert.pth"
CHECKPOINT_REVISION = "6646433b3ad83a10f6e141db76d0ece44312b236"
TOKENIZER_ID = "bert-base-uncased"
MAX_TOKENS = 512

# Head order of the CheXbert checkpoint (No Finding last).
CHEXBERT_ORDER: list[str] = CHEXPERT_CATEGORIES[1:] + [CHEXPERT_CATEGORIES[0]]
# Head class index -> CXRQuant encoding.
CLASS_TO_LABEL = {0: None, 1: 1, 2: 0, 3: -1}


def checkpoint_path() -> str:
    env = os.environ.get("CXRQUANT_CHEXBERT_CHECKPOINT")
    if env:
        return env
    from huggingface_hub import hf_hub_download

    return hf_hub_download(CHECKPOINT_REPO, CHECKPOINT_FILE, revision=CHECKPOINT_REVISION)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ")).strip()


def decode(classes: list[int]) -> dict[str, int | None]:
    """Convert 14 head argmax values (checkpoint order) to CXRQuant labels."""
    if len(classes) != len(CHEXBERT_ORDER):
        raise ValueError("expected 14 CheXbert head outputs")
    labels = {}
    for cat, cls in zip(CHEXBERT_ORDER, classes, strict=True):
        if cls not in CLASS_TO_LABEL or (cat == "No Finding" and cls not in (0, 1)):
            raise ValueError(f"invalid class {cls} for {cat}")
        labels[cat] = CLASS_TO_LABEL[cls]
    return {c: labels[c] for c in CHEXPERT_CATEGORIES}


class CheXbertLabeler:
    def __init__(self, device: str = "cpu", checkpoint: str | None = None):
        import torch
        from torch import nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        class _BertLabeler(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = AutoModel.from_config(AutoConfig.from_pretrained(TOKENIZER_ID))
                hidden = self.bert.config.hidden_size
                self.dropout = nn.Dropout(0.1)
                self.linear_heads = nn.ModuleList(
                    [nn.Linear(hidden, 4) for _ in range(13)] + [nn.Linear(hidden, 2)]
                )

            def forward(self, input_ids, attention_mask):
                cls = self.bert(input_ids, attention_mask=attention_mask)[0][:, 0, :]
                cls = self.dropout(cls)
                return [head(cls) for head in self.linear_heads]

        self.device = torch.device(device)
        self.checkpoint = checkpoint or checkpoint_path()
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        state = state.get("model_state_dict", state)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model = _BertLabeler()
        expected = set(model.state_dict())
        # Buffers such as position_ids differ across transformers versions and are not weights.
        buffers = {k for k in expected ^ set(state) if k.endswith("position_ids")}
        state = {k: v for k, v in state.items() if k not in buffers}
        for k in buffers & expected:
            state[k] = model.state_dict()[k]
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    def label_classes(self, texts: list[str], batch_size: int = 32) -> list[list[int]]:
        import torch

        out = []
        for i in range(0, len(texts), batch_size):
            chunk = [_normalise(t) for t in texts[i : i + batch_size]]
            enc = self.tokenizer(
                chunk, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                heads = self.model(enc["input_ids"], enc["attention_mask"])
            out.extend(torch.stack([h.argmax(dim=1) for h in heads], dim=1).tolist())
        return out

    def extract(self, texts: list[str], batch_size: int = 32) -> list[dict[str, int | None]]:
        return [decode(c) for c in self.label_classes(texts, batch_size)]


@lru_cache(maxsize=1)
def _default_labeler() -> CheXbertLabeler:
    return CheXbertLabeler(device=os.environ.get("CXRQUANT_CHEXBERT_DEVICE", "cpu"))


def extract_labels_chexbert(text: str) -> dict[str, int | None]:
    """Single-report convenience wrapper mirroring ``extract_labels``."""
    return _default_labeler().extract([text])[0]
