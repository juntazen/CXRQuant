# CXRQuant

Paired, finding-level stability auditing of reduced-precision inference for radiology impression generation.

**Release 2.3.0** — code and archived results for the IJIES article *"CXRQuant: Finding-Level Stability Auditing of
Reduced-Precision Inference for Radiology Impression Generation"* (paper ID 20266214, second review).
Archived at Zenodo with a persistent DOI (see the Zenodo record linked from the GitHub release page).

CXRQuant re-executes one trained **source checkpoint** under FP32, FP16, BitsAndBytes INT8 and NF4, pairs every
reduced-precision output with the FP32 output of the same checkpoint and input, and reports four separate axes:
lexical similarity (BLEU/ROUGE), FP32-relative finding stability (false-negative and false-positive flip rates,
FNF/FPF), reference-based category agreement (CE-F1) and measured efficiency. FP32 is a model-relative baseline, not
clinical ground truth.

## Two ways to use this repository

| Path | Hardware | Time | What it does |
|---|---|---|---|
| **CPU verification** (recommended first) | any laptop, no GPU | minutes | Re-derives every number, table and statistic of the paper from the archived predictions: verifier, bootstrap, McNemar, hierarchical models, robustness table, efficiency summaries, figures |
| **GPU regeneration** | NVIDIA GPU ≥ 40 GB (or an A100 server with MIG partitions) | hours | Retrains the six models and regenerates all precisions (`RUN_GPU.md`, `RUN_MULTI_GPU.md`); outputs can differ slightly across GPUs and kernels |

Both paths run the **same code** that produced the paper; there is no separate "CPU version" of the method.

## Quick start (CPU, no GPU)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-cpu-lock.txt -r requirements-revision.txt
pip install --no-deps --no-build-isolation -e .
make test            # unit and reproducibility tests
make verify-all      # recomputes every stored metric of the five training seeds
make data            # downloads the official OpenI report release (1 MB) and rebuilds the corpus
make revision-cpu    # revision analyses: data flow, overlap, CIs, hierarchical models, tolerances, figures
make revision-summary robustness   # summaries of the archived GPU experiments and the robustness table
```

Outputs are written to `results_local/` and can be compared with the archived `results_r2/`.

## What is inside

| Path | Content |
|---|---|
| `src/cxrquant/` | Extractor, metrics, training/generation pipeline, verifier, replay |
| `scripts/` | All analyses; `docs/REVISION_R2.md` maps each reviewer point to a script and output |
| `configs/` | Training/inference configurations: five seeds, EOS-exempt decoding, impression-only loss |
| `runs/` | Archived runs: 6 models × 4 precisions × 134 reports per seed (seeds 42, 123, 777, 2024, 31415), the impression-only loss ablation, replay and smoke runs; checkpoints excluded (hashes in `runs/*/models/*/training.json`) |
| `results_r2/` | Archived outputs of the second-round revision: statistics, robustness, EOS-exempt reruns (5 seeds), randomised efficiency re-measurement, Mistral-7B LoRA audit, CheXagent-8b on IU-XRAY and NIH ChestX-ray14, figures |
| `data/` | Corpus identifiers (`corpus_uids.txt`), partition identifiers (`split_ids.csv`); see `DATA_NOTICE.md` |
| `tests/` | Test suite (CPU) |

## Data

The IU-XRAY/OpenI reports are distributed by the U.S. National Library of Medicine under CC BY-NC-ND 4.0. This
repository ships identifiers and `scripts/rebuild_corpus_from_openi.py`, which rebuilds the 781-report corpus byte for
byte from the official release (SHA-256 checked against every run identity), not a modified copy of the corpus. Archived run records contain
verbatim, unmodified report sentences needed to verify the published numbers. NIH ChestX-ray14 is used without
redistribution (image identifiers only). Details: `DATA_NOTICE.md`.

## Authors

Junta Zeniarja, Guruh Fajar Shidik (corresponding author, guruh.fajar@research.dinus.ac.id), Heru Suhartanto,
Aris Marjuni and Ashraf Alomoush. Affiliations are listed in `CITATION.cff`.

## Funding

This work was supported by Kemdiktisaintek (the Directorate General of Higher Education, Research, and Technology, Ministry of Education, Culture, Research, and Technology, Republic of Indonesia) under Main Contract No. 127/C3/DT.05.00/PL/2025 and derivative contracts No. 028/LL6/PL/AL.04/2025 and No. 118/F.9-05/UDN-09/2025.

## Declaration of generative AI use

See `ai_disclosure/ACTUAL_USE.md`.

## Citation

See `CITATION.cff`. Please cite the IJIES article and this software release.

## Licence

Code: MIT (`LICENSE`). Data remain under the terms of their distributors (`DATA_NOTICE.md`).
