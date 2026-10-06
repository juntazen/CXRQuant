# Changelog

## 2.3.0 - IJIES second-round revision (first public release)

2.3.0 is the first public release of CXRQuant. Earlier versions (up to 2.2.0) were internal development
versions and were not publicly distributed; their entries below are kept for traceability.


- Decoding diagnosis: EOS doubled as the padding token for GPT-2 tokenizers and was penalised by the repetition
  penalty; new option `repetition_penalty_exempt_eos` and five-seed reruns (`scripts/regenerate_checkpoints.py`).
- Training-objective ablation: `loss_mask: impression_only` (`configs/full_seed42_impression_only.json`).
- Revision analyses (`scripts/revision_analyses.py`): data flow from the OpenI release, exact/near-duplicate overlap,
  denominators and bootstrap intervals for every precision, per-category transitions, per-category extractor
  validation, report-level, cluster-bootstrap, GEE and crossed random-effects models, repeated efficiency, tolerances.
- Randomised fresh-process efficiency re-measurement with CUDA events (`scripts/efficiency_repeat.py`).
- Confirmatory audits: LoRA-fine-tuned Mistral-7B (`scripts/train_lora_llm.py`) and CheXagent-8b on IU-XRAY and NIH
  ChestX-ray14 (`scripts/chexagent_audit.py`, `tools/nih_subset.py`).
- Robustness table (`scripts/robustness_table.py`) and GPU-result summaries (`scripts/summarise_reruns.py`).
- Corpus handled as identifiers plus rebuild scripts (`scripts/rebuild_corpus_from_openi.py`,
  `scripts/parse_openi_release.py`, `make data`); corpus origin traced to an archive-order image subset.
- Radiologist review analysis tools (`scripts/radiologist_analysis.py`, `scripts/radiologist_adjudication.py`).
- Archived revision outputs in `results_r2/`; documentation in `docs/REVISION_R2.md`.
- Release housekeeping: removed code paths not used by the published method (AWQ, GPTQ, EfficientQAT and the
  VLM wrapper; unused vocabulary and efficiency helpers), superseded compatibility scripts, an untested Dockerfile,
  pre-correction smoke runs and internal process records. No archived result changed.

## 2.2.0+ijies20260916 (internal development version, not released)

- Token-level decision-margin analysis regenerated from the retained checkpoints (`scripts/logit_margin_analysis.py`);
  all 3,216 seed-42 predictions reproduced exactly.
- Word-level divergence onset and flip localisation for all five runs (`scripts/divergence_analysis.py`).
- pyConTextNLP differential audit on its nine categories (`scripts/pycontext_audit.py`).
- Journal figures grouped by generation-quality stratum (`scripts/make_ijies_figures.py`).
- Portable Makefile targets `divergence`, `pycontext` and `margins`.

## 2.1.0+ijies20260915 (internal development version, not released)

- Four additional training seeds (123, 777, 2024, 31415) with the seed-42 split and held-out set; between-seed
  aggregation with hash and sample-order checks (`scripts/aggregate_seeds.py`).
- Efficiency tables from the recorded GPU timing and memory (`scripts/efficiency_summary.py`) and the
  efficiency-versus-stability figure (`scripts/make_efficiency_figure.py`).
- Ablations: uncertain-finding mapping (`scripts/ablation_uncertain_policy.py`) and generation-quality strata with
  floor-effect diagnostics (`scripts/ablation_stratum.py`; strata fixed on the primary run for replication seeds).
- Independent labellers: CheXbert (`cxrquant.clinical_safety.chexbert_labeler`, strict loading, pinned checkpoint)
  and the original CheXpert labeller via `scripts/chexpert_labeler_bridge.py`; three-way comparison and agreement
  (`scripts/baseline_evaluator_comparison.py`).
- Portable unattended seed queue for multi-GPU MIG servers and ordinary CUDA GPUs (`scripts/multiseed_queue.py`).
- Documentation for running without a GPU, on ordinary GPUs and on multi-GPU MIG servers; Makefile targets for all paths.
