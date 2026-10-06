# IJIES second-round revision (paper ID 20266214)

All analyses added for the second review read the archived runs and write new files; archived run directories are
never modified. Outputs listed below are archived in `results_r2/` (or written to `results_local/` by the CPU targets).

| Reviewer point | Script | Output |
|---|---|---|
| R1-3 data flow, R1-2 overlap | `scripts/revision_analyses.py` | `revision_analyses.json`, `split_ids.csv`, `overlap_sensitivity_all_seeds.csv` |
| R1-7 denominators, CIs, per-category transitions | same | `denominators_ci_all_precisions.csv`, `per_category_transitions.csv` |
| R1-6 per-category validation, low-agreement categories | same | `revision_analyses.json` (`extractor`, `restricted_categories`) |
| R2-4 report-level and hierarchical statistics | same (statsmodels) | `revision_analyses.json` (`hierarchical`) |
| R1-8 repeated efficiency (five runs) | same | `efficiency_all_runs.csv`, `efficiency_repeated_summary.csv` |
| R1-8 randomised fresh-process re-measurement (GPU) | `scripts/efficiency_repeat.py` | `efficiency_repeat/summary.json` |
| R2-6 tolerances | same | `revision_analyses.json` (`tolerance`) |
| R1-4 decoding diagnosis and corrected rerun (GPU) | `scripts/regenerate_checkpoints.py`, `repetition_penalty_exempt_eos` in `src/cxrquant/pipeline.py` | `eosfix/seed*/models/*/*.json` |
| R1-1 impression-only loss ablation (GPU) | `loss_mask` in `src/cxrquant/training_data.py`, `configs/full_seed42_impression_only.json` | `runs/ablation_impression_only_seed42/` |
| R2-2 stronger text generator (GPU) | `scripts/train_lora_llm.py` + `regenerate_checkpoints.py` | `llm7b/` |
| R2-2 image-conditioned generator (GPU) | `scripts/chexagent_audit.py` (Python 3.10, transformers 4.40.2) | `chexagent/*.jsonl` |
| R2-3 / R1-6 blinded radiologist reading protocol (released for future use; no radiologist review is reported in this revision) | `scripts/radiologist_adjudication.py`, `scripts/radiologist_analysis.py` | `docs/RADIOLOGIST_READING_PROTOCOL.md` |
| Summary of GPU experiments | `scripts/summarise_reruns.py` | `rerun_summary.json` |
| Figures (editor format) | `scripts/make_revision_figures.py` | `figures/fig1_workflow.png`, `fig2_fnf.png`, `fig3_axes.png` |
| Corpus rebuild (CC BY-NC-ND 4.0) | `scripts/rebuild_corpus_from_openi.py` | verified 781/781 identical |

Both new options default to the archived behaviour: `loss_mask` defaults to `full_sequence` and
`repetition_penalty_exempt_eos` to `false`, so the archived runs replay unchanged.

Container note (multi-GPU MIG server): the container's `/usr/lib/x86_64-linux-gnu/libcuda.so.1` was a 0-byte file after a restart;
the GPU runs used `LD_LIBRARY_PATH=/usr/local/cuda-12.0/compat` (same driver version, 525.147.05).

## Added after the GPU phase
| Item | Script | Output |
|---|---|---|
| NIH ChestX-ray14 subset (second corpus, 300 patients) | `tools/nih_subset.py` (HTTP range reads of images_001.zip) | `results_r2/nih_subset.json` |
| CheXagent-8b on NIH | `scripts/chexagent_audit.py --source nih` | `chexagent_nih/*.jsonl` |
| Robustness table | `scripts/robustness_table.py` | `robustness.json` |
| Corpus origin | first 882 images of the official NLMCXR_png.tgz (interrupted download) | `corpus_origin_truncated_archive.json` |
