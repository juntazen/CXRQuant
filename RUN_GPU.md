# Running on ordinary CUDA GPUs

This guide retrains the six models and regenerates all four precision settings on any Linux machine with NVIDIA
GPUs. The archived runs were produced on two NVIDIA A100 MIG 7g.40gb instances ([RUN_MULTI_GPU.md](RUN_MULTI_GPU.md)); on other hardware the
predictions can differ slightly because GPU kernels are not bit-identical across devices, and latency and memory
are specific to your hardware. Compare your new run with the archived runs using the CPU tools; do not overwrite
the archived run directories.

## Requirements

- NVIDIA driver with CUDA 11.8 or newer support, Python 3.10+.
- Memory: the largest model (BioGPT-Large, 1.57 B parameters, full FP32 fine-tuning with AdamW) needs about
  30–35 GB on one device. A 40-GB or larger GPU runs `configs/full.json` unchanged. On smaller GPUs, run
  `configs/smoke.json`, or copy `configs/full.json` and remove the larger entries from `models` (the change is
  recorded in the run identity).
- Disk: about 92 GB per full run for checkpoints (`runs/<run>/checkpoints/`), which can be deleted after scoring
  if the hashes in `models/*/training.json` are sufficient for you.
- Internet access to download the pinned model revisions from Hugging Face (no token required).

## Install

```bash
python3 -m venv .venv-gpu && . .venv-gpu/bin/activate
pip install -r requirements-gpu-observed.txt     # versions used for the archived runs (CUDA 11.8 wheels)
pip install --no-deps --no-build-isolation -e .
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## 1. Preflight and smoke test

```bash
make preflight                      # BioGPT-Large forward/backward on the first visible GPU
make smoke                          # one model, four precisions, few reports
```

## 2. A full run or an additional seed

```bash
make full SEED=42                   # uses configs/full.json
make full SEED=123                  # uses configs/full_seed123.json (split and evaluation set unchanged)
```

The launcher uses at most two visible GPUs and splits the six models between them; with one GPU all models run
sequentially in one worker. Select devices with `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=0,1`. An
interrupted run is continued with the same command plus `RESUME=1`; a run whose code, configuration or data
changed is refused and needs a new directory.

Each full run ends with automatic verification and statistics replay. A complete run reports
6 models, 24 configurations, 3,216 predictions and 18 comparisons in `runs/<run>/status.json`.

## 3. All four additional seeds unattended

```bash
make multiseed-queue TAG=$(date +%Y%m%d)
```

The queue waits until the visible GPUs are idle (at most 1 GiB used), runs a preflight, then seeds 123, 777,
2024 and 31415 one after another, stops on the first failure, and finally runs the analyses and the between-seed
aggregation with the archived seed-42 run. Progress is written to `audit/multiseed_queue_state.json`.

## 4. Analyse the new runs on CPU

```bash
python scripts/efficiency_summary.py --run-dir runs/<new run>
python scripts/ablation_uncertain_policy.py --run-dir runs/<new run>
python scripts/ablation_stratum.py --run-dir runs/<new run>
python scripts/aggregate_seeds.py --run-dirs runs/full_seed42_corrected_20260911,runs/<new run> --out-dir runs/<new summary>
```

## 5. Token-level decision margins (optional)

With the checkpoints of a run available, the outputs can be regenerated while recording the log-probability margin
between the two most likely tokens at every step:

```bash
make margins CHECKPOINTS=runs/full_seed42_corrected_20260911/checkpoints
```

The script verifies each checkpoint hash, caps its GPU memory share (`--memory-fraction`), reports how many
regenerated outputs match the archived predictions and writes the margins at the first divergent token. Two
processes, one per device, can be started with different `--models` lists to run in parallel.
