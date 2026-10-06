# Running on a multi-GPU server with MIG partitions (environment of the archived runs)

All five archived full runs were executed in a container on an NVIDIA A100 server with two MIG 7g.40gb instances,
PyTorch 2.7.1+cu118, Transformers 5.10.2 and BitsAndBytes 0.49.2 under driver 525.147.05.

## Container-specific CUDA workaround

The system `libcuda.so.525.147.05` in this container is a zero-byte file. A valid library matching the driver is
provided under `/usr/local/cuda-12.0/compat`:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/compat
```

`scripts/multiseed_queue.py` sets this automatically when the directory exists (disable with
`CXRQUANT_USE_CUDA_COMPAT=0`). On other machines it is not needed.

## One process per MIG instance

NVML lists both MIG UUIDs, but the CUDA runtime creates a context on only one device per process. The launcher
therefore never initialises CUDA itself; it splits the six models round-robin between two worker processes and
pins each worker to one MIG UUID through `CUDA_VISIBLE_DEVICES`. The two 40-GB instances are never treated as one
80-GB device.

## Seed 42 (archived primary run)

```bash
CUDA_VISIBLE_DEVICES=MIG-<uuid-A> \
python -m cxrquant.pipeline --mode preflight --run-dir runs/largest_preflight_compat_20260911 --config configs/full.json

python -m cxrquant.pipeline --mode full --run-dir runs/full_seed42_corrected_20260911 \
  --config configs/full.json --data data/iuxray_paired.json
```

Completed in 1,278.7 s: 6 models, 24 configurations, 3,216 predictions, 18 comparisons, worker exit codes [0, 0].

## Seeds 123, 777, 2024 and 31415 (IJIES extension, 15 September 2026)

Executed unattended with the queue. One MIG instance also carried about 5 GB used by a process outside the
container, so the idle threshold was raised and the device order was set so that the worker with BioGPT-Large used
the instance without that load:

```bash
PYTHONPATH=src python scripts/multiseed_queue.py --tag 20260915 \
  --state ../audit/multiseed_queue_state.json --max-used-mib 6144 --poll 30 --min-devices 2 \
  --devices "MIG-<uuid-B>,MIG-<uuid-A>"
```

| Seed | Duration | Matrix | Worker exit codes |
|---|---|---|---|
| 123 | 20.9 min | 6 / 24 / 3,216 / 18 | [0, 0] |
| 777 | 21.8 min | 6 / 24 / 3,216 / 18 | [0, 0] |
| 2024 | 21.0 min | 6 / 24 / 3,216 / 18 | [0, 0] |
| 31415 | 21.2 min | 6 / 24 / 3,216 / 18 | [0, 0] |

All runs share the seed-42 data hash, split identifiers and held-out sample order; configurations differ only in
`training_seed` and `data_seed`. Per-token latency varied by 0.4–6.4% (coefficient of variation) across the five
seeds, including on the instance with the external load. The queue state and logs are in `audit/ijies_20260915/`.

## Resume and identity

Resume uses the same full command plus `--resume`. Changed code, configuration or data are refused. Completed
results are reused only when run identity and checkpoint hash match. Checkpoints stay in
`runs/<run>/checkpoints/` on the execution machine and are excluded from the packages.

## Token-level margin analysis (16 September 2026)

Run in parallel on the two MIG instances while an external workload also used them (memory capped at 25% and 15%
of the 40-GB instance). All 3,216 regenerated predictions matched the archived seed-42 predictions exactly.

```bash
CUDA_VISIBLE_DEVICES=MIG-<uuid-A> python scripts/logit_margin_analysis.py \
  --run-dir runs/full_seed42_corrected_20260911 --checkpoint-root <checkpoints> \
  --models gpt2-medium,pythia-1b,biogpt-large --memory-fraction 0.25 --out logit_margin_gpu0.json
CUDA_VISIBLE_DEVICES=MIG-<uuid-B> python scripts/logit_margin_analysis.py \
  --run-dir runs/full_seed42_corrected_20260911 --checkpoint-root <checkpoints> \
  --models distilgpt2,gpt2,biogpt --memory-fraction 0.15 --out logit_margin_gpu1.json
```

The two outputs were merged into `runs/full_seed42_corrected_20260911/logit_margin_analysis.json`.
