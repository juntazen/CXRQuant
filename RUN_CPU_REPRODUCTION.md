# Reproduction without a GPU

Everything in this guide runs on a CPU from the files shipped in the package. It does not retrain models or
regenerate text; it re-derives every reported number from the stored predictions. Validated on Linux with
Python 3.13 in a clean extraction of this package.

## 1. Install

```bash
python3 -m venv .venv
. .venv/bin/activate                      # Windows: .venv\Scripts\activate
pip install -r requirements-cpu-lock.txt
pip install --no-deps --no-build-isolation -e .
```

If `make` is not available, run the commands shown inside each target of the `Makefile` directly.

## 2. Tests and numerical verification

```bash
make test          # pytest; tests needing torch/transformers or the CheXbert checkpoint are skipped
make verify-all    # recomputes every stored metric of the five full runs from their predictions
```

`verify-all` must end with "all stored clinical and NLG metrics match recomputation" for each run.

## 3. Statistics replay

```bash
make replay RUN=runs/full_seed123_20260915
```

Replay regenerates bootstrap intervals, pooled and per-category McNemar tests, tables and the FNF heat map in the
run directory. The numerical artefacts are deterministic; PNG bytes may differ across font or matplotlib
versions. Work on a copy of the run directory if you want to compare files byte for byte.

## 4. Extension analyses (efficiency, ablations, seeds, figure)

```bash
make analyses      # efficiency_summary, ablation_uncertain_policy, ablation_stratum for all five runs
make aggregate     # between-seed summary into runs/multiseed_reproduced/
make figure        # efficiency-versus-stability figure into figures_reproduced/
```

- Efficiency values are read from the timing and memory records written during GPU generation; they are not
  re-measured on your CPU.
- The uncertain-policy ablation asserts that the primary mapping reproduces the archived metrics exactly.
- Strata are defined on the seed-42 run and reused for the other seeds; each run also reports its own FP32 BLEU-4
  class as a diagnostic.
- `make aggregate` refuses to overwrite an existing summary; delete `runs/multiseed_reproduced/` to repeat it.

`make divergence` locates, for every report and precision, the first word at which the output departs from FP32
and whether each flipped category was already mentioned before that point (all five runs, CPU only).

## 5. Independent labellers (optional)

CheXbert runs on CPU (about 90 s for 1,939 texts on a 128-thread machine):

```bash
pip install -r requirements-labellers-cpu.txt
make labellers
```

The checkpoint is downloaded from `StanfordAIMI/RRG_scorers` at a pinned revision, loaded with strict weight
matching and its SHA-256 recorded. Set `CXRQUANT_CHEXBERT_CHECKPOINT=/path/chexbert.pth` to use a local copy.

The original CheXpert labeller needs its own Python 3.6 environment (NegBio, BLLIP parser, Stanford CoreNLP, Java).
Its labels for this package are already stored in `runs/full_seed42_corrected_20260911/chexpert_labels.json`.
To regenerate them, create the environment from the labeller's `environment.yml`, then:

```bash
python scripts/chexpert_labeler_bridge.py export --run-dir runs/full_seed42_corrected_20260911 --work-dir work/chexpert
python scripts/chexpert_labeler_bridge.py label  --run-dir runs/full_seed42_corrected_20260911 --work-dir work/chexpert \
  --env /path/to/chexpert-env --labeler /path/to/chexpert-labeler --negbio /path/to/NegBio --nltk-data /path/to/nltk_data
python scripts/chexpert_labeler_bridge.py import --run-dir runs/full_seed42_corrected_20260911 --work-dir work/chexpert
```

If a labeller is unavailable, `baseline_evaluator_comparison.py` records it as unavailable and imputes nothing.

pyConTextNLP (`pip install pyConTextNLP networkx`) re-runs the audit on the nine categories of its rule set:

```bash
make pycontext
```

## 6. Extractor comparison against MeSH silver labels

```bash
python scripts/extractor_comparison.py --data data/iuxray_paired.json --run-dir runs/archive_replay_20260910
```

pyConTextNLP is used if installed; medspaCy is recorded as unavailable when absent.

## What a CPU cannot reproduce

Training and generation require CUDA (`cxrquant.pipeline --mode smoke|full`). The stored predictions, checkpoint
hashes, effective training arguments and token audits allow those steps to be repeated on a GPU; see
[RUN_GPU.md](RUN_GPU.md).
