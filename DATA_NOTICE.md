# Data notice

## IU-XRAY / OpenI (primary corpus)
- Source: Indiana University Chest X-ray Collection, distributed by the U.S. National Library of Medicine (Open-i),
  https://openi.nlm.nih.gov/ — Demner-Fushman et al., J Am Med Inform Assoc 23(2):304-310, 2016.
- Licence: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (as stated in every release record).
- This repository does **not** ship the parsed corpus. It ships `data/corpus_uids.txt` (781 report identifiers),
  `data/split_ids.csv` (partition of the 667 usable reports) and scripts that rebuild the corpus from the official
  release: `make data` downloads `NLMCXR_reports.tgz` (SHA-256
  8fb6de7eec73d8c3665067ad4bb003ccd57f971ae316d2642e1627ac7268667a), rebuilds `data/iuxray_paired.json`
  byte for byte (SHA-256 0597c49a…, the `data_hash` recorded in every run identity) from `data/corpus_image_ids.txt` and writes `external/openi_parsed.json`.
- Corpus origin: the 781 reports are those with a FINDINGS or IMPRESSION section among the 784 studies whose
  radiographs are the first 882 images of the official image archive (an interrupted download during the initial
  data preparation), i.e. an archive-order subset (`results_r2/corpus_origin_truncated_archive.json`).
- Archived run records and CheXagent outputs contain verbatim, unmodified report sentences (inputs and references)
  that are necessary to verify the published results; they are shared non-commercially with attribution.

## NIH ChestX-ray14 (confirmatory corpus)
- Source: NIH Clinical Center; Wang et al., "ChestX-ray8", IEEE CVPR 2017, pp. 3462-3471. Terms: usage unrestricted.
- Only image identifiers, labels and model outputs are included (`results_r2/nih_subset.json`,
  `results_r2/chexagent_nih/`); images are fetched with `tools/nih_subset.py`.

## Models
Pretrained models are downloaded from Hugging Face at pinned revisions (see `configs/`); fine-tuned checkpoints are
not included (hashes in `runs/*/models/*/training.json`).
