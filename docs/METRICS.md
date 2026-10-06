# Metric Definitions

Extractor: 14 CheXpert-style observation categories, dependency-free rules. Differential metrics and CE-F1: 13 categories, excluding No Finding. Default uncertain_policy=positive maps -1 to1, absent/unmentioned to0; negative maps uncertainty to0. Unsupported ignore is rejected rather than silently treated as negative. The comparator experiment uses only definite positives on nine MeSH-mappable categories and human findings+impression, a different validation task.

CFPR=unchanged/total category-report pairs; CFFR=flipped/total; FNF=lost/baseline-positive; FPF=gained/baseline-negative. Version2.1 returns JSON null for zero-denominator flip rates. Bootstrap samples reports with replacement, seed42,2000 resamples. Zero-denominator resamples are counted and excluded; an entirely undefined interval is null. Rates are rounded with Python round(x,4); raw integer counts are independently verified.

CE-F1 uses CXRQuant labels versus reference impressions, not a learned CheXbert model. Micro F1 aggregates TP/FP/FN; macro F1 averages all13 category F1s. A category with no positives is assigned F1=0 by the documented historical F1 convention; this is separate from the null flip-rate policy. Differential agreement is not clinical safety.

BLEU-1 and BLEU-4: corpus clipped n-gram counts, one reference per report, lowercasing, non-word punctuation replaced by spaces, whitespace tokens, brevity penalty, no smoothing. Empty predictions remain in the corpus and return zero BLEU when all candidates are empty. Intermediate bleu_2/bleu_3 helper outputs are modified precisions and are not claimed as cumulative BLEU-2/3.

ROUGE-1/2 are arithmetic means of per-report n-gram recall, not F1. ROUGE-L averages a longest-common-subsequence F-beta measure whose beta is reference length/candidate length; it is not standard fixed-beta ROUGE-L F1. No METEOR is claimed.

Pooled exact two-sided McNemar counts lost and gained pairs across categories. It is exploratory because categories are nested within reports. Per-category exact tests and Bonferroni flags across13 categories are also written. No unequal arrays are silently shortened.

Prospective GPU timing uses synchronized batch boundaries, explicit warmup and total measured seconds divided by total reports. Batch latency, amortized per-report latency, reports/s and output tokens/s are distinct. Peak allocated and reserved CUDA bytes are separate. Only checkpoint bytes actually saved are a measured artifact size; quantized compression ratios from bitwidth guesses are not reported.
