"""
NLG evaluation metrics for CXR report generation.
BLEU-1/4, ROUGE-L, METEOR, and clinical label metrics.
"""

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def clip_count(candidate_ngrams: Counter, references_ngrams: list[Counter]) -> int:
    max_ref = Counter()
    for ref in references_ngrams:
        for ng, cnt in ref.items():
            max_ref[ng] = max(max_ref[ng], cnt)
    clipped = {ng: min(cnt, max_ref[ng]) for ng, cnt in candidate_ngrams.items()}
    return sum(clipped.values())


def sentence_bleu(candidate: str, references: list[str], weights=(0.25, 0.25, 0.25, 0.25)) -> float:
    cand_tokens = tokenize(candidate)
    ref_tokens_list = [tokenize(r) for r in references]

    if not cand_tokens:
        return 0.0

    # Brevity penalty
    c = len(cand_tokens)
    r = min(abs(len(ref) - c) for ref in ref_tokens_list)
    r = min(len(ref) for ref in ref_tokens_list)
    bp = 1.0 if c >= r else math.exp(1 - r / c)

    log_score = 0.0
    for n, w in enumerate(weights, start=1):
        if w == 0:
            continue
        cand_ng = ngrams(cand_tokens, n)
        ref_ngs = [ngrams(rt, n) for rt in ref_tokens_list]
        clipped = clip_count(cand_ng, ref_ngs)
        total = max(len(cand_tokens) - n + 1, 0)
        if total == 0 or clipped == 0:
            log_score += w * float('-inf')
        else:
            log_score += w * math.log(clipped / total)

    if log_score == float('-inf'):
        return 0.0
    return bp * math.exp(log_score)


def corpus_bleu(candidates: list[str], references: list[list[str]], max_n: int = 4) -> dict[str, float]:
    """Corpus-level BLEU for n=1..max_n."""
    total_clipped = [0] * max_n
    total_cand = [0] * max_n
    c_total, r_total = 0, 0

    for cand, refs in zip(candidates, references, strict=True):
        cand_t = tokenize(cand)
        ref_t_list = [tokenize(r) for r in refs]
        c_total += len(cand_t)
        r_total += min(len(rt) for rt in ref_t_list)

        for n in range(1, max_n + 1):
            cng = ngrams(cand_t, n)
            rngs = [ngrams(rt, n) for rt in ref_t_list]
            total_clipped[n-1] += clip_count(cng, rngs)
            total_cand[n-1] += max(len(cand_t) - n + 1, 0)

    bp = 0.0 if c_total == 0 else (1.0 if c_total >= r_total else math.exp(1 - r_total / c_total))

    scores = {}
    log_avg = 0.0
    valid = True
    for n in range(1, max_n + 1):
        if total_cand[n-1] == 0 or total_clipped[n-1] == 0:
            scores[f'bleu_{n}'] = 0.0
            valid = False
        else:
            p = total_clipped[n-1] / total_cand[n-1]
            scores[f'bleu_{n}'] = bp * p
            log_avg += math.log(p) / max_n

    scores['bleu_4'] = bp * math.exp(log_avg) if valid else 0.0
    return scores


def lcs_length(a: list, b: list) -> int:
    """LCS via DP."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l(candidate: str, reference: str) -> float:
    cand_t = tokenize(candidate)
    ref_t = tokenize(reference)
    if not cand_t or not ref_t:
        return 0.0
    lcs = lcs_length(cand_t, ref_t)
    precision = lcs / len(cand_t)
    recall = lcs / len(ref_t)
    if precision + recall == 0:
        return 0.0
    beta = len(ref_t) / len(cand_t) if len(cand_t) > 0 else 1
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def rouge_n(candidate: str, reference: str, n: int = 1) -> float:
    cand_t = tokenize(candidate)
    ref_t = tokenize(reference)
    ref_ng = ngrams(ref_t, n)
    cand_ng = ngrams(cand_t, n)
    overlap = sum(min(cand_ng[ng], ref_ng[ng]) for ng in cand_ng)
    recall_denom = sum(ref_ng.values())
    return overlap / recall_denom if recall_denom > 0 else 0.0


def compute_corpus_rouge(candidates: list[str], references: list[str]) -> dict[str, float]:
    r1_scores = [rouge_n(c, r, 1) for c, r in zip(candidates, references, strict=True)]
    r2_scores = [rouge_n(c, r, 2) for c, r in zip(candidates, references, strict=True)]
    rl_scores = [rouge_l(c, r) for c, r in zip(candidates, references, strict=True)]
    return {
        'rouge_1': sum(r1_scores) / len(r1_scores) if r1_scores else 0.0,
        'rouge_2': sum(r2_scores) / len(r2_scores) if r2_scores else 0.0,
        'rouge_l': sum(rl_scores) / len(rl_scores) if rl_scores else 0.0,
    }


def evaluate_reports(
    candidates: list[str],
    references: list[str],
    verbose: bool = False,
) -> dict[str, float]:
    """Full NLG evaluation: BLEU-1/4 + ROUGE-1/2/L."""
    if len(candidates) != len(references) or not candidates:
        raise ValueError("expected nonempty, paired candidates/references")
    if any(not isinstance(t, str) for t in candidates + references):
        raise ValueError("reports must be strings (empty strings are valid predictions)")

    refs_wrapped = [[r] for r in references]
    bleu_scores = corpus_bleu(candidates, refs_wrapped, max_n=4)
    rouge_scores = compute_corpus_rouge(candidates, references)

    results = {**bleu_scores, **rouge_scores}

    if verbose:
        print('\n=== NLG Evaluation Results ===')
        for k, v in sorted(results.items()):
            print(f'  {k:12s}: {v:.4f}')

    return results


if __name__ == '__main__':
    # Quick sanity check
    cands = [
        "The lungs are clear. No pleural effusion. Normal cardiac silhouette.",
        "There is bilateral consolidation and pleural effusion.",
    ]
    refs = [
        "The lungs appear clear. No evidence of pleural effusion. Cardiac silhouette normal.",
        "Bilateral lung consolidation noted with pleural effusion present.",
    ]
    results = evaluate_reports(cands, refs, verbose=True)
