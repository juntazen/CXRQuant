"""Causal LM encoding with explicit truncation and padding audits."""


def encode_training(tok, item, limit=256, mask_prompt=False):
    prompt = f"FINDINGS: {item['findings']} IMPRESSION:"
    target = " " + item["impression"]
    text = prompt + target + tok.eos_token
    ids = tok.encode(text, add_special_tokens=False)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    # BPE may merge the last prompt token with the first target token.
    boundary = 0
    for a, b in zip(ids, prompt_ids, strict=False):
        if a != b:
            break
        boundary += 1
    kept = ids[:limit]
    mask = [1] * len(kept) + [0] * (limit - len(kept))
    padded = kept + [tok.pad_token_id] * (limit - len(kept))
    labels = [token if valid else -100 for token, valid in zip(padded, mask, strict=True)]
    if mask_prompt:
        # Conditional objective: loss only on IMPRESSION tokens and EOS. The label at position
        # ``boundary`` is the first impression token, so earlier (prompt) positions are ignored.
        labels = [-100 if i < boundary else x for i, x in enumerate(labels)]
    valid_shifted = sum(x != -100 for x in labels[1:])
    impression_kept = max(0, min(len(ids) - 1, limit) - max(1, boundary))
    if not valid_shifted:
        raise ValueError("No target tokens remain after causal shift")
    return {"input_ids": padded, "attention_mask": mask, "labels": labels}, {
        "sample_id": item["uid"],
        "full_tokens": len(ids),
        "kept_tokens": len(kept),
        "truncated": len(ids) > limit,
        "impression_tokens_kept": impression_kept,
        "impression_fully_truncated": impression_kept == 0,
        "valid_shifted_tokens": valid_shifted,
    }


def encode_prompt(tok, findings, limit=224):
    full = tok.encode(f"FINDINGS: {findings} IMPRESSION:", add_special_tokens=False)
    suffix = tok.encode(" IMPRESSION:", add_special_tokens=False)
    prefix = tok.encode("FINDINGS:", add_special_tokens=False)
    if limit <= len(prefix) + len(suffix):
        raise ValueError("prompt limit cannot preserve both markers")
    if len(full) <= limit:
        return full, {"full_tokens": len(full), "truncated": False, "marker_preserved": True}
    # Reserve the final marker; keep the beginning of the findings as historically.
    kept = full[: limit - len(suffix)] + suffix
    return kept, {
        "full_tokens": len(full),
        "truncated": True,
        "marker_preserved": True,
        "legacy_marker_lost": not tok.decode(full[:limit]).rstrip().endswith("IMPRESSION:"),
    }
