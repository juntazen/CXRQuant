"""Small random model test of the loss contract, not a benchmark experiment."""

import pytest


def test_causal_model_loss_matches_shifted_masked_nll():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel, default_data_collator
    from cxrquant.training_data import encode_training

    class Tok:
        eos_token = "~"
        eos_token_id = pad_token_id = 126

        def encode(self, text, **kwargs):
            return [ord(c) for c in text]

    torch.manual_seed(42)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=128,
            n_positions=80,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=126,
            eos_token_id=126,
            pad_token_id=126,
        )
    )
    row, _ = encode_training(Tok(), {"uid": "toy", "findings": "clear", "impression": "normal"}, 80)
    batch = default_data_collator([{k: torch.tensor(v) for k, v in row.items()}])
    model.eval()
    result = model(**batch)
    labels = batch["labels"][:, 1:].contiguous()
    expected = torch.nn.functional.cross_entropy(
        result.logits[:, :-1].contiguous().view(-1, 128), labels.view(-1), ignore_index=-100
    )
    assert result.loss.item() == pytest.approx(expected.item(), rel=1e-6)
    result.loss.backward()
    assert model.transformer.wte.weight.grad is not None
