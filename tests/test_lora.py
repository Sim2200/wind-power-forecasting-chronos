import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from transformers import T5Config, T5ForConditionalGeneration

from wpf import lora


@pytest.fixture
def tiny_t5():
    torch.manual_seed(0)
    cfg = T5Config(vocab_size=64, d_model=16, d_kv=4, d_ff=32, num_layers=2, num_decoder_layers=2, num_heads=4,
                   decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    return T5ForConditionalGeneration(cfg).eval()


def test_only_the_adapters_train_and_their_count_matches_the_formula(tiny_t5):
    total_before = lora.count_parameters(tiny_t5)["total"]
    expected = lora.expected_lora_parameters(tiny_t5, rank=4)  # before wrapping: `apply_lora` mutates in place
    wrapped = lora.apply_lora(tiny_t5, rank=4)
    counts = lora.count_parameters(wrapped)
    assert counts["trainable"] == expected
    assert counts["total"] == total_before + counts["trainable"]
    assert all(("lora_" in n) == p.requires_grad for n, p in wrapped.named_parameters())
    # 2 encoder + 2 decoder self-attention blocks + 2 cross-attention blocks, 4 projections each, 16 x 16 matrices
    assert counts["trainable"] == 6 * 4 * 4 * (16 + 16)


def test_merge_reproduces_the_adapted_forward_pass_in_a_plain_model(tiny_t5):
    wrapped = lora.apply_lora(tiny_t5, rank=4, dropout=0.0)
    with torch.no_grad():  # make the adapters non-trivial: B is zero-initialised by peft
        for n, p in wrapped.named_parameters():
            if "lora_B" in n:
                p.normal_(std=0.1)
    ids = torch.randint(2, 64, (3, 10))
    labels = torch.randint(2, 64, (3, 5))
    wrapped.eval()
    with torch.no_grad():
        adapted = wrapped(input_ids=ids, labels=labels).logits
        plain = lora.merge(wrapped)
        merged = plain(input_ids=ids, labels=labels).logits
    assert isinstance(plain, T5ForConditionalGeneration)
    assert not any("lora_" in n for n, _ in plain.named_parameters())
    assert torch.allclose(adapted, merged, atol=1e-5)


def test_the_adapted_model_differs_from_the_base_once_b_is_nonzero(tiny_t5):
    ids = torch.randint(2, 64, (2, 8))
    labels = torch.randint(2, 64, (2, 4))
    with torch.no_grad():
        base = tiny_t5(input_ids=ids, labels=labels).logits.clone()
    wrapped = lora.apply_lora(tiny_t5, rank=2, dropout=0.0).eval()
    with torch.no_grad():
        assert torch.allclose(wrapped(input_ids=ids, labels=labels).logits, base)  # B = 0: identical at start
        for n, p in wrapped.named_parameters():
            if "lora_B" in n:
                p.normal_(std=0.1)
        assert not torch.allclose(wrapped(input_ids=ids, labels=labels).logits, base)
