import pytest

torch = pytest.importorskip("torch")

from wpf.models.patchtst import PatchTST, PatchTSTConfig

SEQ_LEN = 336


@pytest.mark.parametrize("pred_len", [96, 192, 336, 720])
def test_forward_shape(pred_len):
    model = PatchTST(PatchTSTConfig(pred_len=pred_len)).eval()
    x = torch.randn(4, SEQ_LEN, 7)
    assert model(x).shape == (4, pred_len, 7)


def test_parameter_count_official_etth1_setting():
    # L=336, P=16, stride 8 -> 42 patches; D=16, 4 heads, d_ff=128, 3 layers, head Linear(672 -> 96)
    pred_len, patch_len, n_patches, d_model, d_ff, n_layers = 96, 16, 42, 16, 128, 3
    embed = patch_len * d_model + d_model
    pos = n_patches * d_model
    attn = 4 * (d_model * d_model + d_model)  # q, k, v, out
    norms = 2 * (2 * d_model)
    ff = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    head = d_model * n_patches * pred_len + pred_len
    expected = embed + pos + n_layers * (attn + norms + ff) + head
    model = PatchTST(PatchTSTConfig(pred_len=pred_len))
    assert model.num_parameters() == expected == 81_728


def test_eval_is_deterministic_and_train_uses_dropout():
    torch.manual_seed(0)
    model = PatchTST(PatchTSTConfig())
    x = torch.randn(2, SEQ_LEN, 7)
    model.eval()
    assert torch.equal(model(x), model(x))
    model.train()
    assert not torch.equal(model(x), model(x))


def test_gradients_reach_every_parameter():
    model = PatchTST(PatchTSTConfig()).train()
    x = torch.randn(8, SEQ_LEN, 7)
    model(x).pow(2).mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == []


def test_channels_are_independent_and_share_weights():
    # two identical input channels must produce identical forecasts
    torch.manual_seed(0)
    model = PatchTST(PatchTSTConfig(n_vars=2)).eval()
    s = torch.randn(1, SEQ_LEN, 1)
    x = torch.cat([s, s], dim=-1)
    y = model(x)
    assert torch.allclose(y[..., 0], y[..., 1], atol=1e-6)


def test_revin_restores_scale():
    # a constant offset on the input shows up as the same offset on the output
    torch.manual_seed(0)
    model = PatchTST(PatchTSTConfig()).eval()
    x = torch.randn(2, SEQ_LEN, 7)
    offset = 100.0
    assert torch.allclose(model(x + offset), model(x) + offset, atol=1e-2)
