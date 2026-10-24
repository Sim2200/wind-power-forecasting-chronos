import pytest

torch = pytest.importorskip("torch")

from wpf.models.patchtst import PatchTST, PatchTSTConfig


@pytest.mark.parametrize("pred_len", [96, 192, 336, 720])
def test_forward_shape(pred_len):
    model = PatchTST(PatchTSTConfig(pred_len=pred_len)).eval()
    y = model(torch.randn(4, 336, 7))
    assert y.shape == (4, pred_len, 7)


def test_parameter_count_official_etth1_setting():
    # L=336, P=16, stride 8 -> 42 patches; D=16, 4 heads, d_ff=128, 3 layers, head Linear(672 -> 96)
    model = PatchTST(PatchTSTConfig(pred_len=96))
    embed = 16 * 16 + 16
    pos = 42 * 16
    attn = 4 * (16 * 16 + 16)  # q, k, v, out
    norms = 2 * (2 * 16)
    ff = (16 * 128 + 128) + (128 * 16 + 16)
    head = 672 * 96 + 96
    assert model.num_parameters() == embed + pos + 3 * (attn + norms + ff) + head == 81_728


def test_eval_is_deterministic_and_train_uses_dropout():
    torch.manual_seed(0)
    model = PatchTST(PatchTSTConfig())
    x = torch.randn(2, 336, 7)
    model.eval()
    assert torch.equal(model(x), model(x))
    model.train()
    assert not torch.equal(model(x), model(x))


def test_gradients_reach_every_parameter():
    model = PatchTST(PatchTSTConfig()).train()
    loss = model(torch.randn(8, 336, 7)).pow(2).mean()
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == []


def test_channels_are_independent_and_share_weights():
    # two identical input channels must produce identical forecasts
    torch.manual_seed(0)
    model = PatchTST(PatchTSTConfig(n_vars=2)).eval()
    s = torch.randn(1, 336, 1)
    y = model(torch.cat([s, s], dim=-1))
    assert torch.allclose(y[..., 0], y[..., 1], atol=1e-6)


def test_revin_restores_scale():
    # a constant offset on the input shows up as the same offset on the output
    torch.manual_seed(0)
    model = PatchTST(PatchTSTConfig()).eval()
    x = torch.randn(2, 336, 7)
    assert torch.allclose(model(x + 100.0), model(x) + 100.0, atol=1e-2)
