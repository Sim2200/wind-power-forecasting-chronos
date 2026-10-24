import pytest

torch = pytest.importorskip("torch")

from wpf.models.layers import Patcher, RevIN, merge_channels, num_patches, split_channels


def test_revin_normalizes_each_series():
    torch.manual_seed(0)
    x = torch.randn(4, 336, 7) * 5 + 3
    z = RevIN(7)(x, "norm")
    assert torch.allclose(z.mean(dim=1), torch.zeros(4, 7), atol=1e-4)
    assert torch.allclose(z.std(dim=1, unbiased=False), torch.ones(4, 7), atol=1e-3)


def test_revin_round_trip_is_exact():
    torch.manual_seed(0)
    x = torch.randn(4, 336, 7) * 10 - 2
    revin = RevIN(7)
    assert torch.allclose(revin(revin(x, "norm"), "denorm"), x, atol=1e-4)


def test_revin_round_trip_with_learned_affine():
    torch.manual_seed(0)
    x = torch.randn(2, 96, 3)
    revin = RevIN(3)
    with torch.no_grad():
        revin.affine_weight.copy_(torch.tensor([0.5, 2.0, 1.5]))
        revin.affine_bias.copy_(torch.tensor([0.1, -0.3, 0.0]))
    assert torch.allclose(revin(revin(x, "norm"), "denorm"), x, atol=1e-4)


def test_revin_stats_are_per_sample():
    a = torch.zeros(1, 50, 1)
    b = torch.full((1, 50, 1), 100.0) + torch.arange(50.0).view(1, 50, 1)
    z = RevIN(1, affine=False)(torch.cat([a + torch.arange(50.0).view(1, 50, 1), b]), "norm")
    assert torch.allclose(z[0], z[1], atol=1e-5)  # same shape, different level -> same normalized series


def test_num_patches_matches_official_setting():
    assert num_patches(336, 16, 8) == 42
    assert num_patches(336, 16, 8, pad_end=False) == 41


def test_patcher_shape_and_content():
    x = torch.arange(336.0).view(1, 1, 336).repeat(2, 7, 1)
    p = Patcher(16, 8)(x)
    assert p.shape == (2, 7, 42, 16)
    assert torch.equal(p[0, 0, 0], torch.arange(16.0))
    assert torch.equal(p[0, 0, 5], torch.arange(40.0, 56.0))
    # the last patch covers the final 8 real steps plus the last value repeated 8 times
    assert torch.equal(p[0, 0, -1], torch.cat([torch.arange(328.0, 336.0), torch.full((8,), 335.0)]))


def test_channel_merge_round_trip():
    z = torch.randn(4, 7, 42, 16)
    m = merge_channels(z)
    assert m.shape == (28, 42, 16)
    assert torch.equal(split_channels(m, 7), z)


def test_channels_share_weights():
    # identical series in two channels must give identical outputs after a shared layer
    torch.manual_seed(0)
    series = torch.randn(1, 1, 336)
    x = series.repeat(1, 2, 1)
    p = Patcher()(x)
    out = split_channels(nn_linear(merge_channels(p)), 2)
    assert torch.equal(out[0, 0], out[0, 1])


def nn_linear(z):
    torch.manual_seed(1)
    return torch.nn.Linear(16, 8)(z)
