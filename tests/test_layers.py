import pytest

torch = pytest.importorskip("torch")

from wpf.models.layers import Patcher, RevIN, merge_channels, num_patches, split_channels

SEQ_LEN, PATCH_LEN, STRIDE = 336, 16, 8


def _assert_round_trip(revin: RevIN, x: torch.Tensor) -> None:
    assert torch.allclose(revin(revin(x, "norm"), "denorm"), x, atol=1e-4)


def test_revin_normalizes_each_series():
    torch.manual_seed(0)
    x = torch.randn(4, SEQ_LEN, 7) * 5 + 3
    z = RevIN(7)(x, "norm")
    assert torch.allclose(z.mean(dim=1), torch.zeros(4, 7), atol=1e-4)
    assert torch.allclose(z.std(dim=1, unbiased=False), torch.ones(4, 7), atol=1e-3)


def test_revin_round_trip_is_exact():
    torch.manual_seed(0)
    x = torch.randn(4, SEQ_LEN, 7) * 10 - 2
    _assert_round_trip(RevIN(7), x)


def test_revin_round_trip_with_learned_affine():
    torch.manual_seed(0)
    x = torch.randn(2, 96, 3)
    revin = RevIN(3)
    with torch.no_grad():
        revin.affine_weight.copy_(torch.tensor([0.5, 2.0, 1.5]))
        revin.affine_bias.copy_(torch.tensor([0.1, -0.3, 0.0]))
    _assert_round_trip(revin, x)


def test_revin_stats_are_per_sample():
    # the same ramp at two different levels must normalize to the same series
    ramp = torch.arange(50.0).view(1, 50, 1)
    x = torch.cat([ramp, ramp + 100])
    z = RevIN(1, affine=False)(x, "norm")
    assert torch.allclose(z[0], z[1], atol=1e-5)


def test_num_patches_matches_official_setting():
    assert num_patches(SEQ_LEN, PATCH_LEN, STRIDE) == 42


def test_patcher_shape_and_content():
    x = torch.arange(float(SEQ_LEN)).view(1, 1, SEQ_LEN).repeat(2, 7, 1)
    # the last patch covers the final 8 real steps plus the last value repeated 8 times
    expected_last = torch.cat([torch.arange(328.0, 336.0), torch.full((8,), 335.0)])
    p = Patcher(PATCH_LEN, STRIDE)(x)
    assert p.shape == (2, 7, 42, PATCH_LEN)
    assert torch.equal(p[0, 0, 0], torch.arange(float(PATCH_LEN)))
    assert torch.equal(p[0, 0, 5], torch.arange(40.0, 56.0))
    assert torch.equal(p[0, 0, -1], expected_last)


def test_channel_merge_round_trip():
    n_vars = 7
    z = torch.randn(4, n_vars, 42, 16)
    m = merge_channels(z)
    assert m.shape == (28, 42, 16)
    assert torch.equal(split_channels(m, n_vars), z)


def test_channels_share_weights():
    # identical series in two channels must give identical outputs after a shared layer
    torch.manual_seed(0)
    n_vars = 2
    series = torch.randn(1, 1, SEQ_LEN)
    x = series.repeat(1, n_vars, 1)
    shared = torch.nn.Linear(PATCH_LEN, 8)
    out = split_channels(shared(merge_channels(Patcher()(x))), n_vars)
    assert torch.equal(out[0, 0], out[0, 1])
