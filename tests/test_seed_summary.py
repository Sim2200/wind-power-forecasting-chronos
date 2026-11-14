import json

import pytest

from wpf import seed_summary


def _record(seed, mase, wql=None):
    h = {"mae": 10.0 * seed, "rmse": 12.0, "mase": mase, "mase_turbines": 3, "n_valid": 100, "wql": wql}
    return {"config": {"seed": seed}, "model_type": "lstm", "val": {"6": h, "64": dict(h)}, "test": {"6": h, "64": dict(h)}}


def test_mean_and_sample_std_over_seeds():
    out = seed_summary.summarize([_record(1, 0.5, 0.3), _record(2, 0.7, 0.4), _record(3, 0.6, 0.5)])
    assert out["seeds"] == [1, 2, 3] and out["model_type"] == "lstm"
    assert out["val"]["6"]["mase"] == {"mean": pytest.approx(0.6), "std": pytest.approx(0.1), "n": 3}
    assert out["val"]["6"]["wql"]["mean"] == pytest.approx(0.4) and out["val"]["6"]["n_valid"] == 100


def test_null_metric_and_single_run():
    out = seed_summary.summarize([_record(1, 0.5), _record(2, 0.7)])
    assert out["val"]["6"]["wql"] is None
    one = seed_summary.summarize([_record(1, 0.5, 0.3)])
    assert one["val"]["6"]["mase"] == {"mean": 0.5, "std": None, "n": 1}


def test_mismatched_records_are_rejected():
    bad = _record(2, 0.7, 0.4)
    bad["val"]["6"]["n_valid"] = 99
    with pytest.raises(ValueError, match="different readings"):
        seed_summary.summarize([_record(1, 0.5, 0.3), bad])
    with pytest.raises(ValueError, match="no records"):
        seed_summary.summarize([])


def test_cli_writes_the_summary(tmp_path, monkeypatch):
    paths = []
    for seed in (1, 2):
        p = tmp_path / f"s{seed}.json"
        p.write_text(json.dumps(_record(seed, 0.5 + 0.1 * seed, 0.3)))
        paths.append(str(p))
    out = tmp_path / "seeds.json"
    monkeypatch.setattr("sys.argv", ["seed_summary", *paths, "--out", str(out)])
    seed_summary.main()
    got = json.loads(out.read_text())
    assert got["runs"] == paths and got["test"]["64"]["mase"]["n"] == 2
