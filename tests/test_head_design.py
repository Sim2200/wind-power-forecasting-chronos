import json

import pytest

from wpf import head_design


def _record(wql_by_h, seconds=100.0, parameters=1000):
    val = {str(h): {"mae": 10.0 * h, "rmse": 12.0 * h, "mase": 0.5, "mase_turbines": 3, "n_valid": 100 * h, "wql": w}
           for h, w in wql_by_h.items()}
    return {"val": val, "test": val, "train_seconds": seconds, "parameters": parameters}


def test_shared_wins_unless_every_added_model_beats_it_by_the_margin():
    shared = _record({6: 0.30, 36: 0.60, 64: 0.70}, seconds=300)
    h6, h36 = _record({6: 0.28}, seconds=100), _record({36: 0.55}, seconds=200)
    out = head_design.compare(shared, {6: h6, 36: h36, 64: shared}, margin=0.02)
    assert out["chosen"] == "per_horizon"
    assert out["horizons"]["6"]["wql_gain"] == pytest.approx((0.30 - 0.28) / 0.30)
    assert out["horizons"]["64"]["per_horizon"] == out["horizons"]["64"]["shared"]
    assert out["train_seconds"] == {"shared": 300, "per_horizon": 600}
    assert out["su"] == {"shared": pytest.approx(300 / 3600 * 7, abs=0.01), "per_horizon": pytest.approx(600 / 3600 * 7, abs=0.01)}
    assert head_design.compare(shared, {6: h6, 36: h36, 64: shared}, su_per_hour=48)["su"]["shared"] == 4.0
    close = _record({6: 0.299})  # within the margin at H = 6 -> shared
    assert head_design.compare(shared, {6: close, 36: h36, 64: shared})["chosen"] == "shared"
    worse = _record({36: 0.65})
    assert head_design.compare(shared, {6: h6, 36: worse, 64: shared})["chosen"] == "shared"


def test_compare_needs_validation_metrics_for_the_horizon():
    shared = _record({6: 0.3, 36: 0.6, 64: 0.7})
    with pytest.raises(ValueError, match="H = 36"):
        head_design.compare(shared, {36: _record({6: 0.2})})


def test_cli_writes_the_decision(tmp_path, monkeypatch, capsys):
    shared = _record({6: 0.30, 36: 0.60, 64: 0.70})
    paths = {}
    for name, rec in (("shared", shared), ("h6", _record({6: 0.20})), ("h36", _record({36: 0.40}))):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(rec))
    out = tmp_path / "decision.json"
    monkeypatch.setattr("sys.argv", ["head_design", "--shared", str(paths["shared"]), "--h6", str(paths["h6"]),
                                     "--h36", str(paths["h36"]), "--out", str(out)])
    head_design.main()
    got = json.loads(out.read_text())
    assert got["chosen"] == "per_horizon" and set(got["horizons"]) == {"6", "36", "64"}
    assert "chosen: per_horizon" in capsys.readouterr().out
