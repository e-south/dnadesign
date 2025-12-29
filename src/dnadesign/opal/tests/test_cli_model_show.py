"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_cli_model_show.py

CLI integration tests for model-show.

Module Author(s): Eric J. South (extended by Codex)
Dunlop Lab
--------------------------------------------------------------------------------
"""

import json

import numpy as np
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.registries.models import get_model


def test_model_show_with_model_meta(tmp_path):
    round_dir = tmp_path / "outputs" / "round_0"
    round_dir.mkdir(parents=True, exist_ok=True)
    model_path = round_dir / "model.joblib"

    model = get_model("random_forest", {"n_estimators": 5, "random_state": 0, "oob_score": False})
    X = np.array([[0.1, 0.2], [0.2, 0.3]])
    Y = np.array([[0.4], [0.6]])
    model.fit(X, Y)
    model.save(str(model_path))

    meta = {"model__name": "random_forest", "model__params": {"n_estimators": 5, "random_state": 0}}
    (round_dir / "model_meta.json").write_text(json.dumps(meta))

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "model-show", "--model-path", str(model_path), "--json"])
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["model_type"] == "random_forest"
    assert int(out["params"]["n_estimators"]) == 5
