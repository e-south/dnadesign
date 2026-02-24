"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_analyze_metadata.py

Validate analysis metadata loaders preserve PWM scoring contracts from sample runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from dnadesign.cruncher.app.analyze.metadata import load_pwms_from_config
from dnadesign.cruncher.app.sample.artifacts import _save_config
from dnadesign.cruncher.config.schema_v3 import CatalogConfig, CruncherConfig, WorkspaceConfig
from dnadesign.cruncher.core.pwm import PWM


def _write_config_used(run_dir: Path, payload: dict) -> None:
    meta = run_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "config_used.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))


def test_load_pwms_from_config_preserves_log_odds_matrix(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "sample"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_config_used(
        run_dir,
        {
            "cruncher": {
                "pwms_info": {
                    "lexA": {
                        "pwm_matrix": [
                            [0.70, 0.10, 0.10, 0.10],
                            [0.10, 0.70, 0.10, 0.10],
                        ],
                        "log_odds_matrix": [
                            [6.0, -2.0, -2.0, -2.0],
                            [-2.0, 6.0, -2.0, -2.0],
                        ],
                    }
                }
            }
        },
    )

    pwms, _ = load_pwms_from_config(run_dir)
    assert "lexA" in pwms
    assert pwms["lexA"].log_odds_matrix is not None
    assert np.allclose(
        np.asarray(pwms["lexA"].log_odds_matrix, dtype=float),
        np.asarray(
            [
                [6.0, -2.0, -2.0, -2.0],
                [-2.0, 6.0, -2.0, -2.0],
            ],
            dtype=float,
        ),
    )


def test_load_pwms_from_config_applies_window_to_log_odds_matrix(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "sample"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_config_used(
        run_dir,
        {
            "cruncher": {
                "sample": {
                    "sequence_length": 10,
                    "motif_width": {
                        "maxw": 2,
                        "strategy": "max_info",
                    },
                },
                "pwms_info": {
                    "tfA": {
                        "pwm_matrix": [
                            [0.25, 0.25, 0.25, 0.25],
                            [1.00, 0.00, 0.00, 0.00],
                            [1.00, 0.00, 0.00, 0.00],
                            [0.25, 0.25, 0.25, 0.25],
                        ],
                        "log_odds_matrix": [
                            [0.0, 0.0, 0.0, 0.0],
                            [9.0, -3.0, -3.0, -3.0],
                            [8.0, -2.0, -2.0, -2.0],
                            [1.0, 1.0, 1.0, 1.0],
                        ],
                    }
                },
            }
        },
    )

    pwms, _ = load_pwms_from_config(run_dir)
    pwm = pwms["tfA"]
    assert pwm.length == 2
    assert np.allclose(
        pwm.matrix,
        np.asarray(
            [
                [1.00, 0.00, 0.00, 0.00],
                [1.00, 0.00, 0.00, 0.00],
            ],
            dtype=float,
        ),
    )
    assert pwm.log_odds_matrix is not None
    assert np.allclose(
        np.asarray(pwm.log_odds_matrix, dtype=float),
        np.asarray(
            [
                [9.0, -3.0, -3.0, -3.0],
                [8.0, -2.0, -2.0, -2.0],
            ],
            dtype=float,
        ),
    )


def test_save_config_persists_log_odds_matrix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _FakeStore:
        def get_pwm(self, _ref) -> PWM:
            return PWM(
                name="lexA",
                matrix=np.asarray(
                    [
                        [0.70, 0.10, 0.10, 0.10],
                        [0.10, 0.70, 0.10, 0.10],
                    ],
                    dtype=float,
                ),
                log_odds_matrix=np.asarray(
                    [
                        [6.1234567, -2.2345678, -2.0, -2.0],
                        [-2.0, 6.7654321, -2.0, -2.0],
                    ],
                    dtype=float,
                ),
            )

    cfg = CruncherConfig(
        schema_version=3,
        workspace=WorkspaceConfig(out_dir=Path("outputs"), regulator_sets=[["lexA"]]),
        catalog=CatalogConfig(root=Path(".cruncher"), pwm_source="matrix"),
        sample=None,
        analysis=None,
    )

    monkeypatch.setattr(
        "dnadesign.cruncher.app.sample.artifacts._store",
        lambda _cfg, _config_path: _FakeStore(),
    )
    monkeypatch.setattr(
        "dnadesign.cruncher.app.sample.artifacts._lockmap_for",
        lambda _cfg, _config_path: {"lexA": SimpleNamespace(source="demo", motif_id="lexA")},
    )

    run_dir = tmp_path / "outputs" / "sample"
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_config(
        cfg=cfg,
        batch_dir=run_dir,
        config_path=tmp_path / "config.yaml",
        tfs=["lexA"],
        set_index=0,
        sample_cfg=None,
        log_fn=lambda *_args, **_kwargs: None,
    )
    payload = yaml.safe_load((run_dir / "meta" / "config_used.yaml").read_text())["cruncher"]
    tf_info = payload["pwms_info"]["lexA"]
    assert "log_odds_matrix" in tf_info
    assert np.allclose(
        np.asarray(tf_info["log_odds_matrix"], dtype=float),
        np.asarray(
            [
                [6.123457, -2.234568, -2.0, -2.0],
                [-2.0, 6.765432, -2.0, -2.0],
            ],
            dtype=float,
        ),
    )
