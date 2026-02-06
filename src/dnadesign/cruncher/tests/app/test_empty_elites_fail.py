"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_empty_elites_fail.py

Ensures empty elite pools fail the run and mark status as failed.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dnadesign.cruncher.app.run_service import load_run_index
from dnadesign.cruncher.app.sample.run_set import RunError, _run_sample_for_set
from dnadesign.cruncher.artifacts.layout import status_path
from dnadesign.cruncher.config.schema_v3 import (
    CatalogConfig,
    CruncherConfig,
    SampleBudgetConfig,
    SampleConfig,
    SampleEliteFilterConfig,
    SampleElitesConfig,
    SampleObjectiveConfig,
    WorkspaceConfig,
)
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.utils.paths import resolve_lock_path


class _DummyOptimizer:
    def __init__(self, **_kwargs) -> None:
        self.all_meta = [(0, 0)]
        self.all_samples = [np.array([0, 1, 2, 3, 0, 1], dtype=np.int8)]
        self.all_scores = [{"lexA": 0.0}]
        self.best_score = 0.0
        self.best_meta = (0, 0)
        self.trace_idata = None

    def optimise(self) -> None:
        return

    def stats(self) -> dict[str, object]:
        return {}

    def objective_schedule_summary(self) -> dict[str, object]:
        return {}


def _cfg(
    tmp_path: Path,
    *,
    elite_k: int = 1,
    min_per_tf_norm: float | None = 1.0,
) -> tuple[CruncherConfig, Path, SampleConfig]:
    sample_cfg = SampleConfig(
        seed=7,
        sequence_length=6,
        budget=SampleBudgetConfig(tune=0, draws=1),
        objective=SampleObjectiveConfig(score_scale="normalized-llr"),
        elites=SampleElitesConfig(
            k=elite_k,
            filter=SampleEliteFilterConfig(min_per_tf_norm=min_per_tf_norm, require_all_tfs=True),
        ),
    )
    cfg = CruncherConfig(
        schema_version=3,
        workspace=WorkspaceConfig(out_dir=Path("results"), regulator_sets=[["lexA"]]),
        catalog=CatalogConfig(root=tmp_path / ".cruncher", pwm_source="matrix"),
        sample=sample_cfg,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("cruncher:\n  schema_version: 3\n")
    return cfg, config_path, sample_cfg


def test_empty_elites_fail_run_and_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg, config_path, sample_cfg = _cfg(tmp_path)
    pwm = PWM(name="lexA", matrix=np.full((4, 4), 0.25, dtype=float))
    lockmap = {"lexA": SimpleNamespace(source="demo", motif_id="M1", sha256="sha")}
    lock_path = resolve_lock_path(config_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("{}")

    monkeypatch.setattr(
        "dnadesign.cruncher.app.sample.run_set._load_pwms_for_set",
        lambda **_kwargs: {"lexA": pwm},
    )
    monkeypatch.setattr(
        "dnadesign.cruncher.app.sample.run_set._save_config",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "dnadesign.cruncher.core.optimizers.registry.get_optimizer",
        lambda _name: _DummyOptimizer,
    )

    with pytest.raises(RunError, match="relax .*min_per_tf_norm.*increase .*sequence_length.*increase .*draws"):
        _run_sample_for_set(
            cfg,
            config_path,
            set_index=1,
            set_count=1,
            include_set_index=False,
            tfs=["lexA"],
            lockmap=lockmap,
            sample_cfg=sample_cfg,
        )

    run_index = load_run_index(config_path)
    assert run_index
    payload = next(iter(run_index.values()))
    assert payload.get("status") == "failed"
    run_dir = Path(str(payload.get("run_dir")))
    status_payload = json.loads(status_path(run_dir).read_text())
    assert status_payload.get("status") == "failed"


def test_insufficient_elites_fail_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg, config_path, sample_cfg = _cfg(tmp_path, elite_k=2, min_per_tf_norm=0.0)
    pwm = PWM(name="lexA", matrix=np.full((4, 4), 0.25, dtype=float))
    lockmap = {"lexA": SimpleNamespace(source="demo", motif_id="M1", sha256="sha")}
    lock_path = resolve_lock_path(config_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("{}")

    monkeypatch.setattr(
        "dnadesign.cruncher.app.sample.run_set._load_pwms_for_set",
        lambda **_kwargs: {"lexA": pwm},
    )
    monkeypatch.setattr(
        "dnadesign.cruncher.app.sample.run_set._save_config",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "dnadesign.cruncher.core.optimizers.registry.get_optimizer",
        lambda _name: _DummyOptimizer,
    )

    with pytest.raises(RunError, match="fewer than cruncher.sample.elites.k=2"):
        _run_sample_for_set(
            cfg,
            config_path,
            set_index=1,
            set_count=1,
            include_set_index=False,
            tfs=["lexA"],
            lockmap=lockmap,
            sample_cfg=sample_cfg,
        )
