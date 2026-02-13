"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_empty_elites_fail.py

Ensures empty elite pools fail the run and mark status as failed.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dnadesign.cruncher.app.sample.run_set import RunError, _run_sample_for_set
from dnadesign.cruncher.config.schema_v3 import (
    CatalogConfig,
    CruncherConfig,
    SampleBudgetConfig,
    SampleConfig,
    SampleElitesConfig,
    SampleObjectiveConfig,
    WorkspaceConfig,
)
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.utils.paths import resolve_lock_path


class _DummyOptimizer:
    def __init__(self, **_kwargs) -> None:
        self.all_meta = [(0, 0)]
        self.all_trace_meta = [{"slot_id": 0, "particle_id": 0, "beta": 1.0, "sweep_idx": 0, "phase": "draw"}]
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
) -> tuple[CruncherConfig, Path, SampleConfig]:
    sample_cfg = SampleConfig(
        seed=7,
        sequence_length=6,
        budget=SampleBudgetConfig(tune=0, draws=1),
        objective=SampleObjectiveConfig(score_scale="normalized-llr"),
        elites=SampleElitesConfig(
            k=elite_k,
            select={"diversity": 0.0, "pool_size": "auto"},
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


def test_insufficient_elites_fail_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg, config_path, sample_cfg = _cfg(tmp_path, elite_k=2)
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


def test_non_gibbs_optimizer_kind_is_rejected_early(tmp_path: Path) -> None:
    cfg, config_path, sample_cfg = _cfg(tmp_path, elite_k=1)
    object.__setattr__(sample_cfg.optimizer, "kind", "pt")

    with pytest.raises(ValueError, match="sample.optimizer.kind.*gibbs_anneal"):
        _run_sample_for_set(
            cfg,
            config_path,
            set_index=1,
            set_count=1,
            include_set_index=False,
            tfs=["lexA"],
            lockmap={},
            sample_cfg=sample_cfg,
        )


def test_optimizer_early_stop_is_passed_to_optimizer_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg, config_path, sample_cfg = _cfg(tmp_path, elite_k=1)
    sample_cfg.optimizer.early_stop.enabled = True
    sample_cfg.optimizer.early_stop.patience = 5
    sample_cfg.optimizer.early_stop.min_delta = 0.02
    sample_cfg.optimizer.early_stop.require_min_unique = True
    sample_cfg.optimizer.early_stop.min_unique = 2
    sample_cfg.optimizer.early_stop.success_min_per_tf_norm = 0.3

    pwm = PWM(name="lexA", matrix=np.full((4, 4), 0.25, dtype=float))
    lockmap = {"lexA": SimpleNamespace(source="demo", motif_id="M1", sha256="sha")}
    lock_path = resolve_lock_path(config_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("{}")

    captured: dict[str, object] = {}

    class _CaptureOptimizer(_DummyOptimizer):
        def __init__(self, **kwargs) -> None:
            captured["cfg"] = kwargs["cfg"]
            super().__init__(**kwargs)

        def optimise(self) -> None:
            raise KeyboardInterrupt("stop")

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
        lambda _name: _CaptureOptimizer,
    )

    with pytest.raises(KeyboardInterrupt, match="stop"):
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

    optimizer_cfg = captured.get("cfg")
    assert isinstance(optimizer_cfg, dict)
    early_stop = optimizer_cfg.get("early_stop")
    assert isinstance(early_stop, dict)
    assert early_stop["enabled"] is True
    assert early_stop["patience"] == 5
    assert early_stop["min_delta"] == pytest.approx(0.02)
    assert early_stop["require_min_unique"] is True
    assert early_stop["min_unique"] == 2
    assert early_stop["success_min_per_tf_norm"] == pytest.approx(0.3)
