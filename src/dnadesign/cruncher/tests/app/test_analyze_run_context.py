"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_analyze_run_context.py

Validates run-context resolution used by analyze plotting/publication orchestration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from dnadesign.cruncher.app.analyze.computation import AnalysisTablesAndMetrics
from dnadesign.cruncher.app.analyze.execution import AnalysisRunExecutionContext
from dnadesign.cruncher.app.analyze.run_context import resolve_analysis_run_context


class _AnalysisCfg:
    plot_format = "png"
    plot_dpi = 144

    def model_dump(self, *, mode: str) -> dict[str, object]:
        return {"mode": mode, "plot_format": self.plot_format, "plot_dpi": self.plot_dpi}


def _computed_bundle() -> AnalysisTablesAndMetrics:
    frame = pd.DataFrame()
    return AnalysisTablesAndMetrics(
        objective_from_manifest={"combine": "min"},
        bidirectional=True,
        pwm_pseudocounts=0.1,
        log_odds_clip=None,
        retain_sequences=set(),
        trajectory_df=frame,
        trajectory_lines_df=frame,
        baseline_plot_df=frame,
        elites_plot_df=frame,
        nn_df=frame,
        baseline_nn=frame,
        diagnostics_payload={"status": "ok"},
        objective_components={"weights": []},
        overlap_summary={"overlap_total_bp_median": 0.0},
        table_paths={"scores_summary": Path("/tmp/table.parquet")},
    )


def _execution_context(*, tmp_path: Path, tf_names: list[str]) -> AnalysisRunExecutionContext:
    tmp_root = tmp_path / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    return AnalysisRunExecutionContext(
        run_name="run-1",
        run_dir=tmp_path / "run",
        manifest={"objective": {"combine": "min"}},
        optimizer_stats=None,
        pwms={"lexA": object()},
        used_cfg={"sample": {}},
        tf_names=tf_names,
        sample_meta=SimpleNamespace(mode="opt", top_k=1),
        analysis_id="aid",
        created_at="2026-02-28T00:00:00+00:00",
        analysis_root_path=tmp_path / "run" / "analysis",
        tmp_root=tmp_root,
        analysis_used_file=tmp_root / "reports" / "analysis_used.yaml",
        require_random_baseline=False,
        sequences_df=pd.DataFrame({"sequence": ["AAAA"]}),
        elites_df=pd.DataFrame({"id": ["E1"]}),
        hits_df=pd.DataFrame({"elite_id": ["E1"]}),
        baseline_df=pd.DataFrame({"sequence": ["CCCC"]}),
        baseline_hits_df=pd.DataFrame({"sequence": ["CCCC"]}),
        trace_idata=None,
        elites_meta={"status": "ok"},
    )


def test_resolve_analysis_run_context_rejects_empty_tf_names(tmp_path: Path) -> None:
    execution = _execution_context(tmp_path=tmp_path, tf_names=[])

    with pytest.raises(ValueError, match="at least one TF name"):
        resolve_analysis_run_context(
            analysis_cfg=_AnalysisCfg(),
            execution=execution,
        )


def test_resolve_analysis_run_context_builds_context_with_delegated_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _execution_context(tmp_path=tmp_path, tf_names=["lexA"])
    computed = _computed_bundle()
    score_space_ctx = SimpleNamespace(
        focus_pair=None,
        trajectory_tf_pair=("lexA", "lexA"),
        mode="pair",
        pairs=[("lexA", "lexA")],
        trajectory_scale="llr",
    )

    monkeypatch.setattr(
        "dnadesign.cruncher.app.analyze.run_context.compute_analysis_tables_and_metrics",
        lambda **_kwargs: computed,
    )
    monkeypatch.setattr(
        "dnadesign.cruncher.app.analyze.run_context._resolve_score_space_context",
        lambda **_kwargs: score_space_ctx,
    )

    context = resolve_analysis_run_context(
        analysis_cfg=_AnalysisCfg(),
        execution=execution,
    )

    assert context.execution is execution
    assert context.computed is computed
    assert context.score_space_ctx is score_space_ctx
    assert context.plot_format == "png"
    assert context.plot_kwargs == {"dpi": 144, "png_compress_level": 9}
    assert context.analysis_cfg_payload == {"mode": "json", "plot_format": "png", "plot_dpi": 144}
