"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/review_bundle.py

Materialize the manifest-backed metastudy review bundle.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..core.contracts import MetastudyPaths, RecommendationThresholds, SfxiEvidenceFrame
from ..core.response_contracts import ResponseMetricScreen
from ..reporting.notebook import write_review_notebook
from ..reporting.plots import write_visuals
from ..reporting.report import write_report
from .publication import artifact_inventory
from .response_screen_publication import write_response_screen_tables


@dataclass(frozen=True)
class ReviewBundleEvidence:
    """Typed inputs required to publish one complete review bundle."""

    summary: pd.DataFrame
    pairwise: pd.DataFrame
    candidates: pd.DataFrame
    comparison_panel: pd.DataFrame
    overlap_by_k: pd.DataFrame
    denominator_sensitivity: pd.DataFrame
    pressure_tests: pd.DataFrame
    model_validation: pd.DataFrame
    setpoint_support: pd.DataFrame
    response_screen: ResponseMetricScreen
    response_examples: pd.DataFrame
    rmf_cardinality_pressure: pd.DataFrame
    scored: dict[str, dict[str, pd.DataFrame]]
    sfxi_evidence: tuple[SfxiEvidenceFrame, ...]
    thresholds: RecommendationThresholds
    recommendation: dict[str, object]
    canonical_sfxi_validation: dict[str, object]
    primary_reduction_id: str


def materialize_review_bundle(
    paths: MetastudyPaths,
    evidence: ReviewBundleEvidence,
) -> dict[str, dict[str, object]]:
    """Write every registered review artifact and return its inventory."""

    tables_dir = paths.out_dir / "tables"
    tables_dir.mkdir()
    table_artifacts = _write_core_tables(tables_dir, evidence)
    table_artifacts.update(write_response_screen_tables(evidence.response_screen, tables_dir))

    visual_paths, plot_manifest = write_visuals(
        paths.out_dir,
        summary=evidence.summary,
        pairwise=evidence.pairwise,
        candidates=evidence.candidates,
        overlap_by_k=evidence.overlap_by_k,
        denominator_sensitivity=evidence.denominator_sensitivity,
        comparison_panel=evidence.comparison_panel,
        model_validation=evidence.model_validation,
        setpoint_support=evidence.setpoint_support,
        response_screen=evidence.response_screen,
        response_examples=evidence.response_examples,
        rmf_cardinality_pressure=evidence.rmf_cardinality_pressure,
        scored=evidence.scored,
        sfxi_evidence=evidence.sfxi_evidence,
        thresholds=evidence.thresholds,
        comparison_policy_id=str(evidence.recommendation["comparison_policy_id"]),
        model_support_passed=bool(evidence.recommendation["model_support_passed"]),
        primary_reduction_id=evidence.primary_reduction_id,
    )
    plot_manifest_path = tables_dir / "plot_manifest.csv"
    plot_manifest.to_csv(plot_manifest_path, index=False)
    report_path = write_report(
        paths.out_dir,
        summary=evidence.summary,
        pairwise=evidence.pairwise,
        candidates=evidence.candidates,
        overlap_by_k=evidence.overlap_by_k,
        denominator_sensitivity=evidence.denominator_sensitivity,
        comparison_panel=evidence.comparison_panel,
        model_validation=evidence.model_validation,
        setpoint_support=evidence.setpoint_support,
        response_screen=evidence.response_screen,
        pressure_tests=evidence.pressure_tests,
        plot_manifest=plot_manifest,
        recommendation=evidence.recommendation,
        canonical_sfxi_validation=evidence.canonical_sfxi_validation,
        thresholds=evidence.thresholds,
        primary_reduction_id=evidence.primary_reduction_id,
    )
    review_notebook_path = write_review_notebook(paths.out_dir)
    artifacts = {
        **table_artifacts,
        "table__plot_manifest": plot_manifest_path,
        "report": report_path,
        "review_notebook": review_notebook_path,
        **{f"plot__{plot_id}": path for plot_id, path in visual_paths.items()},
    }
    return artifact_inventory(paths.out_dir, artifacts)


def _write_core_tables(tables_dir: Path, evidence: ReviewBundleEvidence) -> dict[str, Path]:
    frames = {
        "policy_summary": evidence.summary,
        "score_correlations": evidence.pairwise,
        "top_candidates": evidence.candidates,
        "policy_comparison_panel": evidence.comparison_panel,
        "overlap_by_k": evidence.overlap_by_k,
        "denominator_sensitivity": evidence.denominator_sensitivity,
        "pressure_tests": evidence.pressure_tests,
        "model_validation": evidence.model_validation,
        "setpoint_support": evidence.setpoint_support,
        "measured_response_examples": evidence.response_examples,
        "rmf_cardinality_pressure": evidence.rmf_cardinality_pressure,
    }
    paths: dict[str, Path] = {}
    for table_id, frame in frames.items():
        if frame.empty:
            raise ValueError(f"Review bundle table {table_id!r} must not be empty.")
        path = tables_dir / f"{table_id}.csv"
        frame.to_csv(path, index=False)
        paths[f"table__{table_id}"] = path
    return paths


__all__ = ["ReviewBundleEvidence", "materialize_review_bundle"]
