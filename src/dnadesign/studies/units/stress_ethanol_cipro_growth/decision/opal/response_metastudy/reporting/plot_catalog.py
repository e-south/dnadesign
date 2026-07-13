"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/plot_catalog.py

Validated facade for response metric metastudy plot deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .plot_contracts import PLOT_TIER_DIRS, PlotSpec
from .plot_definitions import PLOT_SPECS
from .plot_narrative import PLOT_DATA_TABLES, PLOT_NON_CLAIM_BOUNDARIES, PLOT_RATIONALES

__all__ = ["PLOT_SPECS", "PLOT_TIER_DIRS", "build_plot_manifest", "specs_by_id"]


def _validate_catalog_contract() -> None:
    plot_ids = [spec.plot_id for spec in PLOT_SPECS]
    expected = set(plot_ids)
    if len(plot_ids) != len(expected):
        raise RuntimeError("response metric metastudy plot ids must be unique.")
    primary_steps = [spec.review_step for spec in PLOT_SPECS if spec.tier == "primary_decision"]
    if primary_steps != list(range(1, len(primary_steps) + 1)):
        raise RuntimeError("primary decision plots must declare contiguous review steps in catalog order.")
    if any(spec.review_step is not None for spec in PLOT_SPECS if spec.tier != "primary_decision"):
        raise RuntimeError("only primary decision plots may declare a review step.")
    for field, mapping in (
        ("rationale", PLOT_RATIONALES),
        ("non_claim_boundary", PLOT_NON_CLAIM_BOUNDARIES),
        ("data_table", PLOT_DATA_TABLES),
    ):
        if set(mapping) != expected:
            missing = sorted(expected - set(mapping))
            extra = sorted(set(mapping) - expected)
            raise RuntimeError(f"plot {field} catalog mismatch: missing={missing}, extra={extra}")
        if any(not value.strip() for value in mapping.values()):
            raise RuntimeError(f"plot {field} values must be non-empty.")


_validate_catalog_contract()


def specs_by_id() -> dict[str, PlotSpec]:
    return {spec.plot_id: spec for spec in PLOT_SPECS}


def build_plot_manifest(paths: dict[str, Path], *, root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in PLOT_SPECS:
        path = paths[spec.plot_id]
        rows.append(
            {
                "plot_id": spec.plot_id,
                "filename": spec.filename,
                "tier": spec.tier,
                "visual_type": spec.visual_type,
                "review_step": spec.review_step,
                "title": spec.title,
                "premise": spec.premise,
                "decision_value": spec.decision_value,
                "rationale": spec.rationale,
                "alt_text": spec.alt_text,
                "non_claim_boundary": spec.non_claim_boundary,
                "data_table": spec.data_table,
                "path": path.resolve().relative_to(root.resolve()).as_posix(),
            }
        )
    return pd.DataFrame(rows)
