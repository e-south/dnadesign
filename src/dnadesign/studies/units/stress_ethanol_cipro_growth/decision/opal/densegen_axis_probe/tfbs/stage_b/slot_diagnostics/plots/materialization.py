"""Materialize Stage B slot-diagnostic plots."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ....stage_a.manifests import file_sha256
from ...notebook_visuals.specs import SLOT_DIAGNOSTIC_VISUAL_SPECS, StageBNotebookVisualSpec
from .contracts import (
    SLOT_DIAGNOSTIC_INTERPRETATION_BOUNDARY,
    SLOT_DIAGNOSTIC_PLOT_MANIFEST_FILENAME,
    SLOT_DIAGNOSTIC_PLOT_MANIFEST_SCHEMA_VERSION,
    SLOT_DIAGNOSTIC_STYLE_CONTRACT,
)
from .renderers import slot_diagnostic_renderer


def materialize_tfbs_stage_b_slot_diagnostic_plots(
    *,
    trajectory_csv_path: str | Path,
    pair_summary_csv_path: str | Path,
    count_distribution_csv_path: str | Path,
    out_dir: str | Path,
) -> Path:
    """Write reviewer-facing plots for slot-label count-confound diagnostics."""

    trajectory_path = Path(trajectory_csv_path)
    pair_path = Path(pair_summary_csv_path)
    distribution_path = Path(count_distribution_csv_path)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = _read_csv(trajectory_path, label="slot trajectory")
    pair_summary = _read_csv(pair_path, label="slot pair summary")
    count_distribution = _read_csv(distribution_path, label="slot count distribution")

    plots: list[dict[str, Any]] = []
    for spec in SLOT_DIAGNOSTIC_VISUAL_SPECS.values():
        renderer = slot_diagnostic_renderer(spec)
        plots.append(
            _materialize_plot(
                path=output_dir / spec.plot_filename(),
                spec=spec,
                draw=(
                    lambda path, renderer=renderer: renderer(
                        trajectory,
                        pair_summary,
                        count_distribution,
                        path,
                    )
                ),
            )
        )
    manifest = {
        "schema_version": SLOT_DIAGNOSTIC_PLOT_MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source_trajectory_csv_path": str(trajectory_path),
        "source_trajectory_csv_hash": file_sha256(trajectory_path),
        "source_pair_summary_csv_path": str(pair_path),
        "source_pair_summary_csv_hash": file_sha256(pair_path),
        "source_count_distribution_csv_path": str(distribution_path),
        "source_count_distribution_csv_hash": file_sha256(distribution_path),
        "plot_count": len(plots),
        "plots": plots,
        "style_contract": SLOT_DIAGNOSTIC_STYLE_CONTRACT,
        "interpretation_boundary": SLOT_DIAGNOSTIC_INTERPRETATION_BOUNDARY,
    }
    manifest_path = output_dir / SLOT_DIAGNOSTIC_PLOT_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _read_csv(path: Path, *, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Stage B {label} CSV not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Stage B {label} CSV is empty: {path}")
    return frame


def _materialize_plot(*, path: Path, spec: StageBNotebookVisualSpec, draw: Any) -> dict[str, Any]:
    draw(path)
    return {
        "kind": spec.kind,
        "title": spec.plot_title(),
        "path": str(path),
        "sha256": file_sha256(path),
        "alt_text": spec.alt_text,
    }
