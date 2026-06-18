"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/review/plots/materialization.py

Materialize Stage B realized-label review plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ....stage_a.manifests import file_sha256
from ...notebook_visuals.specs import REALIZED_REVIEW_VISUAL_SPECS, StageBNotebookVisualSpec
from .contracts import (
    REALIZED_REVIEW_INTERPRETATION_BOUNDARY,
    REALIZED_REVIEW_PLOT_MANIFEST_FILENAME,
    REALIZED_REVIEW_PLOT_MANIFEST_SCHEMA_VERSION,
    REALIZED_REVIEW_STYLE_CONTRACT,
)
from .display_text import REALIZED_REVIEW_TEXT_CONTRACT, plot_manifest_alt_text, plot_manifest_title
from .renderers import realized_review_renderer


def materialize_tfbs_stage_b_realized_review_plots(
    *,
    trajectory_csv_path: str | Path,
    pair_summary_csv_path: str | Path,
    out_dir: str | Path,
) -> Path:
    """Write compact true-label plots for Stage B peer-review inspection."""

    trajectory_path = Path(trajectory_csv_path)
    pair_path = Path(pair_summary_csv_path)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = _read_csv(trajectory_path, label="trajectory")
    pair_summary = _read_csv(pair_path, label="pair summary")

    label_names = _shared_label_names(trajectory, pair_summary)
    plots: list[dict[str, Any]] = []
    for label_name in label_names:
        for spec in REALIZED_REVIEW_VISUAL_SPECS.values():
            renderer = realized_review_renderer(spec)
            plots.append(
                _materialize_plot(
                    path=output_dir / spec.plot_filename(label_name=label_name),
                    spec=spec,
                    label_name=label_name,
                    control_role=_label_control_role(trajectory, label_name=label_name),
                    interval=_plot_interval_contract(
                        spec=spec,
                        label_name=label_name,
                        trajectory=trajectory,
                        pair_summary=pair_summary,
                    ),
                    draw=(
                        lambda path, renderer=renderer, label_name=label_name: renderer(
                            trajectory,
                            pair_summary,
                            path,
                            label_name,
                        )
                    ),
                )
            )
    manifest = {
        "schema_version": REALIZED_REVIEW_PLOT_MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source_trajectory_csv_path": str(trajectory_path),
        "source_trajectory_csv_hash": file_sha256(trajectory_path),
        "source_pair_summary_csv_path": str(pair_path),
        "source_pair_summary_csv_hash": file_sha256(pair_path),
        "plot_count": len(plots),
        "plots": plots,
        "style_contract": REALIZED_REVIEW_STYLE_CONTRACT,
        "text_contract": REALIZED_REVIEW_TEXT_CONTRACT,
        "interpretation_boundary": REALIZED_REVIEW_INTERPRETATION_BOUNDARY,
    }
    manifest_path = output_dir / REALIZED_REVIEW_PLOT_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _read_csv(path: Path, *, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Stage B realized review {label} CSV not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Stage B realized review {label} CSV is empty: {path}")
    return frame


def _shared_label_names(trajectory: pd.DataFrame, pair_summary: pd.DataFrame) -> list[str]:
    required = {"label_name"}
    for label, frame in {"trajectory": trajectory, "pair summary": pair_summary}.items():
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Stage B realized review {label} missing column(s): {missing}")
    trajectory_labels = set(trajectory["label_name"].astype(str))
    pair_labels = set(pair_summary["label_name"].astype(str))
    labels = sorted(trajectory_labels & pair_labels)
    if not labels:
        raise ValueError("Stage B realized review plots require at least one shared label_name")
    missing_from_pairs = sorted(trajectory_labels - pair_labels)
    missing_from_trajectory = sorted(pair_labels - trajectory_labels)
    if missing_from_pairs or missing_from_trajectory:
        raise ValueError(
            "Stage B realized review plot label mismatch "
            f"(missing_from_pairs={missing_from_pairs}, missing_from_trajectory={missing_from_trajectory})"
        )
    return labels


def _materialize_plot(
    *,
    path: Path,
    spec: StageBNotebookVisualSpec,
    label_name: str,
    control_role: str,
    interval: dict[str, Any],
    draw: Any,
) -> dict[str, Any]:
    draw(path)
    replicate_count = int(interval["replicate_count"])
    return {
        "kind": spec.kind,
        "title": plot_manifest_title(
            spec.kind,
            label_name=label_name,
            replicate_count=replicate_count,
            control_role=control_role,
        ),
        "label_name": label_name,
        "path": str(path),
        "sha256": file_sha256(path),
        "alt_text": plot_manifest_alt_text(
            spec.kind,
            label_name=label_name,
            replicate_count=replicate_count,
            control_role=control_role,
        ),
        "control_role": control_role,
        "interval_kind": str(interval["kind"]),
        "interval": interval,
    }


def _plot_interval_contract(
    *,
    spec: StageBNotebookVisualSpec,
    label_name: str,
    trajectory: pd.DataFrame,
    pair_summary: pd.DataFrame,
) -> dict[str, Any]:
    if spec.kind == "realized_label_lift_trajectory":
        sub = trajectory.loc[trajectory["label_name"].astype(str) == label_name]
        replicate_count = int(sub.groupby(["oracle_role", "round"], dropna=False).size().max()) if not sub.empty else 0
        return _sample_sd_interval_contract(
            replicate_count=replicate_count,
            unit="seed replicate",
            applies_to="selected label lift ratio by label source and round",
        )
    if spec.kind == "positive_null_lift_summary":
        sub = pair_summary.loc[pair_summary["label_name"].astype(str) == label_name]
        return _sample_sd_interval_contract(
            replicate_count=int(len(sub)),
            unit="sequence-matched/control seed pair",
            applies_to="sequence-matched-minus-control lift summary",
        )
    return {
        "kind": "none",
        "unit": "",
        "is_confidence_interval": False,
        "replicate_count": 0,
        "status": "not_applicable",
        "applies_to": spec.kind,
    }


def _sample_sd_interval_contract(*, replicate_count: int, unit: str, applies_to: str) -> dict[str, Any]:
    if replicate_count > 1:
        return {
            "kind": "sample_sd",
            "unit": unit,
            "is_confidence_interval": False,
            "replicate_count": int(replicate_count),
            "status": "available",
            "estimator": "mean_plus_minus_sample_standard_deviation",
            "applies_to": applies_to,
        }
    return {
        "kind": "none",
        "unit": unit,
        "is_confidence_interval": False,
        "replicate_count": int(replicate_count),
        "status": "not_available_single_seed",
        "applies_to": applies_to,
    }


def _label_control_role(trajectory: pd.DataFrame, *, label_name: str) -> str:
    if "null_control_role" not in trajectory.columns:
        return ""
    sub = trajectory.loc[
        (trajectory["label_name"].astype(str) == str(label_name))
        & (trajectory["oracle_role"].astype(str) == "matched_null"),
        "null_control_role",
    ]
    clean = sorted({str(value) for value in sub.tolist() if str(value) not in {"", "nan", "None"}})
    return clean[0] if len(clean) == 1 else ""
