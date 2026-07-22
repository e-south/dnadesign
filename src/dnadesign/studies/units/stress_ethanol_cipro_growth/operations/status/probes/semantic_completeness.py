"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/operations/status/probes/semantic_completeness.py

USR metadata completeness probes for stress_ethanol_cipro_growth.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.usr import Dataset

from ..record_normalizer import StressEthanolCiproGrowthResolvedContext


def inspect_stress_ethanol_cipro_growth_semantic_completeness(
    *,
    study_context: StressEthanolCiproGrowthResolvedContext,
) -> dict[str, object] | None:
    root = study_context.canonical_usr_root_path
    expected_rows = study_context.densegen_rows
    if root is None or expected_rows is None:
        return None
    if study_context.densegen_dataset_id is None:
        return None

    try:
        source_overlay = _overlay_guardrail_state(
            root=root,
            dataset_id=study_context.densegen_dataset_id,
            namespace="densegen",
        )
        dataset_checks = []
        if study_context.merged_anchor_dataset_id is not None:
            dataset_checks.append(
                _densegen_metadata_projection_state(
                    root=root,
                    dataset_id=study_context.merged_anchor_dataset_id,
                    expected_rows=int(expected_rows),
                    label="anchor",
                )
            )
        if study_context.construct_context_dataset_id is not None:
            dataset_checks.append(
                _densegen_metadata_projection_state(
                    root=root,
                    dataset_id=study_context.construct_context_dataset_id,
                    expected_rows=int(expected_rows),
                    label="construct",
                )
            )
    except Exception as exc:
        return {
            "state": "attention",
            "drives_top_level_attention": True,
            "summary": f"semantic completeness probe failed: {exc}",
            "probe_error": str(exc),
        }

    attention = bool(source_overlay.get("state") == "attention") or any(
        str(check.get("state") or "") == "attention" for check in dataset_checks
    )
    summary_parts = [str(source_overlay["summary"]), *[str(check["summary"]) for check in dataset_checks]]
    return {
        "state": "attention" if attention else "ok",
        "drives_top_level_attention": attention,
        "source_overlay_state": source_overlay,
        "dataset_checks": dataset_checks,
        "summary": "; ".join(summary_parts),
    }


def _overlay_guardrail_state(*, root: Path, dataset_id: str, namespace: str) -> dict[str, object]:
    dataset = Dataset(root, dataset_id)
    overlay = next((item for item in dataset.list_overlays() if item.namespace == namespace), None)
    if overlay is None:
        return {
            "state": "attention",
            "dataset": dataset_id,
            "namespace": namespace,
            "overlay_present": False,
            "overlay_compact": False,
            "summary": f"source overlay guardrail missing {dataset_id}:{namespace}",
        }
    overlay_path = Path(overlay.path)
    overlay_compact = overlay_path.is_file()
    return {
        "state": "ok" if overlay_compact else "attention",
        "dataset": dataset_id,
        "namespace": namespace,
        "overlay_present": True,
        "overlay_compact": overlay_compact,
        "overlay_path": str(overlay_path),
        "summary": (
            f"source overlay compact {dataset_id}:{namespace}"
            if overlay_compact
            else f"source overlay needs compaction {dataset_id}:{namespace}"
        ),
    }


def _densegen_metadata_projection_state(
    *,
    root: Path,
    dataset_id: str,
    expected_rows: int,
    label: str,
) -> dict[str, object]:
    required_columns = ("densegen__plan", "densegen__required_regulators")
    dataset = Dataset(root, dataset_id)
    schema = dataset.schema()
    missing_columns = [column for column in required_columns if column not in schema.names]
    if missing_columns:
        return {
            "state": "attention",
            "dataset": dataset_id,
            "label": label,
            "required_columns": list(required_columns),
            "missing_columns": missing_columns,
            "non_null_counts": {},
            "expected_rows": expected_rows,
            "summary": f"{label} DenseGen metadata columns missing {dataset_id}: {', '.join(missing_columns)}",
        }

    counts = {column: 0 for column in required_columns}
    for batch in dataset.scan(columns=list(required_columns), include_overlays=True, batch_size=65_536):
        for column in required_columns:
            array = batch.column(batch.schema.get_field_index(column))
            counts[column] += int(batch.num_rows - array.null_count)

    complete = all(count >= expected_rows for count in counts.values())
    min_count = min(counts.values()) if counts else 0
    return {
        "state": "ok" if complete else "attention",
        "dataset": dataset_id,
        "label": label,
        "required_columns": list(required_columns),
        "missing_columns": [],
        "non_null_counts": counts,
        "expected_rows": expected_rows,
        "missing_densegen_rows": max(expected_rows - min_count, 0),
        "summary": (
            f"{label} DenseGen metadata ready {dataset_id} {min_count}/{expected_rows}"
            if complete
            else f"{label} DenseGen metadata incomplete {dataset_id} {min_count}/{expected_rows}"
        ),
    }


__all__ = ["inspect_stress_ethanol_cipro_growth_semantic_completeness"]
