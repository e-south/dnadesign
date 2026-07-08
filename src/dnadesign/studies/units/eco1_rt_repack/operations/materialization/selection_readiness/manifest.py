"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/manifest.py

Selection-readiness manifest writer for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.constants import (
    CREATED_BY,
    PLOTS_DIR_NAME,
    SELECTION_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.handoff_readiness import (
    build_handoff_readiness,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.local_structure import (
    LOCAL_STRUCTURE_REGION_IDS,
    LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_ID,
    LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_NOTE,
    LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel import (
    panel_coverage_summary,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.selection_summary import (
    build_selection_summary,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    SELECTION_PLOT_METADATA,
)
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri

_OPTIONAL_REVIEW_SOURCE_KEYS = ("llr_300m", "llr_6b", "sae_window")


def write_selection_readiness_manifest(
    path: Path,
    *,
    paths: dict[str, Path],
    feasibility_path: Path,
    triage_path: Path,
    local_structure_path: Path,
    local_structure_threshold_sensitivity_path: Path,
    region_msa_support_path: Path,
    primary_panel_selection_trace_path: Path,
    panel_path: Path,
    handoff_sequence_csv_path: Path,
    candidate_handoff_path: Path,
    plot_rows: list[dict[str, object]],
    feasibility_rows: list[dict[str, object]],
    triage_rows: list[dict[str, object]],
    local_structure_rows: list[dict[str, object]],
    local_structure_threshold_sensitivity_rows: list[dict[str, object]],
    region_msa_support_rows: list[dict[str, object]],
    primary_panel_selection_trace_rows: list[dict[str, object]],
    local_structure_source_basis_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    panel_rows: list[dict[str, object]],
    handoff_sequence_rows: list[dict[str, object]],
    created_at: str,
) -> None:
    """Write the manifest that ties selection-readiness artifacts together."""

    path.parent.mkdir(parents=True, exist_ok=True)
    optional_review_sources = _optional_review_sources(manifest_root=path.parent, paths=paths)
    missing_optional_review_sources = [
        source_id for source_id, source in optional_review_sources.items() if not source["materialized"]
    ]
    payload = {
        "schema_id": "eco1_rt.selection_readiness_manifest",
        "schema_version": 1,
        "status": "materialized_degraded" if missing_optional_review_sources else "materialized",
        "created_by": CREATED_BY,
        "created_at": created_at,
        "selection_policy_id": SELECTION_POLICY_ID,
        "governing_rule": (
            "Select six primary conservative candidates globally after preservation gates and a chemistry/support "
            "gate. The preservation gate requires strong fold, protected/contact/thumb-track preservation, declared "
            "local RMSD gates, including the C-terminal primer-RNA recognition region. The chemistry/support gate "
            "requires zero acidic gains near retained DNA/RNA and zero unobserved proximal substitutions. The final "
            "selection order uses simple mutation-set dissimilarity, basic-loss and Pro/Gly penalties, regional MSA "
            "support, local RMSD values inside the gate, fold metrics, and a deterministic tie-break. Design classes "
            "remain mask-policy context, not quotas. Do not use ESMC or SAE as positive selection evidence."
        ),
        "primary_panel_policy": {
            "policy_id": SELECTION_POLICY_ID,
            "scope": "global_primary_candidate_pool",
            "interpretation": (
                "Primary panel candidates pass the preservation and chemistry/support gates. Rows that fail those "
                "gates remain context rows and are not conservative handoff choices."
            ),
        },
        "local_structure_rmsd_threshold_policy": {
            "policy_id": LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_ID,
            "policy_note": LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_NOTE,
            "coordinate_scope": "mapped_rt_chain_ca_after_global_fit",
            "thresholds_angstrom": dict(LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM),
        },
        "local_structure_source_basis": local_structure_source_basis_rows,
        "local_structure_regions": _local_structure_region_manifest_rows(local_structure_rows),
        "sae_window_policy": (
            "SAE windows are retained as review evidence but not used for selection because the current pool "
            "does not meaningfully stratify in SAE-window space."
        ),
        "esmc_policy": "ESMC additive LLR rows are retained for review and are not used as panel-selection tie-breaks.",
        "optional_review_sources": optional_review_sources,
        "missing_optional_review_sources": missing_optional_review_sources,
        "source_tables": {
            key: _manifest_relative_path(path.parent, value)
            for key, value in paths.items()
            if value.exists() and value.is_file()
        },
        "artifacts": {
            "feasibility_report": _manifest_relative_path(path.parent, feasibility_path),
            "candidate_triage_table": _manifest_relative_path(path.parent, triage_path),
            "local_structure_region_metrics": _manifest_relative_path(path.parent, local_structure_path),
            "local_structure_threshold_sensitivity": _manifest_relative_path(
                path.parent,
                local_structure_threshold_sensitivity_path,
            ),
            "region_msa_support": _manifest_relative_path(path.parent, region_msa_support_path),
            "primary_panel_selection_trace": _manifest_relative_path(path.parent, primary_panel_selection_trace_path),
            "candidate_selection_panel": _manifest_relative_path(path.parent, panel_path),
            "candidate_handoff_sequences": _manifest_relative_path(path.parent, handoff_sequence_csv_path),
            "plots_root": PLOTS_DIR_NAME,
        },
        "path_policy": "paths_relative_to_selection_manifest",
        "plots": [_plot_manifest_row(row, manifest_root=path.parent) for row in plot_rows],
        "artifact_hashes": {
            key: sha256_uri(value)
            for key, value in {
                **{key: value for key, value in paths.items() if value.exists() and value.is_file()},
                "feasibility_report": feasibility_path,
                "candidate_triage_table": triage_path,
                "local_structure_region_metrics": local_structure_path,
                "local_structure_threshold_sensitivity": local_structure_threshold_sensitivity_path,
                "region_msa_support": region_msa_support_path,
                "primary_panel_selection_trace": primary_panel_selection_trace_path,
                "candidate_selection_panel": panel_path,
                "candidate_handoff_sequences": handoff_sequence_csv_path,
                **{str(row["plot_id"]): Path(str(row["path"])) for row in plot_rows},
            }.items()
        },
        "row_counts": {
            "feasibility_report": len(feasibility_rows),
            "candidate_triage_table": len(triage_rows),
            "local_structure_region_metrics": len(local_structure_rows),
            "local_structure_threshold_sensitivity": len(local_structure_threshold_sensitivity_rows),
            "region_msa_support": len(region_msa_support_rows),
            "primary_panel_selection_trace": len(primary_panel_selection_trace_rows),
            "candidate_selection_panel": len(panel_rows),
            "candidate_handoff_sequences": len(handoff_sequence_rows),
        },
        "gate_counts": {
            "feasibility_status": _count_by(feasibility_rows, "feasibility_status"),
            "hard_gate_status": _count_by(triage_rows, "hard_gate_status"),
            "fold_review_class": _count_by(triage_rows, "fold_review_class"),
            "local_structure_gate_status": _count_by(triage_rows, "local_structure_gate_status"),
            "sae_window_status": _count_by(triage_rows, "sae_window_status"),
            "selection_candidate_tier": _count_by(triage_rows, "selection_candidate_tier"),
        },
        "selection_summary": build_selection_summary(
            triage_rows=triage_rows,
            local_structure_rows=local_structure_rows,
            panel_rows=panel_rows,
            candidate_rows=candidate_rows,
        ),
        "selection_funnel_stages": primary_panel_selection_trace_rows,
        "selected_candidate_ids": [str(row["candidate_id"]) for row in panel_rows],
        "panel_coverage": panel_coverage_summary(panel_rows),
        "handoff_readiness": build_handoff_readiness(
            selection_root=path.parent,
            panel_rows=panel_rows,
            candidate_handoff_path=candidate_handoff_path,
        ),
        "hard_gate_allowed_fold_classes": ["strong_fold_preserved"],
        "default_excluded_fold_classes": ["good_fold_preserved", "low_confidence", "review_band"],
        "panel_tie_break_order": [
            "largest nearest selected mutation-position Jaccard distance",
            "largest nearest selected exact-substitution Jaccard distance",
            "fewest nearest selected shared mutation positions",
            "fewest nearest selected shared exact substitutions",
            "fewest basic losses near retained DNA/RNA",
            "fewest Pro/Gly gains near retained DNA/RNA",
            "selection-support MSA observed fraction",
            "selection-support MSA mean alternate-residue frequency",
            "lowest C-terminal primer-RNA recognition-region C-alpha RMSD inside the gate",
            "lowest Wang thumb-contact-track C-alpha RMSD inside the gate",
            "fold metrics",
            "sequence hash",
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _manifest_relative_path(manifest_root: Path, target: Path) -> str:
    return os.path.relpath(target, start=manifest_root)


def _count_by(rows: list[dict[str, object]], key: str) -> dict[str, int]:
    counts = Counter(str(row.get(key) or "missing") for row in rows)
    return {value: counts[value] for value in sorted(counts)}


def _optional_review_sources(*, manifest_root: Path, paths: dict[str, Path]) -> dict[str, dict[str, object]]:
    return {
        key: {
            "path": _manifest_relative_path(manifest_root, paths[key]),
            "materialized": paths[key].exists(),
            "panel_selection_role": "review_annotation_not_selector",
        }
        for key in _OPTIONAL_REVIEW_SOURCE_KEYS
    }


def _local_structure_region_manifest_rows(local_structure_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    first_row_by_region: dict[str, dict[str, object]] = {}
    for row in local_structure_rows:
        region_id = str(row.get("region_id") or "")
        if region_id and region_id not in first_row_by_region:
            first_row_by_region[region_id] = row
    rows: list[dict[str, object]] = []
    for region_id in LOCAL_STRUCTURE_REGION_IDS:
        row = first_row_by_region.get(region_id)
        if row is None:
            raise ValueError(f"Missing local-structure region rows for manifest region: {region_id}")
        rows.append(
            {
                "region_id": region_id,
                "region_label": str(row.get("region_label") or ""),
                "region_role": str(row.get("region_role") or ""),
                "region_position_count": int(row.get("region_position_count") or 0),
                "region_position_spec": str(row.get("region_position_spec") or ""),
                "region_position_source": str(row.get("region_position_source") or ""),
                "region_source_basis_ids": _json_list(row.get("region_source_basis_ids_json")),
                "coordinate_scope": str(row.get("coordinate_scope") or ""),
                "local_ca_rmsd_threshold_angstrom": float(row["local_ca_rmsd_threshold_angstrom"]),
            }
        )
    return rows


def build_local_structure_source_basis_rows(
    *,
    repo_root: Path,
    local_structure_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return manifest source-basis rows referenced by local-structure regions."""

    used_ids = {
        source_id for row in local_structure_rows for source_id in _json_list(row.get("region_source_basis_ids_json"))
    }
    source_path = repo_root / "docs/studies/eco1_rt_repack/workbench/ontology/manual-mask-authority.yaml"
    if not used_ids or not source_path.exists():
        return []
    loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected manual mask authority mapping at {source_path}")
    rows: list[dict[str, object]] = []
    for source in loaded.get("source_basis", []):
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("id") or "")
        if source_id not in used_ids:
            continue
        rows.append(
            {
                "id": source_id,
                "role": str(source.get("role") or ""),
                "source_ref": str(source.get("source_ref") or ""),
                "note": str(source.get("note") or ""),
            }
        )
    missing = sorted(used_ids - {str(row["id"]) for row in rows})
    if missing:
        raise ValueError(f"Missing manual-mask source_basis rows for local-structure refs: {', '.join(missing)}")
    return rows


def _json_list(value: object) -> list[str]:
    if not value:
        return []
    loaded = yaml.safe_load(str(value))
    if not isinstance(loaded, list):
        return []
    return [str(item) for item in loaded]


def _plot_manifest_row(row: dict[str, object], *, manifest_root: Path) -> dict[str, object]:
    normalized = dict(row)
    normalized["path"] = str(Path(str(row["path"])).relative_to(manifest_root))
    normalized.update(SELECTION_PLOT_METADATA.get(str(row.get("plot_id") or ""), {}))
    return normalized
