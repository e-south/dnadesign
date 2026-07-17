"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_bundle_verification.py

Fail-closed verification for multistate behavior shadow bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..evaluation.multistate_behavior_protocol import load_multistate_behavior_protocol
from .multistate_behavior_bundle_contract import SCHEMA_ID, TABLE_COLUMNS, TABLE_IDS
from .multistate_behavior_completion_verification import verify_behavior_completion_tables
from .multistate_behavior_decision_verification import verify_behavior_decision_artifacts
from .multistate_behavior_json import load_strict_behavior_json
from .multistate_behavior_semantic_verification import verify_behavior_record_semantics
from .multistate_behavior_source_equivalence import verify_source_equivalence_receipt
from .multistate_behavior_source_receipt import verify_behavior_prediction_source_receipt
from .multistate_behavior_table_coverage import verify_behavior_table_coverage
from .multistate_behavior_table_derivations import verify_behavior_table_derivations
from .multistate_behavior_table_provenance import verify_behavior_table_provenance
from .publication import sha256_file

_PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "config/multistate_response_behavior_shadow_v1.yaml"


def verify_multistate_behavior_shadow(bundle_root: Path) -> dict[str, object]:
    """Verify bytes, semantics, derivations, provenance, and shadow posture."""

    root = Path(bundle_root).resolve()
    manifest_path = root / "manifest.json"
    manifest = load_strict_behavior_json(manifest_path)
    artifacts = _verify_artifacts(root, manifest_path=manifest_path, manifest=manifest)
    normalization = load_strict_behavior_json(root / str(artifacts["normalization"]["path"]))
    decision = load_strict_behavior_json(root / str(artifacts["decision"]["path"]))
    audit = load_strict_behavior_json(root / str(artifacts["independent_adversarial_audit"]["path"]))
    source_equivalence = load_strict_behavior_json(root / str(artifacts["source_equivalence"]["path"]))
    protocol = load_multistate_behavior_protocol(_PROTOCOL_PATH)
    semantics = verify_behavior_record_semantics(
        manifest,
        normalization,
        protocol=protocol,
    )
    prediction_source = verify_behavior_prediction_source_receipt(manifest["source"]["prediction"])
    if prediction_source["candidate_count"] != semantics.prediction_count:
        raise ValueError("prediction receipt count disagrees with bundle semantics.")
    tables = _load_tables(root, manifest=manifest, artifacts=artifacts)
    normalization_values = normalization["normalization"]
    verify_behavior_table_coverage(
        tables,
        semantics=semantics,
        scale_quantile=float(normalization_values["scale_quantile"]),
        quantile_method=str(normalization_values["quantile_method"]),
    )
    comparator_semantics = (
        f"{protocol.comparator_objective_name}.{protocol.comparator_score_channel}.{protocol.comparator_direction}"
    )
    verify_behavior_table_provenance(
        tables,
        semantics=semantics,
        objective_name=protocol.objective_name,
        primary_reduction_id=protocol.primary_reduction_id,
        comparator_semantics=comparator_semantics,
    )
    verify_behavior_table_derivations(
        tables,
        semantics=semantics,
        comparator_semantics=comparator_semantics,
    )
    verify_behavior_completion_tables(
        tables,
        semantics=semantics,
        protocol=protocol,
        reader_bundle_manifest_sha256=str(manifest["source"]["reader_bundle_manifest_sha256"]),
    )
    verify_behavior_decision_artifacts(
        root,
        manifest=manifest,
        artifacts=artifacts,
        decision=decision,
        audit=audit,
        tables=tables,
        protocol=protocol,
    )
    verify_source_equivalence_receipt(
        source_equivalence,
        decision_source=decision["source_equivalence"],
        study_id=protocol.study_id,
        protocol_id=protocol.protocol_id,
        corrected_reader_bundle_manifest_sha256=str(manifest["source"]["reader_bundle_manifest_sha256"]),
        promoted_candidate_count=int(tables["grouped_objective_validation"]["candidate_id"].nunique()),
        grouped_validation=tables["grouped_objective_validation"],
    )
    return manifest


def _verify_artifacts(
    root: Path,
    *,
    manifest_path: Path,
    manifest: dict[str, object],
) -> dict[str, dict[str, object]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("multistate behavior bundle lacks an artifact inventory.")
    expected_ids = {
        "normalization",
        "decision",
        "report",
        "independent_adversarial_audit",
        "source_equivalence",
        "plot__normalization_robustness",
        "plot__grouped_objective_validation",
        "plot__allocation_family_decomposition",
        *(f"table__{table_id}" for table_id in TABLE_IDS),
    }
    if set(artifacts) != expected_ids:
        raise ValueError("multistate behavior artifact identities are incomplete or unexpected.")
    expected_paths = {manifest_path}
    for artifact_id, record in artifacts.items():
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise ValueError(f"artifact {artifact_id!r} receipt is invalid.")
        raw_path = record["path"]
        if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
            raise ValueError(f"artifact {artifact_id!r} path is invalid.")
        path = (root / raw_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"artifact {artifact_id!r} is missing or escapes the bundle.")
        size = record["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"artifact {artifact_id!r} byte count is invalid.")
        if path.stat().st_size != size or sha256_file(path) != record["sha256"]:
            raise ValueError(f"artifact {artifact_id!r} size or digest mismatch.")
        expected_paths.add(path)
    actual_paths = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if actual_paths != expected_paths:
        raise ValueError("multistate behavior artifact inventory does not match bundle files.")
    return artifacts


def _load_tables(
    root: Path,
    *,
    manifest: dict[str, object],
    artifacts: dict[str, dict[str, object]],
) -> dict[str, pd.DataFrame]:
    contracts = manifest.get("tables")
    if not isinstance(contracts, dict) or set(contracts) != TABLE_IDS:
        raise ValueError("multistate behavior table contracts are incomplete or unexpected.")
    loaded: dict[str, pd.DataFrame] = {}
    for table_id, expected_columns in TABLE_COLUMNS.items():
        contract = contracts[table_id]
        if not isinstance(contract, dict) or set(contract) != {"rows", "columns"}:
            raise ValueError(f"table {table_id!r} contract must declare rows and columns.")
        rows = contract["rows"]
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise ValueError(f"table {table_id!r} row count must be a nonnegative integer.")
        if contract["columns"] != list(expected_columns):
            raise ValueError(f"table {table_id!r} manifest columns disagree with the schema.")
        frame = pd.read_parquet(root / str(artifacts[f"table__{table_id}"]["path"]))
        if len(frame) != rows or tuple(frame.columns) != expected_columns:
            raise ValueError(f"table {table_id!r} row count or columns drifted.")
        loaded[table_id] = frame
    return loaded


__all__ = ["SCHEMA_ID", "TABLE_IDS", "verify_multistate_behavior_shadow"]
