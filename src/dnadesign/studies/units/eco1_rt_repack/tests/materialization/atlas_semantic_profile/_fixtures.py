"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/atlas_semantic_profile/_fixtures.py

Fixtures for Eco1 Atlas semantic-profile materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_request import (
    materialize_foldcheck_request,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.foldcheck_request._fixtures import (
    write_minimal_foldcheck_inputs,
)
from dnadesign.thread.adapters.esm_atlas import sequence_md5
from dnadesign.thread.foldcheck import write_foldcheck_report


class FakeAtlasClient:
    def __init__(self, *, fail_for: set[str] | None = None, folded_for: set[str] | None = None) -> None:
        self._fail_for = fail_for or set()
        self._folded_for = folded_for or set()
        self.fold_on_miss_values: list[bool] = []

    def protein_lookup_by_sequence(
        self,
        sequence: str,
        *,
        topk_features: int,
        fold_on_miss: bool,
        normalize_features: bool,
    ) -> dict[str, Any]:
        del topk_features, normalize_features
        normalized = sequence.strip().upper()
        candidate_id = "thread_candidate_test" if normalized.startswith("AAAE") else "wild_type"
        self.fold_on_miss_values.append(fold_on_miss)
        if candidate_id in self._fail_for:
            raise RuntimeError(f"Atlas API fixture miss for {candidate_id}")
        response = protein_response(normalized)
        if candidate_id in self._folded_for:
            response.update(
                {
                    "folded_on_demand": True,
                    "pdb": PDB_TEXT,
                    "mean_plddt": 86.0,
                    "ptm": 0.91,
                    "pae_summary": {"mean": 4.0},
                }
            )
        return response


def write_foldcheck_report_fixture(tmp_path: Path, *, accepted_candidate_ids: set[str]) -> dict[str, Any]:
    """Write a two-sequence fold-check request/report fixture for Atlas tests."""

    write_minimal_foldcheck_inputs(tmp_path)
    request = materialize_foldcheck_request(repo_root=Path.cwd(), output_root=tmp_path)
    manifest = yaml.safe_load(request.request_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise AssertionError("fold-check request fixture must be a mapping")
    sequence_hashes = {str(row["sequence_id"]): str(row["sequence_hash"]) for row in manifest["sequences"]}
    rows: list[dict[str, object]] = []
    for candidate_id in ("wild_type", "thread_candidate_test"):
        if candidate_id in accepted_candidate_ids:
            rows.append(_accepted_foldcheck_row(candidate_id, sequence_hashes[candidate_id]))
        else:
            rows.append(_errored_foldcheck_row(candidate_id, sequence_hashes[candidate_id]))
    write_foldcheck_report(tmp_path / "foldcheck_report.parquet", rows, request_hash=str(manifest["request_hash"]))
    return manifest


def protein_response(sequence: str) -> dict[str, object]:
    return {
        "protein_hash": sequence_md5(sequence),
        "accession": "fixture_accession",
        "source": "fixture_source",
        "sequence": sequence,
        "sequence_length": len(sequence),
        "folded_on_demand": False,
        "sae_features": [
            {
                "feature_index": 14365,
                "value": 1.2,
                "label": "Polymerase thumb/palm nucleic acid binding",
                "description": "Fixture feature",
                "residue_regions": [{"start": 0, "end": 3, "peak_residue": 1, "mean_activation": 3.0}],
            },
            {
                "feature_index": 10777,
                "value": 0.9,
                "label": "RT/RdRp pre-catalytic region",
                "description": "Fixture feature",
                "residue_regions": [{"start": 1, "end": 2, "peak_residue": 1, "mean_activation": 2.0}],
            },
        ],
        "protein_activations": {
            "shape": [16384],
            "indices": [[14365, 10777]],
            "values": [1.2, 0.9],
        },
        "per_residue_activations": {
            "shape": [len(sequence), 16384],
            "indices": [[0, 1, 2, 3], [14365, 14365, 10777, 9008]],
            "values": [4.0, 3.0, 2.0, 1.0],
        },
    }


PDB_TEXT = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 86.00           C\nEND\n"


def _accepted_foldcheck_row(candidate_id: str, input_sequence_hash: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "runtime_kind": "alphafold_family_colabfold",
        "runtime_version": "colabfold-test",
        "input_sequence_hash": input_sequence_hash,
        "reference_structure_id": "ec86kit_7v9u_protomer1",
        "wt_baseline_artifact_id": "wild_type" if candidate_id != "wild_type" else "self",
        "runtime_parameters_hash": "sha256:" + "3" * 64,
        "threshold_id": "eco1_rt_foldcheck_thresholds_v1",
        "threshold_values": {"min_plddt": 70.0},
        "plddt": 80.0,
        "pae_summary": {"mean": 4.0},
        "backbone_rmsd_to_reference": 1.2,
        "protected_contact_retention": True,
        "status": "accepted",
        "rejection_reason": "",
        "missing_metric_reason": "",
    }


def _errored_foldcheck_row(candidate_id: str, input_sequence_hash: str) -> dict[str, object]:
    row = _accepted_foldcheck_row(candidate_id, input_sequence_hash)
    row.update(
        {
            "plddt": None,
            "pae_summary": {},
            "backbone_rmsd_to_reference": None,
            "protected_contact_retention": None,
            "status": "errored",
            "rejection_reason": "colabfold_output_missing",
            "missing_metric_reason": "colabfold_output_missing",
        }
    )
    return row
