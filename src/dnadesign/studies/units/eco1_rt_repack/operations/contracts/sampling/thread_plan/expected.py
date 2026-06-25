"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/sampling/thread_plan/expected.py

Expected thread-plan fields derived from the profile and mask set.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling.thread_plan.constants import (
    MASK_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling.thread_plan.io import (
    require_mapping,
    sha256_file,
)


def expected_request_fields(
    *,
    profile: Mapping[str, Any],
    mask_set: Mapping[str, Any],
    mask_set_path: Path,
) -> dict[str, Any]:
    sampling_policy = require_mapping(profile.get("sampling_policy"))
    backend_kind = str(sampling_policy.get("selected_backend"))
    seed_set = list(sampling_policy.get("seed_set", []))
    temperatures = [float(value) for value in sampling_policy.get("temperatures", [])]
    num_seq_per_target = int(sampling_policy.get("num_seq_per_target", 0))
    residues = mask_set.get("residues")
    rows = [row for row in residues if isinstance(row, Mapping)] if isinstance(residues, list) else []
    fixed_positions = [int(row["canonical_position"]) for row in rows if row.get("protected") is True]
    mutable_positions = [int(row["canonical_position"]) for row in rows if row.get("non_fixed") is True]
    missing_positions = [
        int(row["canonical_position"]) for row in rows if row.get("non_fixed_missing_backbone") is True
    ]
    return {
        "profile_id": profile.get("profile_id"),
        "backend_kind": backend_kind,
        "seed_set": seed_set,
        "temperature_schedule": temperatures,
        "batch_id": str(sampling_policy.get("batch_id")),
        "num_seq_per_target": num_seq_per_target,
        "batch_size": int(sampling_policy.get("batch_size", 0)),
        "expected_sample_count": len(seed_set) * len(temperatures) * num_seq_per_target,
        "fixed_positions": fixed_positions,
        "mutable_positions": mutable_positions,
        "excluded_non_fixed_missing_backbone_positions": missing_positions,
        "fixed_position_source": {
            "artifact_id": mask_set.get("artifact_id"),
            "path": str(mask_set_path),
            "hash": "sha256:" + sha256_file(mask_set_path),
            "mask_policy_id": MASK_POLICY_ID,
        },
    }
