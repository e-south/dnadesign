"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/msd/tests/test_public_surface.py

Contract tests for the public Retron MSD compiler surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.msd import (
    RankedPrimitiveSelectorSpec,
    RetronMsdRegistryError,
    compile_msd_design_unit,
    compute_scar_nick_profile,
    resolve_msd_compiler_spec_payload,
    validate_dna_sequence,
)


def _registry(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "contract": "retron_msd_design_registry_v1",
                "payloads": {},
                "caps": {},
                "constructs": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_explicit_request_compiles_without_a_study_directory(tmp_path: Path) -> None:
    resolved = resolve_msd_compiler_spec_payload(
        {
            "contract": "retron_msd_compiler_spec_v1",
            "schema_version": 1,
            "allow_non_ligatable_s0": True,
            "designs": [
                {
                    "construct_id": "candidate-1",
                    "payload_id": "payload-1",
                    "cap_id": "C1",
                    "left_base": "CGGG",
                    "right_base": "ACAG",
                }
            ],
            "payload_sequences": {"payload-1": "ACGT"},
            "cap_sequences": {"C1": "AGGC"},
        },
        registry_path=_registry(tmp_path / "registry.yaml"),
    )

    unit = compile_msd_design_unit(
        resolved.catalog.records[0],
        payload_sequences=resolved.payload_sequences,
        cap_sequences=resolved.cap_sequences,
    )

    assert unit.contract == "msd_compiled_unit_v1"
    assert unit.segment_sequence("payload_primary") == "ACGT"
    assert unit.segment_sequence("payload_complement") == "ACGT"


def test_registry_path_is_explicit_and_required(tmp_path: Path) -> None:
    with pytest.raises(RetronMsdRegistryError, match="not found"):
        resolve_msd_compiler_spec_payload(
            {
                "contract": "retron_msd_compiler_spec_v1",
                "schema_version": 1,
                "labels": ["candidate-msd[payload]; C1-LCGGG-RACAG"],
            },
            registry_path=tmp_path / "missing.yaml",
        )


def test_public_helpers_validate_sequence_and_rank_selection() -> None:
    assert validate_dna_sequence("acgt", label="payload") == "acgt"
    assert RankedPrimitiveSelectorSpec(rank=3).requested_ranks() == [3]
    assert compute_scar_nick_profile(left_base="CGGT", right_base="ACAG") == "MXMM"
