"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/compiler/test_cap_sources.py

Cap-source lookup tests for the Retron MSD compiler.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.studies.units.retron_hairpin_design.catalog.cap_sources import (
    RetronMsdCapSourceError,
    load_msd_cap_source_lookup,
    parse_cap_source_label,
)

from ..support.paths import repo_root_from


def test_parse_cap_source_label_extracts_5to3_de033_cap_sequence() -> None:
    parsed = parse_cap_source_label("pES-retron-172-msd[TetR]; 033-GAG-AGA-CTC")

    assert parsed.construct_id == "pES-retron-172"
    assert parsed.payload_id == "TetR"
    assert parsed.source_family == "033"
    assert parsed.sequence_5to3 == "GAGAGACTC"


def test_checked_in_cap_source_lookup_keeps_de033_sources_explicit() -> None:
    repo_root = repo_root_from(__file__)
    registry = load_msd_cap_source_lookup(repo_root / "docs" / "studies" / "retron_hairpin_design")

    assert registry.sources["C26"].sequence_5to3 == "AGGC"
    assert registry.sources["C43"].sequence_5to3 == "TCCTCAGCCCGCTGAGGA"
    assert registry.sources["C43"].source_label == "retron-43-msd[TetR]; full-tCCTCAGcccGCTGAGGa"
    assert {
        cap_id: source.sequence_5to3
        for cap_id, source in registry.sources.items()
        if cap_id in {"C172", "C173", "C174", "C175", "C176"}
    } == {
        "C172": "GAGAGACTC",
        "C173": "GGAAGATCC",
        "C174": "AGAGACTCT",
        "C175": "GTAACGTAC",
        "C176": "GTGACGCAC",
    }
    assert registry.sources["C172"].source_label == "pES-retron-172-msd[TetR]; 033-GAG-AGA-CTC"


def test_cap_source_lookup_rejects_duplicate_yaml_mapping_keys(tmp_path: Path) -> None:
    study_dir = tmp_path / "study"
    compiler_dir = study_dir / "compiler" / "catalog"
    compiler_dir.mkdir(parents=True)
    (compiler_dir / "msd_cap_sources.yaml").write_text(
        """
contract: retron_msd_cap_source_lookup_v1
schema_version: 1
sources:
  C172:
    source_label: pES-retron-172-msd[TetR]; 033-GAG-AGA-CTC
    source_construct: pES-retron-172
    payload_id: TetR
    source_family: "033"
    sequence_5to3: GAGAGACTC
  C172:
    source_label: pES-retron-172-msd[TetR]; 033-GGA-AGA-TCC
    source_construct: pES-retron-172
    payload_id: TetR
    source_family: "033"
    sequence_5to3: GGAAGATCC
""",
        encoding="utf-8",
    )

    with pytest.raises(RetronMsdCapSourceError, match="duplicate mapping key: 'C172'"):
        load_msd_cap_source_lookup(study_dir)
