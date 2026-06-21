"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/support/registry.py

Minimal Retron MSD registry fixtures for compiler-spec tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def write_minimal_retron_msd_registry(tmp_path: Path) -> Path:
    study_dir = tmp_path / "study"
    compiler_dir = study_dir / "compiler" / "catalog"
    compiler_dir.mkdir(parents=True)
    (compiler_dir / "msd_design_registry.yaml").write_text(
        """
contract: retron_msd_design_registry_v1
schema_version: 1
payloads:
  TetR:
    display_name: msd[teto]
caps:
  C172:
    source_construct: retron-172
    snapback_topology:
      kind: snapback_foldback_geometry_v1
      retained_stem_span: {start: 0, end: 3}
      cap_span: {start: 3, end: 6}
      foldback_return_span: {start: 6, end: 9}
      source: de033 released-product 0/3/3 foldback geometry
constructs:
  pES-retron-177:
    source_notes: 26-derived base / 172-cap crossover; tests 172-cap permissiveness.
    scar_nick:
      route_status: note_only
      route_note: 26-derived base / 172-cap crossover
""",
        encoding="utf-8",
    )
    return study_dir
