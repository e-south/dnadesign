"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_composition_bundle.py

Adversarial bundle-publication tests for linear ssDNA composition outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.construct.src.composition.runtime import run_linear_ssdna_composition
from dnadesign.construct.src.contracts.errors import ValidationError


def _write_minimal_composition_config(tmp_path: Path, *, artifact_bundle: Path) -> Path:
    config_path = tmp_path / "minimal_composition.yaml"
    config_path.write_text(
        f"""
contract: linear_ssdna_composition_v1
schema_version: 1
composition_id: synthetic_x3
units:
  - unit_id: synthetic_unit
    repeat_count: 3
    segments:
      - segment_id: left
        sequence: AAAA
      - segment_id: payload
        sequence: ACGT
      - segment_id: payload_rc
        sequence: ACGT
        transform:
          kind: reverse_complement
          source_segment_id: payload
      - segment_id: right
        sequence: TTTT
    annotations:
      - annotation_id: payload_annotation
        role: payload
        location:
          basis: segment
          segment_id: payload
          start: 0
          end: 4
output:
  artifact_bundle: {artifact_bundle.as_posix()}
""",
        encoding="utf-8",
    )
    return config_path


def test_composition_bundle_rejects_nonempty_deprecated_visual_contract_dir(tmp_path: Path) -> None:
    artifact_bundle = tmp_path / "artifacts" / "synthetic_x3"
    stale_contract = artifact_bundle / "visual" / "contracts" / "stale.json"
    stale_contract.parent.mkdir(parents=True)
    stale_contract.write_text("{}\n", encoding="utf-8")
    config_path = _write_minimal_composition_config(tmp_path, artifact_bundle=artifact_bundle)

    with pytest.raises(
        ValidationError,
        match="Deprecated generated artifact directory 'visual/contracts' is non-empty",
    ):
        run_linear_ssdna_composition(config_path)

    assert stale_contract.is_file()
    assert not (artifact_bundle / "manifest.json").exists()
