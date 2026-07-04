"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/contact_geometry/test_yaml_io.py

Tests for Eco1 RT materialization YAML helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    dump_yaml,
    load_yaml,
    write_yaml,
)


def test_yaml_io_rejects_unsafe_python_tags(tmp_path: Path) -> None:
    payload_path = tmp_path / "unsafe.yaml"
    payload_path.write_text('!!python/object/apply:os.system ["echo unsafe"]\n', encoding="utf-8")

    with pytest.raises(yaml.constructor.ConstructorError):
        load_yaml(payload_path)


def test_yaml_io_preserves_mapping_order_when_writing(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.yaml"

    write_yaml(payload_path, {"schema_id": "demo", "schema_version": 1, "status": "ok"})

    assert payload_path.read_text(encoding="utf-8").splitlines()[:3] == [
        "schema_id: demo",
        "schema_version: 1",
        "status: ok",
    ]
    assert dump_yaml({"b": 1, "a": 2}).splitlines() == ["b: 1", "a: 2"]
