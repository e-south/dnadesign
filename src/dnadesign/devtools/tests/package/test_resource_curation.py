"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/package/test_resource_curation.py

Checks for complete, portable example resources in installed distributions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path


def _packaged_resources(tool: str) -> set[Path]:
    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
    project = tomllib.loads((root / "pyproject.toml").read_text())
    tool_root = root / "src/dnadesign" / tool
    patterns = project["tool"]["setuptools"]["package-data"][f"dnadesign.{tool}"]
    return {path for pattern in patterns for path in tool_root.glob(pattern) if path.is_file()}


def test_packaged_cruncher_examples_are_complete_and_curated() -> None:
    paths = _packaged_resources("cruncher")
    workspaces = {path.parts[path.parts.index("workspaces") + 1] for path in paths if "workspaces" in path.parts}
    assert workspaces == {"demo_pairwise", "demo_monotypic_tetr", "demo_yiu_payload"}
    motifs = [path for path in paths if "local_motifs" in path.parts]
    assert {path.name for path in motifs} == {"cpxR.txt", "lexA.txt", "tetR.tsv"}
    assert len(motifs) == 3
    assert any(path.name == "example_pwm_context.yaml" for path in paths)
    for path in paths:
        if path.suffix in {".yaml", ".txt", ".tsv"}:
            assert not re.search(r"/(?:global|scratch|Users|home)/", path.read_text()), path


def test_installed_ops_does_not_supply_a_lab_account() -> None:
    assert not any(path.name == "init_presets.yaml" for path in _packaged_resources("ops"))


def test_approved_binary_examples_have_no_machine_home_paths() -> None:
    import pyarrow.parquet as pq

    for tool in ("baserender", "opal"):
        for path in _packaged_resources(tool):
            if path.suffix != ".parquet":
                continue
            table = pq.read_table(path)
            text = json.dumps(table.to_pylist(), default=str) + str(table.schema.metadata)
            assert not re.search(r"/(?:Users|home)/[^/\s]+/", text), path
