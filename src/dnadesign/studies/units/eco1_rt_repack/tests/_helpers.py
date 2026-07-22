"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/_helpers.py

Shared test helpers for the Eco1 RT repack study unit.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def load_yaml(rel_path: str) -> dict[str, Any]:
    payload = yaml.safe_load((repo_root() / rel_path).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def ec86kit_source_artifacts_available() -> bool:
    root = repo_root()
    structure_sources = load_yaml("docs/studies/eco1_rt_repack/workbench/provenance/structure-sources.yaml")
    numbering_policy = load_yaml("docs/studies/eco1_rt_repack/workbench/provenance/residue-numbering-policy.yaml")
    selected_source = structure_sources["selected_source"]
    required_refs = (
        selected_source["ec86kit_manifest_ref"],
        selected_source["ec86kit_model_ref"],
        numbering_policy["source_map_ref"],
        numbering_policy["source_distance_profile_ref"],
    )
    return all(_resolve_contract_ref(root, str(ref)).exists() for ref in required_refs)


def require_ec86kit_source_artifacts() -> None:
    if not ec86kit_source_artifacts_available():
        pytest.skip("requires sibling ec86kit structure-authority artifacts")


def _resolve_contract_ref(root: Path, source_ref: str) -> Path:
    if source_ref.startswith("sibling:"):
        return (root / source_ref.removeprefix("sibling:")).resolve()
    if source_ref.startswith("repo:"):
        return (root / source_ref.removeprefix("repo:")).resolve()
    path = Path(source_ref).expanduser()
    return path if path.is_absolute() else (root / path).resolve()
