"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/io/manifest_io.py

Manifest IO helpers for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from ..contracts.errors import ContractViolationError
from .hashing import sha256_path
from .implementation import execution_identity
from .json_io import write_json


def write_manifest(path: Path, payload: dict) -> None:
    root = path.parent.resolve()
    outputs = []
    for output in payload.get("outputs", []):
        target = (root / output["path"]).resolve()
        if not target.is_relative_to(root) or target == root or target == path.resolve():
            raise ContractViolationError(
                f"artifact output must be contained in its artifact directory: {output['path']}"
            )
        outputs.append({**output, "digest": sha256_path(target)})
    write_json(path, {**payload, "outputs": outputs, "execution_identity": execution_identity()})
