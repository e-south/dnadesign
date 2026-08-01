"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/subject_bindings/sources.py

Confined source access for RT-lnRNA subject-binding authorities.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from Bio import SeqIO

from .contracts import SubjectBindingContractError


class SourceCache:
    """Reuse immutable, confined source documents during one registry load."""

    def __init__(self) -> None:
        self._yaml_by_path: dict[Path, object] = {}
        self._genbank_sequence_by_path: dict[Path, str] = {}
        self._file_sha256_by_path: dict[Path, str] = {}

    def load_yaml(self, path: Path) -> object:
        resolved = path.resolve()
        if resolved not in self._yaml_by_path:
            self._yaml_by_path[resolved] = load_yaml(resolved)
        return self._yaml_by_path[resolved]

    def load_genbank_sequence(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved not in self._genbank_sequence_by_path:
            self._genbank_sequence_by_path[resolved] = str(SeqIO.read(resolved, "genbank").seq).upper()
        return self._genbank_sequence_by_path[resolved]

    def load_file_sha256(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved not in self._file_sha256_by_path:
            self._file_sha256_by_path[resolved] = f"sha256:{hashlib.sha256(resolved.read_bytes()).hexdigest()}"
        return self._file_sha256_by_path[resolved]


def source_path(root: Path, value: str, *, label: str) -> Path:
    return contained_file(root, value, label=f"{label}.source_path")


def contained_file(base: Path, value: str, *, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise SubjectBindingContractError(f"{label}: path must be relative without parent traversal")
    resolved_base = Path(base).resolve()
    path = (resolved_base / relative).resolve()
    try:
        path.relative_to(resolved_base)
    except ValueError as exc:
        raise SubjectBindingContractError(f"{label}: resolved path must remain inside its owning directory") from exc
    if not path.is_file():
        raise SubjectBindingContractError(f"{label}: source file is missing: {value}")
    return path


def load_yaml(path: Path) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SubjectBindingContractError(f"unable to read {path}: {exc}") from exc


__all__ = ["SourceCache", "contained_file", "load_yaml", "source_path"]
