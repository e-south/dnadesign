"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/colabfold/index.py

ColabFold output-file indexing.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_RANK_PATTERN = re.compile(r"(?:^|_)rank_?(\d+)(?:_|\.|$)")
_COLABFOLD_GENERATED_PREFIXES = (
    "_relaxed_",
    "_unrelaxed_",
    "_scores_",
    "_predicted_aligned_error_",
    "_pae_",
)


@dataclass(frozen=True)
class ColabFoldOutputIndex:
    """One-pass index over a ColabFold output directory."""

    model_paths: tuple[Path, ...]
    score_paths: tuple[Path, ...]
    model_paths_by_sequence: dict[str, tuple[Path, ...]]
    score_paths_by_sequence: dict[str, tuple[Path, ...]]

    @classmethod
    def from_output_root(
        cls,
        output_root: Path,
        *,
        sequence_ids: Iterable[str] | None = None,
    ) -> "ColabFoldOutputIndex":
        """Build a reusable file index for ColabFold output discovery."""

        model_paths: list[Path] = []
        score_paths: list[Path] = []
        for path in output_root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix == ".pdb":
                model_paths.append(path)
            elif path.suffix == ".json" and _is_score_like_json(path.name):
                score_paths.append(path)

        ordered_sequence_ids = _ordered_sequence_ids(sequence_ids)
        return cls(
            model_paths=tuple(model_paths),
            score_paths=tuple(score_paths),
            model_paths_by_sequence=_bucket_paths(model_paths, ordered_sequence_ids),
            score_paths_by_sequence=_bucket_paths(score_paths, ordered_sequence_ids),
        )

    def select_model_pdb(self, sequence_id: str) -> Path | None:
        candidates = self.model_paths_by_sequence.get(sequence_id)
        if candidates is None:
            candidates = tuple(path for path in self.model_paths if _matches_sequence_file(path.name, sequence_id))
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda path: (_rank_key(path.name), _model_kind_priority(path.name), path.name),
        )[0]

    def select_score_json(self, sequence_id: str) -> Path | None:
        candidates = self.score_paths_by_sequence.get(sequence_id)
        if candidates is None:
            candidates = tuple(path for path in self.score_paths if _matches_sequence_file(path.name, sequence_id))
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda path: (_rank_key(path.name), _score_kind_priority(path.name), path.name),
        )[0]


def _bucket_paths(paths: Iterable[Path], ordered_sequence_ids: tuple[str, ...]) -> dict[str, tuple[Path, ...]]:
    buckets: dict[str, list[Path]] = {sequence_id: [] for sequence_id in ordered_sequence_ids}
    for path in paths:
        for sequence_id in ordered_sequence_ids:
            if _matches_sequence_file(path.name, sequence_id):
                buckets[sequence_id].append(path)
                break
    return {sequence_id: tuple(sequence_paths) for sequence_id, sequence_paths in buckets.items()}


def _ordered_sequence_ids(sequence_ids: Iterable[str] | None) -> tuple[str, ...]:
    if sequence_ids is None:
        return ()
    clean_ids = {str(sequence_id) for sequence_id in sequence_ids if str(sequence_id)}
    return tuple(sorted(clean_ids, key=lambda sequence_id: (-len(sequence_id), sequence_id)))


def _matches_sequence_file(name: str, sequence_id: str) -> bool:
    if name == f"{sequence_id}{Path(name).suffix}":
        return True
    return any(name.startswith(f"{sequence_id}{prefix}") for prefix in _COLABFOLD_GENERATED_PREFIXES)


def _rank_key(name: str) -> int:
    match = _RANK_PATTERN.search(name)
    return int(match.group(1)) if match is not None else 9999


def _model_kind_priority(name: str) -> int:
    if "_relaxed_" in name and "_unrelaxed_" not in name:
        return 0
    return 1


def _score_kind_priority(name: str) -> int:
    if "score" in name:
        return 0
    if "pae" in name or "aligned_error" in name:
        return 1
    return 2


def _is_score_like_json(name: str) -> bool:
    return "score" in name or "pae" in name or "aligned_error" in name
