"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/pinned_checkout.py

Content identity checks for files in a pinned LigandMPNN checkout.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath


def attested_working_tree_path_bytes(root: Path, commit: str, path: str) -> bytes | None:
    """Return working-tree bytes only when they match the pinned commit blob."""

    pinned_bytes = _pinned_path_bytes(root, commit, path)
    working_tree_bytes = _working_tree_path_bytes(root, path)
    if pinned_bytes is None or working_tree_bytes is None:
        return None
    return working_tree_bytes if working_tree_bytes == pinned_bytes else None


def working_tree_path_matches_commit(root: Path, commit: str, path: str) -> bool | None:
    """Report whether working-tree bytes match the path's pinned commit blob."""

    pinned_bytes = _pinned_path_bytes(root, commit, path)
    working_tree_bytes = _working_tree_path_bytes(root, path)
    if pinned_bytes is None or working_tree_bytes is None:
        return None
    return working_tree_bytes == pinned_bytes


def index_path_matches_commit(root: Path, commit: str, path: str) -> bool | None:
    """Report whether the stage-0 index blob matches the pinned commit blob."""

    pinned_bytes = _pinned_path_bytes(root, commit, path)
    index_bytes = _index_path_bytes(root, path)
    if pinned_bytes is None or index_bytes is None:
        return None
    return index_bytes == pinned_bytes


def materialize_pinned_tree(root: Path, commit: str, destination: Path) -> None:
    """Materialize regular tracked blobs from one exact commit without replacement refs."""

    try:
        tree = subprocess.check_output(
            ["git", "--no-replace-objects", "-C", str(root), "ls-tree", "-r", "-z", "--full-tree", commit],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("LigandMPNN pinned source tree could not be read") from exc
    for record in tree.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split()
            relative_path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("LigandMPNN pinned source tree contains an invalid entry") from exc
        path = PurePosixPath(relative_path)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("LigandMPNN pinned source tree contains an unsafe path")
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(f"LigandMPNN pinned source tree contains unsupported entry: {relative_path}")
        if path.suffix == ".pyc" or "__pycache__" in path.parts:
            continue
        try:
            payload = subprocess.check_output(
                ["git", "--no-replace-objects", "-C", str(root), "cat-file", "blob", object_id],
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(f"LigandMPNN pinned blob could not be read: {relative_path}") from exc
        output_path = destination.joinpath(*path.parts)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)
        output_path.chmod(0o755 if mode == "100755" else 0o644)


def _pinned_path_bytes(root: Path, commit: str, path: str) -> bytes | None:
    try:
        return subprocess.check_output(
            ["git", "--no-replace-objects", "-C", str(root), "show", f"{commit}:{path}"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _working_tree_path_bytes(root: Path, path: str) -> bytes | None:
    try:
        return (root / path).read_bytes()
    except OSError:
        return None


def _index_path_bytes(root: Path, path: str) -> bytes | None:
    try:
        return subprocess.check_output(
            ["git", "--no-replace-objects", "-C", str(root), "show", f":{path}"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
