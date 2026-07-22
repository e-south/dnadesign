"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/bundle_io.py

Confined paths, digests, and atomic publication for binding bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from uuid import uuid4

from .contracts import PromoterCandidateBindingsError


def confined_path(path: Path, *, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PromoterCandidateBindingsError(f"{label} is outside allowed output root {root}: {resolved}") from exc
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_complete_bundle(*, staged_dir: Path, output_dir: Path) -> None:
    """Replace a bundle directory and preserve a recoverable prior bundle on failure."""

    if not output_dir.exists():
        try:
            os.replace(staged_dir, output_dir)
        except OSError as exc:
            raise PromoterCandidateBindingsError(f"Could not publish complete binding bundle: {exc}") from exc
        return

    backup_dir = output_dir.parent / f".{output_dir.name}.backup-{uuid4().hex}"
    try:
        os.replace(output_dir, backup_dir)
    except OSError as exc:
        raise PromoterCandidateBindingsError(f"Could not prepare binding-bundle replacement: {exc}") from exc
    try:
        os.replace(staged_dir, output_dir)
    except OSError as publish_exc:
        try:
            os.replace(backup_dir, output_dir)
        except OSError as rollback_exc:
            raise PromoterCandidateBindingsError(
                "Could not publish the binding bundle or restore its prior version; "
                f"the prior bundle remains recoverable at {backup_dir}: "
                f"publish={publish_exc}; rollback={rollback_exc}"
            ) from rollback_exc
        raise PromoterCandidateBindingsError(
            f"Could not publish complete binding bundle; restored prior bundle: {publish_exc}"
        ) from publish_exc
    try:
        shutil.rmtree(backup_dir)
    except OSError as exc:
        raise PromoterCandidateBindingsError(
            f"Published the binding bundle but could not remove prior-bundle backup {backup_dir}: {exc}"
        ) from exc


__all__ = ["confined_path", "file_sha256", "publish_complete_bundle"]
