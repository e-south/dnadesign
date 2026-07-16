"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/ledger/run_artifacts.py

Digest and path verification shared by run-pinned analysis artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from ...core.utils import ExitCodes, OpalError, file_sha256
from ...storage.artifacts import run_scoped_artifact_path
from .io import require_columns


def verified_run_artifact(
    runs_df: pl.DataFrame,
    *,
    outputs_dir: Path,
    round_k: int,
    run_id: str,
    artifact_key: str,
    context: str,
) -> tuple[Path, str]:
    """Resolve one digest-bound artifact inside its exact round directory."""

    require_columns(
        runs_df,
        ("as_of_round", "run_id", "artifacts"),
        ctx=context.lower(),
    )
    scoped = runs_df.filter((pl.col("as_of_round") == int(round_k)) & (pl.col("run_id").cast(pl.Utf8) == str(run_id)))
    if scoped.height != 1:
        raise OpalError(
            f"{context} require exactly one run row for round={int(round_k)}, "
            f"run_id={str(run_id)!r}; found {scoped.height}.",
            ExitCodes.CONTRACT_VIOLATION,
        )

    artifact_sha256, artifact_path = _artifact_reference(
        scoped.to_dicts()[0].get("artifacts"),
        artifact_key=artifact_key,
    )
    round_dir = (Path(outputs_dir) / "rounds" / f"round_{int(round_k)}").resolve()
    path = Path(artifact_path).expanduser().resolve()
    try:
        path.relative_to(round_dir)
    except ValueError as exc:
        raise OpalError(
            f"{context} artifact is outside its round directory: {path}",
            ExitCodes.CONTRACT_VIOLATION,
        ) from exc
    expected_path = run_scoped_artifact_path(
        round_dir,
        run_id=str(run_id),
        artifact_key=str(artifact_key),
    )
    if path != expected_path:
        raise OpalError(
            f"{context} artifact does not match its run-scoped path: expected={expected_path}, observed={path}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if not path.is_file():
        raise OpalError(f"{context} artifact not found: {path}", ExitCodes.CONTRACT_VIOLATION)
    actual_sha256 = file_sha256(path)
    if actual_sha256 != artifact_sha256:
        raise OpalError(
            f"{context} artifact SHA-256 does not match the run ledger "
            f"(expected={artifact_sha256}, actual={actual_sha256}).",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return path, artifact_sha256


def _artifact_reference(artifacts: object, *, artifact_key: str) -> tuple[str, str]:
    if not isinstance(artifacts, Mapping) or artifact_key not in artifacts:
        raise OpalError(
            f"Run ledger is missing artifact {artifact_key!r}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    value: Any = artifacts[artifact_key]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise OpalError(
            f"Run artifact {artifact_key!r} must contain [sha256, path].",
            ExitCodes.CONTRACT_VIOLATION,
        )
    sha256 = str(value[0]).strip().lower()
    path = str(value[1]).strip()
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise OpalError(
            f"Run artifact {artifact_key!r} has an invalid SHA-256 digest.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if not path:
        raise OpalError(
            f"Run artifact {artifact_key!r} has an empty path.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return sha256, path


__all__ = ["verified_run_artifact"]
