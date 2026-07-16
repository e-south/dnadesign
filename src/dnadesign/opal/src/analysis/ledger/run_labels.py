"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/ledger/run_labels.py

Verified access to the observed-label snapshot pinned by one OPAL run.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from ...core.utils import ExitCodes, OpalError, file_sha256
from .io import require_columns

LABELS_USED_ARTIFACT_KEY = "labels/labels_used.parquet"


@dataclass(frozen=True)
class RunLabelsUsed:
    """A verified label snapshot and the run scope that consumed it."""

    frame: pl.DataFrame
    path: Path
    sha256: str
    round_k: int
    run_id: str


def read_run_labels_used(
    runs_df: pl.DataFrame,
    *,
    outputs_dir: Path,
    round_k: int,
    run_id: str,
) -> RunLabelsUsed:
    """Read the digest-bound ``labels_used`` artifact for exactly one run."""

    require_columns(
        runs_df,
        ("as_of_round", "run_id", "artifacts"),
        ctx="run-pinned labels",
    )
    scoped = runs_df.filter((pl.col("as_of_round") == int(round_k)) & (pl.col("run_id").cast(pl.Utf8) == str(run_id)))
    if scoped.height != 1:
        raise OpalError(
            f"Run-pinned labels require exactly one run row for round={int(round_k)}, "
            f"run_id={str(run_id)!r}; found {scoped.height}.",
            ExitCodes.CONTRACT_VIOLATION,
        )

    artifact_sha256, artifact_path = _artifact_reference(scoped.to_dicts()[0].get("artifacts"))
    round_dir = (Path(outputs_dir) / "rounds" / f"round_{int(round_k)}").resolve()
    path = Path(artifact_path).expanduser().resolve()
    try:
        path.relative_to(round_dir)
    except ValueError as exc:
        raise OpalError(
            f"Run-pinned labels artifact is outside its round directory: {path}",
            ExitCodes.CONTRACT_VIOLATION,
        ) from exc
    if not path.is_file():
        raise OpalError(f"Run-pinned labels artifact not found: {path}", ExitCodes.CONTRACT_VIOLATION)
    actual_sha256 = file_sha256(path)
    if actual_sha256 != artifact_sha256:
        raise OpalError(
            "Run-pinned labels artifact SHA-256 does not match the run ledger "
            f"(expected={artifact_sha256}, actual={actual_sha256}).",
            ExitCodes.CONTRACT_VIOLATION,
        )

    frame = pl.read_parquet(path)
    _validate_labels_used_scope(frame, round_k=int(round_k), run_id=str(run_id))
    return RunLabelsUsed(
        frame=frame,
        path=path,
        sha256=artifact_sha256,
        round_k=int(round_k),
        run_id=str(run_id),
    )


def _artifact_reference(artifacts: object) -> tuple[str, str]:
    if not isinstance(artifacts, Mapping) or LABELS_USED_ARTIFACT_KEY not in artifacts:
        raise OpalError(
            f"Run ledger is missing artifact {LABELS_USED_ARTIFACT_KEY!r}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    value: Any = artifacts[LABELS_USED_ARTIFACT_KEY]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise OpalError(
            f"Run artifact {LABELS_USED_ARTIFACT_KEY!r} must contain [sha256, path].",
            ExitCodes.CONTRACT_VIOLATION,
        )
    sha256 = str(value[0]).strip().lower()
    path = str(value[1]).strip()
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise OpalError(
            f"Run artifact {LABELS_USED_ARTIFACT_KEY!r} has an invalid SHA-256 digest.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if not path:
        raise OpalError(
            f"Run artifact {LABELS_USED_ARTIFACT_KEY!r} has an empty path.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return sha256, path


def _validate_labels_used_scope(frame: pl.DataFrame, *, round_k: int, run_id: str) -> None:
    require_columns(
        frame,
        ("run_id", "as_of_round", "observed_round", "id", "y_obs"),
        ctx="run-pinned labels artifact",
    )
    if frame.is_empty():
        raise OpalError("Run-pinned labels artifact contains no rows.", ExitCodes.CONTRACT_VIOLATION)
    run_ids = {str(value) for value in frame.get_column("run_id").drop_nulls().to_list()}
    if run_ids != {str(run_id)} or frame.get_column("run_id").null_count():
        raise OpalError(
            f"Run-pinned labels artifact does not bind exactly to run_id={run_id!r}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    as_of_rounds = {int(value) for value in frame.get_column("as_of_round").drop_nulls().to_list()}
    if as_of_rounds != {int(round_k)} or frame.get_column("as_of_round").null_count():
        raise OpalError(
            f"Run-pinned labels artifact does not bind exactly to as_of_round={int(round_k)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    observed_rounds = frame.get_column("observed_round")
    if observed_rounds.null_count() or any(int(value) > int(round_k) for value in observed_rounds.to_list()):
        raise OpalError(
            "Run-pinned labels artifact contains labels observed after the run scope.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    ids = frame.get_column("id")
    string_ids = ids.cast(pl.Utf8)
    normalised_ids = string_ids.str.strip_chars()
    if (
        ids.null_count()
        or normalised_ids.eq("").any()
        or string_ids.ne(normalised_ids).any()
        or normalised_ids.n_unique() != frame.height
    ):
        raise OpalError(
            "Run-pinned labels artifact requires one canonical, non-empty row per candidate ID.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if frame.get_column("y_obs").null_count():
        raise OpalError("Run-pinned labels artifact contains null y_obs values.", ExitCodes.CONTRACT_VIOLATION)


__all__ = ["LABELS_USED_ARTIFACT_KEY", "RunLabelsUsed", "read_run_labels_used"]
