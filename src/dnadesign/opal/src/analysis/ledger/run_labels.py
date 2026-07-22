"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/ledger/run_labels.py

Verified access to the training-label snapshot pinned by one OPAL run.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ...core.utils import ExitCodes, OpalError
from ...storage.artifacts import LABELS_USED_ARTIFACT_KEY
from .io import require_columns
from .run_artifacts import verified_run_artifact


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

    path, artifact_sha256 = verified_run_artifact(
        runs_df,
        outputs_dir=outputs_dir,
        round_k=int(round_k),
        run_id=str(run_id),
        artifact_key=LABELS_USED_ARTIFACT_KEY,
        context="Run-pinned labels",
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
