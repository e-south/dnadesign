"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/ledger/run_observed_events.py

Verified access to every observed-label event available to one OPAL run.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ...core.utils import ExitCodes, OpalError
from ...storage.artifacts import OBSERVED_EVENTS_ARTIFACT_KEY
from .io import require_columns
from .run_artifacts import verified_run_artifact

SUPPORTED_LABEL_SOURCE_KINDS = frozenset({"campaign_history", "usr_sidecar"})


@dataclass(frozen=True)
class RunObservedEvents:
    """Every verified observed-label event available to one run."""

    frame: pl.DataFrame
    path: Path
    sha256: str
    round_k: int
    run_id: str


def read_run_observed_events(
    runs_df: pl.DataFrame,
    *,
    outputs_dir: Path,
    round_k: int,
    run_id: str,
) -> RunObservedEvents:
    """Read every digest-bound observed event visible to exactly one run."""

    path, artifact_sha256 = verified_run_artifact(
        runs_df,
        outputs_dir=outputs_dir,
        round_k=int(round_k),
        run_id=str(run_id),
        artifact_key=OBSERVED_EVENTS_ARTIFACT_KEY,
        context="Run-pinned observed events",
    )
    frame = pl.read_parquet(path)
    _validate_observed_events_scope(frame, round_k=int(round_k), run_id=str(run_id))
    return RunObservedEvents(
        frame=frame,
        path=path,
        sha256=artifact_sha256,
        round_k=int(round_k),
        run_id=str(run_id),
    )


def _validate_observed_events_scope(frame: pl.DataFrame, *, round_k: int, run_id: str) -> None:
    require_columns(
        frame,
        (
            "run_id",
            "as_of_round",
            "id",
            "display_label",
            "sequence",
            "observed_round",
            "batch_id",
            "y_space",
            "y_obs",
            "label_source_kind",
        ),
        ctx="run-pinned observed-events artifact",
    )
    if frame.is_empty():
        raise OpalError("Run-pinned observed-events artifact contains no rows.", ExitCodes.CONTRACT_VIOLATION)
    run_ids = {str(value) for value in frame.get_column("run_id").drop_nulls().to_list()}
    if run_ids != {str(run_id)} or frame.get_column("run_id").null_count():
        raise OpalError(
            f"Run-pinned observed-events artifact does not bind exactly to run_id={run_id!r}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    as_of_rounds = {int(value) for value in frame.get_column("as_of_round").drop_nulls().to_list()}
    if as_of_rounds != {int(round_k)} or frame.get_column("as_of_round").null_count():
        raise OpalError(
            f"Run-pinned observed-events artifact does not bind exactly to as_of_round={int(round_k)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    observed_rounds = frame.get_column("observed_round")
    if (
        observed_rounds.null_count()
        or any(int(value) < 0 for value in observed_rounds.to_list())
        or any(int(value) > int(round_k) for value in observed_rounds.to_list())
    ):
        raise OpalError(
            "Run-pinned observed-events artifact contains labels observed after the run scope or before round zero.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    _require_canonical_nonempty_strings(frame, "id", context="candidate IDs")
    _validate_optional_canonical_strings(frame, "display_label")
    _require_canonical_nonempty_strings(frame, "label_source_kind", context="label-source kinds")
    source_kinds = frame.get_column("label_source_kind").cast(pl.Utf8)
    if source_kinds.n_unique() != 1:
        raise OpalError(
            "Run-pinned observed-events artifact must contain exactly one label-source kind.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    source_kind = str(source_kinds[0])
    if source_kind not in SUPPORTED_LABEL_SOURCE_KINDS:
        raise OpalError(
            "Run-pinned observed-events artifact label-source kind must be supported; "
            f"observed={source_kind!r}, supported={sorted(SUPPORTED_LABEL_SOURCE_KINDS)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    _validate_nullable_source_identifier(
        frame,
        column="y_space",
        source_kinds=source_kinds,
        identifier_name="Y-space identifier",
    )
    if frame.get_column("y_space").drop_nulls().n_unique() > 1:
        raise OpalError(
            "Run-pinned observed-events artifact must contain exactly one Y space.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    _validate_nullable_source_identifier(
        frame,
        column="batch_id",
        source_kinds=source_kinds,
        identifier_name="batch_id",
    )
    if frame.get_column("y_obs").null_count():
        raise OpalError(
            "Run-pinned observed-events artifact contains null y_obs values.",
            ExitCodes.CONTRACT_VIOLATION,
        )


def _validate_nullable_source_identifier(
    frame: pl.DataFrame,
    *,
    column: str,
    source_kinds: pl.Series,
    identifier_name: str,
) -> None:
    values = frame.get_column(column)
    strings = values.cast(pl.Utf8)
    canonical_strings = strings.str.strip_chars()
    noncanonical = values.is_not_null() & (canonical_strings.eq("") | strings.ne(canonical_strings))
    disallowed_nulls = values.is_null() & source_kinds.ne("campaign_history")
    if noncanonical.any() or disallowed_nulls.any():
        raise OpalError(
            f"Run-pinned usr_sidecar observed events require a canonical, non-null, non-blank {identifier_name}; "
            f"campaign_history events may leave {column} null.",
            ExitCodes.CONTRACT_VIOLATION,
        )


def _require_canonical_nonempty_strings(frame: pl.DataFrame, column: str, *, context: str) -> None:
    values = frame.get_column(column)
    strings = values.cast(pl.Utf8)
    normalised = strings.str.strip_chars()
    if values.null_count() or normalised.eq("").any() or strings.ne(normalised).any():
        raise OpalError(
            f"Run-pinned observed-events artifact requires canonical, non-empty {context}.",
            ExitCodes.CONTRACT_VIOLATION,
        )


def _validate_optional_canonical_strings(frame: pl.DataFrame, column: str) -> None:
    values = frame.get_column(column)
    if values.dtype not in (pl.String, pl.Null):
        raise OpalError(
            f"Run-pinned observed-events artifact {column} values must be null or canonical non-blank strings.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    non_null = values.drop_nulls()
    if non_null.is_empty():
        return
    normalised = non_null.str.strip_chars()
    if normalised.eq("").any() or non_null.ne(normalised).any():
        raise OpalError(
            f"Run-pinned observed-events artifact {column} values must be null or canonical non-blank strings.",
            ExitCodes.CONTRACT_VIOLATION,
        )


__all__ = ["OBSERVED_EVENTS_ARTIFACT_KEY", "RunObservedEvents", "read_run_observed_events"]
