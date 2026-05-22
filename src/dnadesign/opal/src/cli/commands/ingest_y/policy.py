"""
Unknown-sequence policy orchestration for `opal ingest-y`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ....core.utils import OpalError, print_stdout
from ....runtime.ingest import IngestPreview
from ....runtime.ingest_runtime import IngestRuntimeContract
from ....storage.label_sources import SharedObservedLabelSource
from .._common import prompt_confirm
from .metadata import apply_defaults_for_unknown, metadata_defaults_for_missing, missing_required_values_for_unknown
from .values import build_unknown_mask
from .x_checks import raise_on_stringified_list_x, raise_on_unknown_missing_x


@dataclass(frozen=True)
class UnknownSequencePolicyResult:
    labels_df: pd.DataFrame
    csv_df: pd.DataFrame
    unknown_sequences_policy: str
    unknown_count_after_policy: int
    dropped_missing_x: int
    ingest_runtime: IngestRuntimeContract
    aborted: bool = False


def apply_unknown_sequence_policy(
    *,
    records_df: pd.DataFrame,
    csv_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    preview: IngestPreview,
    label_source: object,
    unknown_sequences_policy: str,
    required_cols: list[str],
    x_column_name: str,
    apply: bool,
    infer_missing_required: bool,
    ingest_runtime: IngestRuntimeContract,
) -> UnknownSequencePolicyResult:
    unknown_mask = _unknown_mask(records_df=records_df, labels_df=labels_df)
    unknown_count = int(unknown_mask.sum())
    if (
        isinstance(label_source, SharedObservedLabelSource)
        and unknown_count > 0
        and unknown_sequences_policy == "create"
    ):
        raise OpalError(
            "Shared usr_sidecar label sources use a fixed candidate universe; "
            "unknown labels cannot create records. Use --unknown-sequences error to fail explicitly "
            "or --unknown-sequences drop to skip them."
        )

    raise_on_stringified_list_x(
        records_df=records_df,
        csv_df=csv_df,
        preview=preview,
        unknown_sequences_policy=unknown_sequences_policy,
        required_cols=required_cols,
        x_column_name=x_column_name,
    )

    if unknown_count > 0:
        if unknown_sequences_policy == "error":
            raise OpalError(
                f"{unknown_count} sequences not found in records. "
                "Use --unknown-sequences drop to skip them or provide required columns to create new rows."
            )
        if unknown_sequences_policy == "create" and x_column_name in required_cols:
            raise_on_unknown_missing_x(
                csv_df=csv_df,
                labels_df=labels_df,
                unknown_mask=unknown_mask,
                x_column_name=x_column_name,
            )
        if unknown_sequences_policy == "create":
            policy_result = _handle_create_policy(
                records_df=records_df,
                csv_df=csv_df,
                labels_df=labels_df,
                unknown_mask=unknown_mask,
                unknown_count=unknown_count,
                unknown_sequences_policy=unknown_sequences_policy,
                required_cols=required_cols,
                x_column_name=x_column_name,
                apply=apply,
                infer_missing_required=infer_missing_required,
                ingest_runtime=ingest_runtime,
            )
            if policy_result.aborted:
                return policy_result
            csv_df = policy_result.csv_df
            unknown_sequences_policy = policy_result.unknown_sequences_policy

        if unknown_sequences_policy == "drop":
            labels_df = labels_df.loc[~unknown_mask].copy()
            unknown_count = 0

    ingest_runtime = ingest_runtime.with_policy_counts(
        unknown_count_after_policy=unknown_count,
        labels_after_unknown_policy=len(labels_df),
    )
    return UnknownSequencePolicyResult(
        labels_df=labels_df,
        csv_df=csv_df,
        unknown_sequences_policy=unknown_sequences_policy,
        unknown_count_after_policy=unknown_count,
        dropped_missing_x=0,
        ingest_runtime=ingest_runtime,
    )


def _unknown_mask(*, records_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.Series:
    known_ids = set(records_df["id"].astype(str).tolist()) if "id" in records_df.columns else set()
    seq_to_id = {}
    if "sequence" in records_df.columns and "id" in records_df.columns:
        seq_to_id = (
            records_df[["sequence", "id"]].dropna().astype(str).drop_duplicates().set_index("sequence")["id"].to_dict()
        )
    return build_unknown_mask(labels_df, known_ids=known_ids, known_sequences=set(seq_to_id.keys()))


def _handle_create_policy(
    *,
    records_df: pd.DataFrame,
    csv_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    unknown_mask: pd.Series,
    unknown_count: int,
    unknown_sequences_policy: str,
    required_cols: list[str],
    x_column_name: str,
    apply: bool,
    infer_missing_required: bool,
    ingest_runtime: IngestRuntimeContract,
) -> UnknownSequencePolicyResult:
    unknown_sequences = []
    if "sequence" in labels_df.columns:
        unknown_sequences = labels_df.loc[unknown_mask, "sequence"].dropna().astype(str).unique().tolist()

    missing_set = _missing_metadata_set(
        csv_df=csv_df,
        unknown_sequences=unknown_sequences,
        required_cols=required_cols,
        x_column_name=x_column_name,
    )
    if missing_set and missing_set.issubset({"bio_type", "alphabet"}):
        defaults = metadata_defaults_for_missing(records_df=records_df, missing_set=missing_set)
        if infer_missing_required:
            apply_defaults_for_unknown(csv_df=csv_df, unknown_sequences=unknown_sequences, defaults=defaults)
        elif not apply:
            prompt = (
                "Input missing required metadata for new sequences: "
                f"{', '.join(sorted(missing_set))}. "
                f"Inferred defaults: {defaults}. Fill and continue? (y/N): "
            )
            if not prompt_confirm(
                prompt,
                non_interactive_hint=(
                    "No TTY available. Re-run without --apply to confirm defaults or pass --infer-missing-required."
                ),
            ):
                print_stdout("Aborted.")
                return _policy_result(
                    labels_df=labels_df,
                    csv_df=csv_df,
                    unknown_sequences_policy=unknown_sequences_policy,
                    unknown_count=unknown_count,
                    ingest_runtime=ingest_runtime,
                    aborted=True,
                )
            apply_defaults_for_unknown(csv_df=csv_df, unknown_sequences=unknown_sequences, defaults=defaults)
        else:
            raise OpalError(
                "Missing required metadata for new sequences. Re-run without --apply to confirm defaults "
                "or pass --infer-missing-required."
            )
        missing_set = _missing_metadata_set(
            csv_df=csv_df,
            unknown_sequences=unknown_sequences,
            required_cols=required_cols,
            x_column_name=x_column_name,
        )

    if missing_set:
        if not apply:
            prompt = (
                "Input missing required columns/values for new sequences: "
                f"{', '.join(sorted(missing_set))}. "
                "Drop unknown sequences and continue? (y/N): "
            )
            if prompt_confirm(
                prompt,
                non_interactive_hint="No TTY available. Re-run with --unknown-sequences drop to skip unknown rows.",
            ):
                unknown_sequences_policy = "drop"
            else:
                print_stdout("Aborted.")
                return _policy_result(
                    labels_df=labels_df,
                    csv_df=csv_df,
                    unknown_sequences_policy=unknown_sequences_policy,
                    unknown_count=unknown_count,
                    ingest_runtime=ingest_runtime,
                    aborted=True,
                )
        else:
            raise OpalError(
                "Missing required columns/values for new sequences. "
                "Provide the columns/values or use --unknown-sequences drop."
            )

    return _policy_result(
        labels_df=labels_df,
        csv_df=csv_df,
        unknown_sequences_policy=unknown_sequences_policy,
        unknown_count=unknown_count,
        ingest_runtime=ingest_runtime,
    )


def _missing_metadata_set(
    *,
    csv_df: pd.DataFrame,
    unknown_sequences: list[str],
    required_cols: list[str],
    x_column_name: str,
) -> set[str]:
    missing_required = [column for column in required_cols if column not in csv_df.columns]
    missing_required_values = missing_required_values_for_unknown(
        csv_df=csv_df,
        unknown_sequences=unknown_sequences,
        required_cols=required_cols,
        x_column_name=x_column_name,
    )
    return set(missing_required) | set(missing_required_values)


def _policy_result(
    *,
    labels_df: pd.DataFrame,
    csv_df: pd.DataFrame,
    unknown_sequences_policy: str,
    unknown_count: int,
    ingest_runtime: IngestRuntimeContract,
    aborted: bool = False,
) -> UnknownSequencePolicyResult:
    return UnknownSequencePolicyResult(
        labels_df=labels_df,
        csv_df=csv_df,
        unknown_sequences_policy=unknown_sequences_policy,
        unknown_count_after_policy=unknown_count,
        dropped_missing_x=0,
        ingest_runtime=ingest_runtime,
        aborted=aborted,
    )
