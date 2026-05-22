"""
--------------------------------------------------------------------------------
<dnadesign project>
 src/dnadesign/opal/src/cli/commands/ingest_y/command.py

CLI command to ingest labels into OPAL campaigns. Validates inputs, applies
transforms, and writes the configured label source.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer

from ....core.utils import ExitCodes, OpalError, print_stdout
from ....runtime.ingest import run_ingest
from ...formatting import (
    bullet_list,
    render_ingest_commit_text,
    render_ingest_preview_text,
    render_ingest_runtime_text,
)
from ...guidance_hints import maybe_print_hints
from ...registry import cli_command
from .._common import (
    internal_error,
    json_out,
    opal_error,
    print_config_context,
    prompt_confirm,
)
from .context import build_ingest_command_context, normalize_unknown_sequences_policy
from .input_files import read_label_input_table
from .policy import (
    apply_unknown_sequence_policy,
)
from .preview import (
    build_ingest_nudges,
    rewrite_preview_warnings,
)
from .transforms import build_transform_context, resolve_transform_settings
from .values import required_columns_for_new_rows
from .writeback import commit_ingest_labels


@cli_command(
    "ingest-y",
    help="Ingest tidy CSV -> Y (strict checks), update the configured label source, and emit label events.",
)
def cmd_ingest_y(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: int = typer.Option(
        ...,
        "--round",
        "-r",
        "--observed-round",
        help="Observed round stamp for these labels.",
    ),
    csv: Path = typer.Option(..., "--csv", "--in", help="CSV/Parquet with raw reads"),
    transform: str = typer.Option(None, "--transform", help="Override YAML transform name"),
    params: Optional[Path] = typer.Option(None, "--params", help="JSON file (.json) with transform params"),
    apply: bool = typer.Option(False, "--apply", help="Apply ingest without interactive confirmation."),
    unknown_sequences: str = typer.Option(
        "create",
        "--unknown-sequences",
        help="Handling for sequences not in records: create (default), drop, or error.",
        case_sensitive=False,
    ),
    infer_missing_required: bool = typer.Option(
        False,
        "--infer-missing-required",
        help="Fill missing required columns for new sequences using inferred defaults (bio_type/alphabet).",
    ),
    if_exists: str = typer.Option(
        "fail",
        "--if-exists",
        help="Behavior if (id, round) already exists in the configured label source: fail, skip, or replace.",
        case_sensitive=False,
    ),
    no_hints: bool = typer.Option(False, "--no-hints", help="Disable next-step hints in text output."),
    json: bool = typer.Option(False, "--json/--text", help="Output format (default: text)"),
):
    try:
        unknown_sequences = normalize_unknown_sequences_policy(unknown_sequences)
        ingest_context = build_ingest_command_context(config, unknown_sequences=unknown_sequences)
        cfg_path = ingest_context.cfg_path
        cfg = ingest_context.cfg
        store = ingest_context.store
        label_source = ingest_context.label_source
        df = ingest_context.records_df
        ingest_runtime = ingest_context.ingest_runtime
        shared_label_source = ingest_context.shared_label_source
        if not json:
            print_config_context(cfg_path, cfg=cfg, records_path=store.records_path)

        csv_path, csv_df = read_label_input_table(csv)
        t_name, t_params = resolve_transform_settings(
            cfg,
            transform_name_override=transform,
            params_path=params,
            input_path=csv_path,
        )
        tctx = build_transform_context(cfg, round_index=int(round), transform_name=t_name)

        labels_df, preview = run_ingest(
            df,
            csv_df,
            transform_name=t_name,
            transform_params=t_params,
            y_expected_length=cfg.data.y_expected_length,
            y_column_name=cfg.data.y_column_name,
            duplicate_policy=cfg.ingest.duplicate_policy,
            ctx=tctx,
        )
        ingest_runtime = ingest_runtime.with_preview_counts(
            input_rows=len(csv_df),
            transformed_label_rows=len(labels_df),
            unknown_count_initial=int(preview.unknown_sequences or 0),
        )

        rewrite_preview_warnings(preview, label_source=label_source, unknown_sequences_policy=unknown_sequences)
        sample = labels_df.head(5).to_dict(orient="records")
        preview_payload = {
            "preview": asdict(preview),
            "sample": sample,
            "ingest_runtime": ingest_runtime.to_dict(),
        }
        if not json:
            print_stdout(render_ingest_preview_text(preview, sample, transform_name=t_name))
            print_stdout(render_ingest_runtime_text(ingest_runtime.to_dict()))

        required_cols = required_columns_for_new_rows(
            records_df=df,
            x_column_name=cfg.data.x_column_name,
            require_x_column=cfg.safety.write_back_requires_columns_present,
            shared_label_source=shared_label_source,
        )
        policy_result = apply_unknown_sequence_policy(
            records_df=df,
            csv_df=csv_df,
            labels_df=labels_df,
            preview=preview,
            label_source=label_source,
            unknown_sequences_policy=unknown_sequences,
            required_cols=required_cols,
            x_column_name=cfg.data.x_column_name,
            apply=apply,
            infer_missing_required=infer_missing_required,
            ingest_runtime=ingest_runtime,
        )
        labels_df = policy_result.labels_df
        csv_df = policy_result.csv_df
        unknown_sequences = policy_result.unknown_sequences_policy
        unknown_count = policy_result.unknown_count_after_policy
        dropped_missing_x = policy_result.dropped_missing_x
        ingest_runtime = policy_result.ingest_runtime
        if policy_result.aborted:
            return
        preview_payload["ingest_runtime"] = ingest_runtime.to_dict()

        if not json:
            nudges = build_ingest_nudges(
                preview,
                unknown_sequences_policy=unknown_sequences,
                unknown_count_after_policy=unknown_count,
                required_cols=required_cols,
                dropped_missing_x=dropped_missing_x,
            )
            if nudges:
                print_stdout(bullet_list("Nudges", nudges))

        if json and not apply:
            json_out({"ok": True, "applied": False, **preview_payload})
            return

        if not apply:
            if not prompt_confirm(
                f"Proceed to append {len(labels_df)} labels at observed_round={round}? (y/N): ",
                non_interactive_hint="No TTY available. Re-run with --apply to confirm ingest-y.",
            ):
                print_stdout("Aborted.")
                return

        commit = commit_ingest_labels(
            cfg=cfg,
            cfg_path=cfg_path,
            store=store,
            label_source=label_source,
            records_df=df,
            labels_df=labels_df,
            csv_df=csv_df,
            required_cols=required_cols,
            round_index=int(round),
            if_exists=if_exists,
            shared_label_source=shared_label_source,
        )
        out = commit.to_dict()
        if json:
            out.update(preview_payload)

        if json:
            json_out(out)
        else:
            print_stdout(
                render_ingest_commit_text(
                    round_index=out["round"],
                    labels_appended=out["labels_appended"],
                    labels_skipped=out["labels_skipped"],
                    y_column_updated=out["y_column_updated"],
                )
            )
            maybe_print_hints(
                command_name="ingest",
                cfg_path=cfg_path,
                no_hints=no_hints,
                json_output=json,
                observed_round=int(round),
            )
    except OpalError as e:
        opal_error("ingest-y", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("ingest-y", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
