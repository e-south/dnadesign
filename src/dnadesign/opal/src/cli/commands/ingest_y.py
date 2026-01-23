# ABOUTME: CLI command to ingest labels into OPAL campaigns.
# ABOUTME: Validates inputs, applies transforms, and writes label history.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/ingest_y.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from ...core.round_context import PluginRegistryView, RoundCtx
from ...core.utils import ExitCodes, OpalError, now_iso, print_stdout
from ...registries.transforms_y import get_transform_y
from ...runtime.ingest import run_ingest
from ...storage.data_access import RecordsStore
from ...storage.ledger import LedgerWriter
from ...storage.locks import CampaignLock
from ...storage.parquet_io import read_parquet_df
from ...storage.workspace import CampaignWorkspace
from ...storage.writebacks import build_label_events
from ..formatting import (
    bullet_list,
    render_ingest_commit_human,
    render_ingest_preview_human,
)
from ..registry import cli_command
from ._common import (
    internal_error,
    json_out,
    load_cli_config,
    opal_error,
    print_config_context,
    prompt_confirm,
    resolve_config_path,
    resolve_json_path,
    resolve_table_path,
    store_from_cfg,
)


@cli_command(
    "ingest-y",
    help="Ingest tidy CSV → Y (strict checks), update label_hist, and emit label events.",
)
def cmd_ingest_y(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: int = typer.Option(
        ...,
        "--round",
        "-r",
        "--observed-round",
        help="Observed round stamp for these labels (writes to label history).",
    ),
    csv: Path = typer.Option(..., "--csv", "--in", help="CSV/Parquet with raw reads"),
    transform: str = typer.Option(None, "--transform", help="Override YAML transform name"),
    params: Optional[Path] = typer.Option(None, "--params", help="JSON file (.json) with transform params"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive prompt"),
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
        help="Behavior if (id, round) already exists in label history: 'fail' (default), 'skip', or 'replace'.",
        case_sensitive=False,
    ),
    json: bool = typer.Option(False, "--json/--human", help="Output format (default: human)"),
):
    try:
        import pandas as pd

        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        store: RecordsStore = store_from_cfg(cfg)
        df = store.load()
        if not json:
            print_config_context(cfg_path, cfg=cfg, records_path=store.records_path)

        # Resolve and read input file
        csv_path = resolve_table_path(csv, label="--csv", must_exist=True, allow_xlsx=True)
        csv_suffix = csv_path.suffix.lower()
        if csv_suffix in (".pq", ".parquet"):
            csv_df = read_parquet_df(csv_path)
        elif csv_suffix == ".xlsx":
            csv_df = pd.read_excel(csv_path)
        else:
            csv_df = pd.read_csv(csv_path)

        t_name = (transform or cfg.data.transforms_y.name).strip()
        t_params = cfg.data.transforms_y.params
        if params:
            import json as _json

            params_path = resolve_json_path(params, label="--params", must_exist=True)
            t_params = _json.loads(params_path.read_text())

        # Default SFXI delta enforcement (explicit Reader→OPAL hand-off)
        if t_name == "sfxi_vec8_from_table_v1":
            t_params = dict(t_params or {})
            if "expected_log2_offset_delta" not in t_params:
                obj_params = cfg.objective.objective.params or {}
                t_params["expected_log2_offset_delta"] = float(obj_params.get("intensity_log2_offset_delta", 0.0))
            if "enforce_log2_offset_match" not in t_params:
                t_params["enforce_log2_offset_match"] = True
            if "sfxi_log_json" not in t_params:
                candidate = csv_path.parent / "sfxi_log.json"
                if candidate.exists():
                    t_params["sfxi_log_json"] = str(candidate)

        reg = PluginRegistryView(
            model=cfg.model.name,
            objective=cfg.objective.objective.name,
            selection=cfg.selection.selection.name,
            transform_x=cfg.data.transforms_x.name,
            transform_y=t_name,
        )
        rctx = RoundCtx(
            core={
                "core/run_id": f"ingest-{now_iso()}",
                "core/round_index": int(round),
                "core/campaign_slug": cfg.campaign.slug,
                "core/labels_as_of_round": int(round),
                "core/plugins/transforms_y/name": t_name,
            },
            registry=reg,
        )
        tctx = rctx.for_plugin(category="transform_y", name=t_name, plugin=get_transform_y(t_name))

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

        # Preview
        sample = labels_df.head(5).to_dict(orient="records")
        if json:
            json_out({"preview": asdict(preview), "sample": sample})
        else:
            print_stdout(render_ingest_preview_human(preview, sample, transform_name=t_name))

        # Required columns for new rows (used for nudges + strict checks below)
        required_cols = ["bio_type", "alphabet"]
        if cfg.safety.write_back_requires_columns_present:
            if cfg.data.x_column_name not in df.columns:
                raise OpalError(f"records.parquet missing required X column '{cfg.data.x_column_name}'.")
            required_cols.append(cfg.data.x_column_name)

        unknown_sequences = (unknown_sequences or "create").strip().lower()
        if unknown_sequences not in {"create", "drop", "error"}:
            raise OpalError("--unknown-sequences must be one of: create, drop, error.")

        # Identify unknown rows (ids not resolved to existing records)
        known_ids = set(df["id"].astype(str).tolist()) if "id" in df.columns else set()
        unknown_mask = pd.Series(False, index=labels_df.index)
        if "id" in labels_df.columns:
            id_series = labels_df["id"]
            unknown_mask = id_series.isna()
            if id_series.notna().any() and known_ids:
                unknown_mask |= ~id_series.astype(str).isin(known_ids)
        unknown_count = int(unknown_mask.sum())

        # Missing required columns for new rows (only relevant if unknown rows exist and we intend to create)
        missing_required = [c for c in required_cols if c not in csv_df.columns]

        # Assertive check: list-valued X columns must arrive as lists (Parquet recommended).
        def _is_listlike(val: object) -> bool:
            return isinstance(val, (list, tuple, np.ndarray))

        def _col_is_listlike(series: pd.Series) -> bool:
            for v in series.head(20).tolist():
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                return _is_listlike(v)
            return False

        def _col_has_str(series: pd.Series) -> bool:
            return any(isinstance(v, str) for v in series.head(20).tolist())

        if preview.unknown_sequences and cfg.data.x_column_name in required_cols:
            x_col = cfg.data.x_column_name
            if x_col in df.columns and x_col in csv_df.columns:
                if _col_is_listlike(df[x_col]) and _col_has_str(csv_df[x_col]):
                    raise OpalError(
                        f"Column '{x_col}' appears list-valued in records.parquet, but CSV values are strings. "
                        "Use Parquet input (or a true list column) when adding new sequences with X."
                    )

        def _infer_default(series: pd.Series) -> Optional[str]:
            s = series.dropna().astype(str)
            if s.empty:
                return None
            mode = s.mode()
            if mode.empty:
                return None
            return str(mode.iloc[0])

        # Handle unknown sequences before final confirmation
        if unknown_count > 0:
            if unknown_sequences == "error":
                raise OpalError(
                    f"{unknown_count} sequences not found in records. "
                    "Use --unknown-sequences drop to skip them or provide required columns to create new rows."
                )
            if unknown_sequences == "create" and missing_required:
                missing_set = set(missing_required)
                defaults = {}
                if missing_set.issubset({"bio_type", "alphabet"}):
                    if "bio_type" in missing_set:
                        defaults["bio_type"] = _infer_default(df.get("bio_type", pd.Series(dtype=object)))
                    if "alphabet" in missing_set:
                        defaults["alphabet"] = _infer_default(df.get("alphabet", pd.Series(dtype=object)))
                    if any(v is None for v in defaults.values()):
                        raise OpalError(
                            "Missing required columns for new sequences, and defaults could not be inferred. "
                            "Provide the missing columns or use --unknown-sequences drop."
                        )
                    if infer_missing_required:
                        for col, val in defaults.items():
                            csv_df[col] = val
                    elif not yes:
                        prompt = (
                            "Input missing required columns for new sequences: "
                            f"{', '.join(missing_required)}. "
                            f"Inferred defaults: {defaults}. Fill and continue? (y/N): "
                        )
                        if not prompt_confirm(
                            prompt,
                            non_interactive_hint=(
                                "No TTY available. Re-run without --yes to confirm defaults or "
                                "pass --infer-missing-required."
                            ),
                        ):
                            print_stdout("Aborted.")
                            return
                        for col, val in defaults.items():
                            csv_df[col] = val
                    else:
                        raise OpalError(
                            "Missing required columns for new sequences. Re-run without --yes to confirm defaults "
                            "or pass --infer-missing-required."
                        )
                    missing_required = [c for c in required_cols if c not in csv_df.columns]
                if missing_required:
                    if not yes:
                        prompt = (
                            "Input missing required columns for new sequences: "
                            f"{', '.join(missing_required)}. "
                            "Drop unknown sequences and continue? (y/N): "
                        )
                        if prompt_confirm(
                            prompt,
                            non_interactive_hint=(
                                "No TTY available. Re-run with --unknown-sequences drop to skip unknown rows."
                            ),
                        ):
                            unknown_sequences = "drop"
                        else:
                            print_stdout("Aborted.")
                            return
                    else:
                        raise OpalError(
                            "Missing required columns for new sequences. "
                            "Provide the columns or use --unknown-sequences drop."
                        )

            if unknown_sequences == "drop":
                labels_df = labels_df.loc[~unknown_mask].copy()
                unknown_count = 0

        # Human-friendly nudges (non-fatal)
        if not json:
            nudges = list(preview.warnings or [])
            if preview.unknown_sequences and unknown_sequences == "drop":
                nudges = [
                    n
                    for n in nudges
                    if "sequences not found" not in n.lower() and "new rows will be created" not in n.lower()
                ]
                nudges.append(f"Dropping {preview.unknown_sequences} unknown sequences (--unknown-sequences drop).")
            if preview.unknown_sequences and required_cols and unknown_sequences != "drop":
                nudges.append(
                    "New sequences will be created; required columns for new rows: " + ", ".join(required_cols) + "."
                )
            if nudges:
                print_stdout(bullet_list("Nudges", nudges))

        if not yes:
            if not prompt_confirm(
                f"Proceed to append {len(labels_df)} labels at observed_round={round}? (y/N): ",
                non_interactive_hint="No TTY available. Re-run with --yes to confirm ingest-y.",
            ):
                print_stdout("Aborted.")
                return

        # Ensure rows exist; append to label_hist
        with CampaignLock(Path(cfg.campaign.workdir)):
            df = store.ensure_rows_exist(
                df,
                labels_df,
                csv_df,
                required_cols=required_cols,
                conflict_policy=cfg.safety.conflict_policy_on_duplicate_ids,
            )

            # Resolve ids for any rows that were missing id at transform time (new sequences)
            if labels_df["id"].isna().any():
                seq_to_id_full = (
                    df.set_index("sequence")["id"].astype(str).to_dict()
                    if "sequence" in df.columns and "id" in df.columns
                    else {}
                )
                labels_df = labels_df.copy()
                miss = labels_df["id"].isna()
                labels_df.loc[miss, "id"] = labels_df.loc[miss, "sequence"].map(seq_to_id_full)
                if labels_df["id"].isna().any():
                    raise OpalError("Failed to resolve ids for some labels; provide id or ensure sequences exist.")

            # Optional: preview duplicates at this round for better UX
            existing_ids: set[str] = set()
            try:
                lh = store.label_hist_col()
                ids_in = set(labels_df["id"].dropna().astype(str))
                if ids_in:
                    maybe = df.loc[df["id"].astype(str).isin(ids_in), ["id", lh]]
                    dup = 0
                    for _id, cell in maybe.itertuples(index=False, name=None):
                        for e in store._normalize_hist_cell(cell):
                            if e.get("kind") == "label" and int(e.get("observed_round", -1)) == int(round):
                                dup += 1
                                existing_ids.add(str(_id))
                                break
                    if dup > 0:
                        print_stdout(
                            f"[notice] {dup}/{len(ids_in)} incoming labels already have r={int(round)};"
                            f"applying --if-exists={if_exists}."
                        )  # noqa
            except Exception:
                pass

            # Apply skip policy by filtering labels before writes/events.
            if str(if_exists).lower().strip() == "skip" and existing_ids:
                labels_effective = labels_df.loc[~labels_df["id"].astype(str).isin(existing_ids)].copy()
            else:
                labels_effective = labels_df

            # 1) append to immutable label history (SSoT)
            df2 = store.append_labels_from_df(
                df,
                labels_effective[["id", "y"]],  # ids are now concrete
                r=int(round),
                src="ingest_y",
                fail_if_any_existing_labels=(str(if_exists).lower().strip() == "fail"),
                if_exists=str(if_exists).lower().strip(),
            )
            # 2) mirror "current y" into configured y_column_name for convenience
            df3 = store.upsert_current_y_column(df2, labels_effective[["id", "y"]], cfg.data.y_column_name)
            store.save_atomic(df3)

            # Emit label events (canonical SSoT)
            seq_map = df2.set_index("id")["sequence"].to_dict() if "sequence" in df2.columns else {}
            if not labels_effective.empty:
                events = build_label_events(
                    ids=labels_effective["id"].astype(str).tolist(),
                    sequences=[seq_map.get(str(_id)) for _id in labels_effective["id"].astype(str).tolist()],
                    y_obs=labels_effective["y"].tolist(),
                    observed_round=int(round),
                    src="ingest_y",
                    note=None,
                )
                ws = CampaignWorkspace.from_config(cfg, cfg_path)
                LedgerWriter(ws).append_label(events)

        out = {
            "ok": True,
            "round": int(round),
            "labels_appended": int(len(labels_effective)),
            "labels_skipped": int(len(labels_df) - len(labels_effective)),
            "y_column_updated": cfg.data.y_column_name,
        }

        if json:
            json_out(out)
        else:
            print_stdout(
                render_ingest_commit_human(
                    round_index=out["round"],
                    labels_appended=out["labels_appended"],
                    labels_skipped=out["labels_skipped"],
                    y_column_updated=out["y_column_updated"],
                )
            )
    except OpalError as e:
        opal_error("ingest-y", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("ingest-y", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
