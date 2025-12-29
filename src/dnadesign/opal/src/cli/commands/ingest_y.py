"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/ingest_y.py

Ingest tidy CSV → Y (transform_y), preview, confirm, write to records caches,
AND emit canonical label events (kind="label") to outputs/ledger.labels.parquet.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from ...artifacts import append_events, events_path
from ...data_access import RecordsStore
from ...ingest import run_ingest
from ...locks import CampaignLock
from ...utils import ExitCodes, OpalError, print_stdout
from ...writebacks import build_label_events
from ..formatting import render_ingest_commit_human, render_ingest_preview_human
from ..registry import cli_command
from ._common import (
    internal_error,
    json_out,
    load_cli_config,
    opal_error,
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
        help="Observed round to stamp on these labels (alias: --observed-round)",
    ),
    csv: Path = typer.Option(..., "--csv", "--in", help="CSV/Parquet with raw reads"),
    transform: str = typer.Option(None, "--transform", help="Override YAML transform name"),
    params: Optional[Path] = typer.Option(None, "--params", help="JSON file with transform params"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive prompt"),
    if_exists: str = typer.Option(
        "fail",
        "--if-exists",
        help="Behavior if (id, round) already exists in label history: 'fail' (default), 'skip', or 'replace'.",
        case_sensitive=False,
    ),
    json: bool = typer.Option(False, "--json/--human", help="Output format (default: human)"),
):
    try:
        cfg = load_cli_config(config)
        store: RecordsStore = store_from_cfg(cfg)
        df = store.load()

        # Resolve and read input file
        csv_path = Path(csv)
        if not csv_path.is_absolute():
            csv_path = (Path.cwd() / csv_path).resolve()
        if not csv_path.exists():
            raise OpalError(f"CSV not found: {csv_path}")
        csv_df = pd.read_parquet(csv_path) if csv_path.suffix.lower() in (".pq", ".parquet") else pd.read_csv(csv_path)

        t_name = (transform or cfg.data.transforms_y.name).strip()
        t_params = cfg.data.transforms_y.params
        if params:
            import json as _json

            t_params = _json.loads(Path(params).read_text())

        labels_df, preview = run_ingest(
            df,
            csv_df,
            transform_name=t_name,
            transform_params=t_params,
            y_expected_length=cfg.data.y_expected_length,
            y_column_name=cfg.data.y_column_name,
        )

        # Preview
        sample = labels_df.head(5).to_dict(orient="records")
        if json:
            json_out({"preview": asdict(preview), "sample": sample})
        else:
            print_stdout(render_ingest_preview_human(preview, sample, transform_name=t_name))

        if not yes:
            resp = (
                input(f"Proceed to append {len(labels_df)} labels at observed_round={round}? (y/N): ").strip().lower()
            )
            if resp not in ("y", "yes"):
                print_stdout("Aborted.")
                return

        # Ensure rows exist; append to label_hist
        required_cols = ["bio_type", "alphabet"]
        if cfg.safety.write_back_requires_columns_present:
            if cfg.data.x_column_name not in df.columns:
                raise OpalError(f"records.parquet missing required X column '{cfg.data.x_column_name}'.")
            required_cols.append(cfg.data.x_column_name)

        # Optional: preview duplicates at this round for better UX
        try:
            lh = store.label_hist_col()
            ids_in = set(labels_df["id"].dropna().astype(str))
            maybe = df.loc[df["id"].astype(str).isin(ids_in), ["id", lh]]
            dup = 0
            for _, cell in maybe[lh].items():
                for e in store._normalize_hist_cell(cell):
                    if int(e.get("r", -1)) == int(round):
                        dup += 1
                        break
            if dup > 0:
                print_stdout(
                    f"[notice] {dup}/{len(ids_in)} incoming labels already have r={int(round)};"
                    f"applying --if-exists={if_exists}."
                )  # noqa
        except Exception:
            pass

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

            # 1) append to immutable label history (SSoT)
            df2 = store.append_labels_from_df(
                df,
                labels_df[["id", "y"]],  # ids are now concrete
                r=int(round),
                src="ingest_y",
                fail_if_any_existing_labels=(str(if_exists).lower().strip() == "fail"),
                if_exists=str(if_exists).lower().strip(),
            )
            # 2) mirror "current y" into configured y_column_name for convenience
            df3 = store.upsert_current_y_column(df2, labels_df[["id", "y"]], cfg.data.y_column_name)
            store.save_atomic(df3)

            # Emit label events (canonical SSoT)
            seq_map = df2.set_index("id")["sequence"].to_dict() if "sequence" in df2.columns else {}
            events = build_label_events(
                ids=labels_df["id"].astype(str).tolist(),
                sequences=[seq_map.get(str(_id)) for _id in labels_df["id"].astype(str).tolist()],
                y_obs=labels_df["y"].tolist(),
                observed_round=int(round),
                src="ingest_y",
                note=None,
            )
            append_events(events_path(Path(cfg.campaign.workdir)), events)

        out = {
            "ok": True,
            "round": int(round),
            "labels_appended": int(len(labels_df)),
            "y_column_updated": cfg.data.y_column_name,
        }

        if json:
            json_out(out)
        else:
            print_stdout(
                render_ingest_commit_human(
                    round_index=out["round"],
                    labels_appended=out["labels_appended"],
                    y_column_updated=out["y_column_updated"],
                )
            )
    except OpalError as e:
        opal_error("run", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("ingest-y", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
