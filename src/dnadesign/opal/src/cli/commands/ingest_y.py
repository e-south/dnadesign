"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/ingest_y.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from ...config import load_config
from ...ingest import run_ingest
from ...utils import ExitCodes, OpalError, print_stdout
from ..registry import cli_command
from ._common import internal_error, json_out, resolve_config_path, store_from_cfg


@cli_command(
    "ingest-y",
    help=(
        "Ingest tidy CSV → Y (preview + confirmation; strict checks). "
        "If the CSV lacks an 'id' column, OPAL resolves rows by 'sequence'. "
        "IDs are never inferred from other columns."
    ),
)
def cmd_ingest_y(
    config: Path = typer.Option(
        None, "--config", "-c", envvar="OPAL_CONFIG", help="campaign.yaml"
    ),
    round: int = typer.Option(
        ..., "--round", "-r", help="Round index to append to label history"
    ),
    csv: Path = typer.Option(..., "--csv", help="Tidy CSV/Parquet with raw reads"),
    transform: str = typer.Option(
        None, "--transform", help="Override YAML transform name (optional)"
    ),
    params: Path = typer.Option(
        None, "--params", help="JSON file with transform params (optional)"
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip interactive prompt (non-TTY safe)"
    ),
    allow_missing_x: bool = typer.Option(
        False,
        "--allow-missing-x",
        help="Proceed even if some labels would be written to rows without X.",
    ),
    no_create: bool = typer.Option(
        False,
        "--no-create",
        help="Refuse to create new rows for unknown sequences.",
    ),
):
    """
    Behavior:
      • Uses transform from YAML (overridable via flags).
      • Prints a preview (counts + sample) before any write.
      • If CSV has no 'id', OPAL resolves by 'sequence':
          - If sequence exists → use that row's id.
          - If already labeled in this campaign → fail fast.
          - If not present → add a new row with essentials and a generated id.
      • History is immutable: per-(id, round) conflicts cause an error.
    """
    try:
        cfg = load_config(resolve_config_path(config))
        store = store_from_cfg(cfg)
        df = store.load()

        # Resolve CSV path with helpful fallbacks
        def _resolve_csv(p: Path) -> Path:
            if p.is_absolute() and p.exists():
                return p
            # try CWD
            q = Path.cwd() / p
            if q.exists():
                return q
            # try workdir
            wd = Path(cfg.campaign.workdir)
            q = wd / p
            if q.exists():
                return q
            # try common input locations
            candidates = [
                wd / "inputs" / p.name,
                wd / "inputs" / f"r{round}" / p.name,
                wd / "inputs" / f"r{round}" / str(p),
            ]
            for c in candidates:
                if c.exists():
                    return c
            # last resort: search by basename under inputs/
            matches = list((wd / "inputs").rglob(p.name))
            if len(matches) == 1:
                return matches[0]
            hint = (
                ("\n  suggestions:\n    - " + "\n    - ".join(map(str, matches[:10])))
                if matches
                else ""
            )
            raise OpalError(
                f"CSV not found: {p}.\n  cwd={Path.cwd()}\n  workdir={wd}{hint}"
            )

        csv_path = _resolve_csv(Path(csv))
        # Load with format sniffing
        csv_df = (
            pd.read_parquet(csv_path)
            if csv_path.suffix.lower() in (".pq", ".parquet")
            else pd.read_csv(csv_path)
        )

        # choose effective transform name & params
        t_name = (transform or cfg.data.transforms_y.name).strip()
        t_params = cfg.data.transforms_y.params
        if params:
            import json as _json

            t_params = _json.loads(Path(params).read_text())
            typer.echo("[WARN] Transform params overridden via --params", err=True)
        if t_name != cfg.data.transforms_y.name:
            typer.echo(
                f"[WARN] CLI transform '{t_name}' != YAML '{cfg.data.transforms_y.name}'",
                err=True,
            )

        labels_df, preview = run_ingest(
            df,
            csv_df,
            transform_name=t_name,
            transform_params=t_params,
            y_expected_length=cfg.data.y_expected_length,
            setpoint_vector=cfg.objective.objective.params.get(
                "setpoint_vector", [0, 0, 0, 1]
            ),
            y_column_name=cfg.data.y_column_name,
        )

        # STRICT: Y must be correct length and all finite (no NaN/Inf).
        exp_len = cfg.data.y_expected_length
        if exp_len:
            import numpy as _np

            def _vec_ok(v) -> bool:
                try:
                    arr = _np.asarray(v, dtype=float).ravel()
                except Exception:
                    return False
                return (arr.size == exp_len) and _np.all(_np.isfinite(arr))

            bad_mask = ~labels_df["y"].map(_vec_ok)
            if bool(bad_mask.any()):
                bad_rows = labels_df.loc[bad_mask].head(5).to_dict(orient="records")
                raise OpalError(
                    f"ingest-y: Y must be length {exp_len} with finite values only; "
                    f"invalid rows={int(bad_mask.sum())}. Sample: {bad_rows}"
                )

        # -------- PREWRITE FORECAST --------
        xcol = cfg.data.x_column_name
        have_id_in_labels = "id" in labels_df.columns
        have_seq_in_labels = "sequence" in labels_df.columns
        id_map = set(df["id"].astype(str)) if "id" in df.columns else set()
        seq_to_id = (
            df.dropna(subset=["sequence"])[["sequence", "id"]]
            .astype({"sequence": str, "id": str})
            .drop_duplicates()
            .set_index("sequence")["id"]
            .to_dict()
            if "sequence" in df.columns and "id" in df.columns
            else {}
        )

        ids_in_labels = set(labels_df["id"].astype(str)) if have_id_in_labels else set()
        seqs_in_labels = (
            set(labels_df["sequence"].astype(str)) if have_seq_in_labels else set()
        )

        resolve_by_id = len([_id for _id in ids_in_labels if _id in id_map])
        resolve_by_seq = len([s for s in seqs_in_labels if s in seq_to_id])
        new_seqs = [s for s in seqs_in_labels if s not in seq_to_id]

        will_create = len(new_seqs) if not have_id_in_labels else 0
        # Predict missing-X: newly created rows won't have X unless your CSV provides it (we don't write X here).
        missing_x_forecast = 0
        if will_create > 0:
            if xcol in csv_df.columns:
                # If CSV already carries X for those sequences, they won't be missing; count those that still lack it.
                csv_seq_with_x = (
                    set(
                        csv_df.loc[csv_df[xcol].notna(), "sequence"]
                        .astype(str)
                        .tolist()
                    )
                    if "sequence" in csv_df.columns
                    else set()
                )
                missing_x_forecast = len(
                    [s for s in new_seqs if s not in csv_seq_with_x]
                )
            else:
                missing_x_forecast = will_create

        # COMPACT SUMMARY (prewrite)
        summary = {
            "source_csv": str(csv_path),
            "y_column": cfg.data.y_column_name,
            "y_expected_length": cfg.data.y_expected_length,
            "rows_in_csv": int(len(csv_df)),
            "labels_to_write": int(len(labels_df)),
            "resolve_by_id": int(resolve_by_id),
            "resolve_by_sequence": int(resolve_by_seq),
            "new_records_to_create": int(will_create),
            "x_column_name": xcol,
            "would_be_missing_x": int(missing_x_forecast),
        }
        json_out(
            {"preview": summary, "sample": labels_df.head(5).to_dict(orient="records")}
        )

        # Policy guards
        if no_create and will_create > 0:
            raise OpalError(
                f"--no-create set but {will_create} new records would be created from unknown sequences."
            )
        if missing_x_forecast > 0 and not allow_missing_x:
            # Force interactive confirmation unless --yes AND --allow-missing-x
            if yes:
                raise OpalError(
                    f"{missing_x_forecast} labels would be written to rows without X '{xcol}'. "
                    "Re-run with --allow-missing-x to proceed, or add X first."
                )
            resp2 = (
                input(
                    f"HARD WARNING: {missing_x_forecast} labels would be written to rows missing X '{xcol}'.\n"
                    "Type 'proceed' to continue, or press Enter to abort: "
                )
                .strip()
                .lower()
            )
            if resp2 != "proceed":
                print_stdout("Aborted (missing X).")
                return

        # Prompt unless --yes
        if not yes:
            resp = (
                input(f"Proceed to write labels for round {round}? (y/N): ")
                .strip()
                .lower()
            )
            if resp not in ("y", "yes"):
                print_stdout("Aborted by user.")
                return

        # Ensure rows exist / resolve IDs (by sequence when id is absent)
        df_before = df.copy()
        df = store.ensure_rows_exist(df, labels_df, csv_df)

        # Figure out how many were pre-existing vs created now
        ids_before = (
            set(df_before["id"].astype(str)) if "id" in df_before.columns else set()
        )
        label_ids = set(labels_df["id"].astype(str))
        new_ids = label_ids - ids_before
        existing_ids = label_ids & ids_before

        # Append labels and save atomically (fail if the sample already has ANY labels)
        df2 = store.append_labels_from_df(
            df,
            labels_df[["id", "y"]],
            r=round,
            fail_if_any_existing_labels=True,
        )
        store.save_atomic(df2)

        # Post-write check: any labeled rows missing X? (should match forecast)
        missing_x_count = 0
        missing_x_ids: list[str] = []
        if xcol in df2.columns:
            mask_labeled = df2["id"].astype(str).isin(label_ids)
            miss = df2[mask_labeled][xcol].isna()
            if hasattr(miss, "to_numpy"):
                missing_x_count = int(miss.sum())
                if missing_x_count:
                    missing_x_ids = (
                        df2.loc[mask_labeled & miss, "id"].astype(str).head(10).tolist()
                    )
        else:
            # If X column itself is missing, future steps will fail; flag loudly.
            missing_x_count = len(labels_df)
            missing_x_ids = list(map(str, list(label_ids)[:10]))

        json_out(
            {
                "ok": True,
                "round": int(round),
                "write": {
                    "labels_ingested": int(len(labels_df)),
                    "y_column_written": cfg.data.y_column_name,
                    "resolved_existing_ids": len(existing_ids),
                    "new_sequences_added": len(new_ids),
                },
                "post_checks": {
                    "missing_x_count": missing_x_count,
                    "missing_x_ids_sample": missing_x_ids,
                    "hard_warning": missing_x_count > 0,
                },
            }
        )
    except OpalError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("ingest-y", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
