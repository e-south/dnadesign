"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/label_hist.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer

from ...core.utils import ExitCodes, OpalError, print_stdout
from ..formatting import kv_block
from ..registry import cli_command
from ._common import (
    internal_error,
    json_out,
    load_cli_config,
    opal_error,
    print_config_context,
    resolve_config_path,
    store_from_cfg,
)


@cli_command(
    "label-hist",
    help="Validate or repair the label_hist column (explicit, no silent fixes).",
)
def cmd_label_hist(
    action: str = typer.Argument(..., help="Action: validate | repair | attach-from-y"),
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    apply: bool = typer.Option(False, "--apply", help="Apply changes (default: dry-run)."),
    round: Optional[int] = typer.Option(None, "--round", "-r", help="Round stamp for attach-from-y."),
    src: str = typer.Option("manual_attach", "--src", help="label_hist src tag for attach-from-y."),
    json: bool = typer.Option(False, "--json/--human", help="Output format (default: human)."),
) -> None:
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        store = store_from_cfg(cfg)
        df = store.load()
        if not json:
            print_config_context(cfg_path, cfg=cfg, records_path=store.records_path)

        action = str(action).strip().lower()
        if action in ("validate", "check"):
            store.validate_label_hist(df, require=True)
            out = {"ok": True, "action": "validate"}
            if json:
                json_out(out)
            else:
                print_stdout(kv_block("label-hist", out))
            return
        if action in ("attach-from-y", "attach_from_y", "attach"):
            if round is None:
                raise OpalError("--round is required for attach-from-y.")
            y_col = cfg.data.y_column_name
            if y_col not in df.columns:
                raise OpalError(f"Expected Y column '{y_col}' not found in records.parquet.")
            lh = store.label_hist_col()
            if lh not in df.columns:
                df[lh] = None

            to_attach = []
            bad_ids = []
            for _id, y, hist in df[["id", y_col, lh]].itertuples(index=False, name=None):
                if y is None or (isinstance(y, float) and np.isnan(y)):
                    continue
                hist_norm = store._normalize_hist_cell(hist)
                if len(hist_norm) > 0:
                    continue
                try:
                    vec = np.asarray(y, dtype=float).ravel().tolist()
                    if not np.all(np.isfinite(np.asarray(vec, dtype=float))):
                        raise ValueError("non-finite")
                except Exception:
                    bad_ids.append(str(_id))
                    continue
                to_attach.append((str(_id), vec))

            if bad_ids:
                raise OpalError(f"attach-from-y failed: non-finite or invalid y values for ids (sample={bad_ids[:5]}).")

            out = {
                "ok": True,
                "action": "attach-from-y",
                "round": int(round),
                "candidates": int(len(to_attach)),
                "applied": bool(apply),
                "attached": int(len(to_attach) if apply else 0),
                "src": str(src),
            }
            if apply and to_attach:
                import pandas as pd

                labels_df = pd.DataFrame({"id": [t[0] for t in to_attach], "y": [t[1] for t in to_attach]})
                df2 = store.append_labels_from_df(
                    df,
                    labels_df,
                    r=int(round),
                    src=str(src),
                    fail_if_any_existing_labels=True,
                    if_exists="fail",
                )
                store.save_atomic(df2)
            if json:
                json_out(out)
            else:
                print_stdout(kv_block("label-hist", out))
            return
        if action != "repair":
            raise OpalError("Unknown action. Use 'validate', 'repair', or 'attach-from-y'.")

        cleaned, report = store.repair_label_hist(df)
        out = {
            "ok": True,
            "action": "repair",
            "rows_changed": report.get("rows_changed", 0),
            "entries_dropped": report.get("entries_dropped", 0),
            "applied": bool(apply),
        }
        if apply:
            store.save_atomic(cleaned)
        if json:
            json_out(out)
        else:
            print_stdout(kv_block("label-hist", out))
    except OpalError as e:
        opal_error("label-hist", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("label-hist", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
