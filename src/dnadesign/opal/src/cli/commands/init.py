"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/cli/commands/init.py

Initializes OPAL campaign workspaces and validates records layout. Writes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from ...core.utils import ExitCodes, OpalError, ensure_dir, print_stdout
from ...storage.state import CampaignState
from ..formatting import render_init_text
from ..guidance_hints import maybe_print_hints
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


def _records_metadata(records_path: Path) -> dict[str, object]:
    stat = records_path.stat()
    return {
        "records_size_bytes": int(stat.st_size),
        "records_mtime_ns": int(stat.st_mtime_ns),
        "records_fingerprint_kind": "file_metadata",
    }


@cli_command("init", help="Initialize the campaign workspace and write state.json.")
def cmd_init(
    config: Path = typer.Option(None, "--config", "-c", help="Path to campaign.yaml", envvar="OPAL_CONFIG"),
    no_hints: bool = typer.Option(False, "--no-hints", help="Disable next-step hints in text output."),
    json: bool = typer.Option(False, "--json/--text", help="Output format (default: text)"),
):
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        if not json:
            print_config_context(cfg_path, cfg=cfg)
        workdir = Path(cfg.campaign.workdir)
        outputs_dir = workdir / "outputs"
        ensure_dir(outputs_dir / "ledger")
        ensure_dir(outputs_dir / "rounds")

        store = store_from_cfg(cfg)
        schema_columns = store.schema_columns()
        scalar_columns = ["id"]
        if cfg.safety.require_biotype_and_alphabet_on_init:
            scalar_columns.extend(["bio_type", "alphabet"])
        df = store.load_columns(scalar_columns)
        store.assert_unique_ids(df)
        if cfg.safety.require_biotype_and_alphabet_on_init:
            missing = [c for c in ("bio_type", "alphabet") if c not in schema_columns]
            if missing:
                raise OpalError(f"records.parquet missing required columns: {missing}")
            for col in ("bio_type", "alphabet"):
                if df[col].isna().any():
                    bad = df.loc[df[col].isna(), "id"].astype(str).tolist()[:10]
                    raise OpalError(f"Missing values in '{col}' (sample ids={bad}).")
        if cfg.writeback.prediction_records == "label_history" and store.label_hist_col() not in schema_columns:
            store.append_null_column_atomic(store.label_hist_col())
        data_location = {
            "kind": store.kind,
            "records_path": str(store.records_path.resolve()),
            **_records_metadata(store.records_path),
        }
        st = CampaignState(
            campaign_slug=cfg.campaign.slug,
            campaign_name=cfg.campaign.name,
            workdir=str(workdir.resolve()),
            data_location=data_location,
            x_column_name=cfg.data.x_column_name,
            y_column_name=cfg.data.y_column_name,
            representation_transform={
                "name": cfg.data.transforms_x.name,
                "params": cfg.data.transforms_x.params,
            },
            training_policy=cfg.training.policy,
            performance={
                "score_batch_size": cfg.scoring.score_batch_size,
                "objectives": [o.name for o in cfg.objectives.objectives],
            },
            representation_vector_dimension=0,
            backlog={"number_of_selected_but_not_yet_labeled_candidates_total": 0},
        )
        st.save(Path(cfg.campaign.workdir) / "state.json")
        out = {"ok": True, "workdir": str(workdir.resolve())}
        if json:
            json_out(out)
        else:
            print_stdout(render_init_text(workdir=Path(out["workdir"])))
            maybe_print_hints(command_name="init", cfg_path=cfg_path, no_hints=no_hints, json_output=json)
    except OpalError as e:
        opal_error("init", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("init", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
