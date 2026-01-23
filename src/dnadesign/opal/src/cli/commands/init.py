"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/init.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from ...core.utils import ExitCodes, OpalError, ensure_dir, file_sha256, print_stdout
from ...storage.state import CampaignState
from ..formatting import render_init_human
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


@cli_command("init", help="Initialize/validate the campaign workspace; write state.json.")
def cmd_init(
    config: Path = typer.Option(None, "--config", "-c", help="Path to campaign.yaml", envvar="OPAL_CONFIG"),
    json: bool = typer.Option(False, "--json/--human", help="Output format (default: human)"),
):
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        if not json:
            print_config_context(cfg_path, cfg=cfg)
        workdir = Path(cfg.campaign.workdir)
        ensure_dir(workdir / "outputs")
        ensure_dir(workdir / "inputs")

        # Write a workspace marker so future commands can resolve fast from any child
        marker_dir = workdir / ".opal"
        ensure_dir(marker_dir)
        # store the path relative to the marker for portability
        rel = cfg_path if cfg_path.is_absolute() else cfg_path.resolve()
        try:
            rel = rel.relative_to(marker_dir)
        except Exception:
            try:
                rel = rel.relative_to(workdir)
            except Exception:
                pass
        (marker_dir / "config").write_text(str(rel))

        store = store_from_cfg(cfg)
        df = store.load()
        store.assert_unique_ids(df)
        if cfg.safety.require_biotype_and_alphabet_on_init:
            missing = [c for c in ("bio_type", "alphabet") if c not in df.columns]
            if missing:
                raise OpalError(f"records.parquet missing required columns: {missing}")
            for col in ("bio_type", "alphabet"):
                if df[col].isna().any():
                    bad = df.loc[df[col].isna(), "id"].astype(str).tolist()[:10]
                    raise OpalError(f"Missing values in '{col}' (sample ids={bad}).")
        df2, added = store.ensure_label_hist_column(df)
        if added:
            store.save_atomic(df2)
        st = CampaignState(
            campaign_slug=cfg.campaign.slug,
            campaign_name=cfg.campaign.name,
            workdir=str(workdir.resolve()),
            data_location={
                "kind": store.kind,
                "records_path": str(store.records_path.resolve()),
                "records_sha256": (file_sha256(store.records_path) if store.records_path.exists() else ""),
            },
            x_column_name=cfg.data.x_column_name,
            y_column_name=cfg.data.y_column_name,
            representation_transform={
                "name": cfg.data.transforms_x.name,
                "params": cfg.data.transforms_x.params,
            },
            training_policy=cfg.training.policy,
            performance={
                "score_batch_size": cfg.scoring.score_batch_size,
                "objective": cfg.objective.objective.name,
            },
            representation_vector_dimension=0,
            backlog={"number_of_selected_but_not_yet_labeled_candidates_total": 0},
        )
        st.save(Path(cfg.campaign.workdir) / "state.json")
        out = {"ok": True, "workdir": str(workdir.resolve())}
        if json:
            json_out(out)
        else:
            print_stdout(render_init_human(workdir=Path(out["workdir"])))
    except OpalError as e:
        opal_error("init", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("init", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
