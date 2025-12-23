"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/init.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from ...config import load_config
from ...state import CampaignState
from ...utils import ExitCodes, OpalError, ensure_dir, file_sha256, print_stdout
from ..formatting import render_init_human
from ..registry import cli_command
from ._common import (
    internal_error,
    json_out,
    opal_error,
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
        cfg = load_config(cfg_path)
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
        opal_error("run", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("init", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
