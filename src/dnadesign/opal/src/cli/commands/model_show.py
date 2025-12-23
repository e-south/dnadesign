"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/model_show.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from ...models.random_forest import RandomForestModel
from ...state import CampaignState
from ...utils import ExitCodes, OpalError, ensure_dir, print_stdout
from ..formatting import render_model_show_human
from ..registry import cli_command
from ._common import internal_error, json_out, load_cli_config, opal_error


@cli_command(
    "model-show",
    help="Inspect a saved model; optionally dump full feature importances.",
)
def cmd_model_show(
    model_path: Optional[Path] = typer.Option(None, "--model-path"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: Optional[int] = typer.Option(None, "--round", "-r", help="Round index; default = latest"),
    json: bool = typer.Option(False, "--json/--human", help="Output format (default: human)"),
):
    try:
        # Resolve model_path if not provided
        if model_path is None:
            if not config:
                raise OpalError("Provide --model-path or --config to auto-resolve from state.json.")
            cfg = load_cli_config(config)
            st_path = Path(cfg.campaign.workdir) / "state.json"
            st = CampaignState.load(st_path)
            rounds = sorted(st.rounds, key=lambda r: int(r.round_index))
            if not rounds:
                raise OpalError(f"No rounds found in {st_path}")
            entry = (
                next((r for r in rounds if int(r.round_index) == int(round)), None) if round is not None else rounds[-1]
            )
            if entry is None:
                raise OpalError(f"Round {round} not found in {st_path}")
            # Prefer recorded artifact path; fallback to conventional path
            mp = Path(entry.model.get("artifact_path", "")) if entry.model else None
            if not mp or not mp.exists():
                mp = Path(entry.round_dir) / "model.joblib"
            model_path = mp
        mdl = RandomForestModel.load(str(model_path))
        info = {"model_type": "random_forest", "params": mdl.get_params()}
        if out_dir:
            ensure_dir(out_dir)
            imps = mdl.feature_importances()
            if imps is not None:
                fi = pd.DataFrame({"feature_index": np.arange(len(imps)), "feature_importance": imps})
                fi["feature_rank"] = fi["feature_importance"].rank(ascending=False, method="min").astype(int)
                fi.sort_values("feature_rank").to_csv(out_dir / "feature_importance_full.csv", index=False)
                info["feature_importance_top20"] = fi.nlargest(20, "feature_importance").to_dict(orient="records")
        if json:
            json_out(info)
        else:
            print_stdout(render_model_show_human(info))
    except OpalError as e:
        opal_error("run", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("model-show", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
