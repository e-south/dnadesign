# ABOUTME: CLI command to inspect stored model artifacts and parameters.
# ABOUTME: Resolves model paths from state or explicit paths for reporting.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/model_show.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Optional

import typer

from ...core.rounds import resolve_round_index_from_state
from ...core.utils import ExitCodes, OpalError, ensure_dir, print_stdout
from ...registries.models import load_model
from ...storage.state import CampaignState
from ..formatting import render_model_show_human
from ..registry import cli_command
from ._common import (
    internal_error,
    json_out,
    load_cli_config,
    opal_error,
    print_config_context,
    resolve_config_path,
    resolve_json_path,
)


@cli_command(
    "model-show",
    help="Inspect a saved model; optionally dump full feature importances.",
)
def cmd_model_show(
    model_path: Optional[Path] = typer.Option(None, "--model-path"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: Optional[str] = typer.Option(None, "--round", "-r", help="Round selector: int or 'latest'."),
    model_name: Optional[str] = typer.Option(
        None,
        "--model-name",
        help="Explicit model registry name (required if model_meta.json is missing).",
    ),
    model_params: Optional[Path] = typer.Option(
        None,
        "--model-params",
        help="Optional JSON file (.json) with model params (used with --model-name).",
    ),
    json: bool = typer.Option(False, "--json/--human", help="Output format (default: human)"),
):
    try:
        import numpy as np
        import pandas as pd

        # Resolve model_path if not provided
        if model_path is None:
            if not config:
                raise OpalError("Provide --model-path or --config to auto-resolve from state.json.")
            cfg_path = resolve_config_path(config)
            cfg = load_cli_config(cfg_path)
            if not json:
                print_config_context(cfg_path, cfg=cfg)
            st_path = Path(cfg.campaign.workdir) / "state.json"
            st = CampaignState.load(st_path)
            rounds = sorted(st.rounds, key=lambda r: int(r.round_index))
            if not rounds:
                raise OpalError(f"No rounds found in {st_path}")
            round_sel = resolve_round_index_from_state(st_path, round)
            entry = next((r for r in rounds if int(r.round_index) == int(round_sel)), None)
            if entry is None:
                raise OpalError(f"Round {round_sel} not found in {st_path}")
            # Prefer recorded artifact path; fallback to conventional path
            mp = Path(entry.model.get("artifact_path", "")) if entry.model else None
            if not mp or not mp.exists():
                mp = Path(entry.round_dir) / "model" / "model.joblib"
            model_path = mp
        params_obj = None
        if model_params:
            model_params = resolve_json_path(model_params, label="--model-params", must_exist=True)
            params_obj = _json.loads(model_params.read_text())
            if not model_name:
                raise OpalError("Use --model-name with --model-params.")

        meta_path = Path(model_path).parent / "model_meta.json"
        if model_name is None:
            if not meta_path.exists():
                raise OpalError(
                    f"model_meta.json not found next to {model_path}. "
                    "Provide --model-name/--model-params or re-run a round."
                )
            meta = _json.loads(meta_path.read_text())
            model_name = meta.get("model__name")
            params_obj = meta.get("model__params")
            if not model_name:
                raise OpalError(f"model_meta.json missing model__name: {meta_path}")

        mdl = load_model(str(model_name), str(model_path), params=params_obj)
        if not hasattr(mdl, "get_params"):
            raise OpalError(f"Model '{model_name}' does not implement get_params().")
        info = {"model_type": str(model_name), "params": mdl.get_params()}
        if out_dir:
            ensure_dir(out_dir)
            if not hasattr(mdl, "feature_importances"):
                raise OpalError(f"Model '{model_name}' does not support feature_importances().")
            imps = mdl.feature_importances()
            if imps is None:
                raise OpalError(f"Model '{model_name}' returned no feature importances.")
            fi = pd.DataFrame({"feature_index": np.arange(len(imps)), "feature_importance": imps})
            fi["feature_rank"] = fi["feature_importance"].rank(ascending=False, method="min").astype(int)
            fi.sort_values("feature_rank").to_csv(out_dir / "feature_importance_full.csv", index=False)
            info["feature_importance_top20"] = fi.nlargest(20, "feature_importance").to_dict(orient="records")
        if json:
            json_out(info)
        else:
            print_stdout(render_model_show_human(info))
    except OpalError as e:
        opal_error("model-show", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("model-show", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
