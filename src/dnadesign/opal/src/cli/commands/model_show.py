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

import numpy as np
import pandas as pd
import typer

from ...models.random_forest import RandomForestModel
from ...utils import ExitCodes, OpalError, ensure_dir
from ..registry import cli_command
from ._common import internal_error, json_out


@cli_command(
    "model-show",
    help="Inspect a saved model; optionally dump full feature importances.",
)
def cmd_model_show(
    model_path: Path = typer.Option(..., "--model-path"),
    out_dir: Path = typer.Option(None, "--out-dir"),
):
    try:
        mdl = RandomForestModel.load(str(model_path))
        info = {"model_type": "random_forest", "params": mdl.get_params()}
        if out_dir:
            ensure_dir(out_dir)
            imps = mdl.feature_importances()
            if imps is not None:
                fi = pd.DataFrame(
                    {"feature_index": np.arange(len(imps)), "feature_importance": imps}
                )
                fi["feature_rank"] = (
                    fi["feature_importance"]
                    .rank(ascending=False, method="min")
                    .astype(int)
                )
                fi.sort_values("feature_rank").to_csv(
                    out_dir / "feature_importance_full.csv", index=False
                )
                info["feature_importance_top20"] = fi.nlargest(
                    20, "feature_importance"
                ).to_dict(orient="records")
        json_out(info)
    except OpalError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("model-show", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
