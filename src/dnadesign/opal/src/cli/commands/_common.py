"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/_common.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ...config import LocationLocal, LocationUSR, RootConfig
from ...data_access import RecordsStore
from ...utils import ExitCodes, OpalError, print_stderr, print_stdout


def resolve_config_path(opt: Optional[Path]) -> Path:
    if opt:
        return opt.resolve()

    env = os.getenv("OPAL_CONFIG")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p.resolve()

    cur = Path.cwd()
    for base in (cur, *cur.parents):
        cand = base / "campaign.yaml"
        if cand.exists():
            return cand.resolve()

    root = Path("src/dnadesign/opal/campaigns")
    if root.exists():
        found = list(root.glob("*/campaign.yaml"))
        if len(found) == 1:
            return found[0].resolve()

    raise OpalError(
        "No campaign.yaml found. Use --config, set $OPAL_CONFIG, or run from within a campaign folder.",
        ExitCodes.BAD_ARGS,
    )


def store_from_cfg(cfg: RootConfig) -> RecordsStore:
    loc = cfg.data.location
    if isinstance(loc, LocationUSR):
        records = Path(loc.usr_root) / loc.dataset / "records.parquet"
        data_location = {
            "kind": "usr",
            "dataset": loc.dataset,
            "usr_root": str(Path(loc.usr_root).resolve()),
            "records_path": str(records.resolve()),
        }
    elif isinstance(loc, LocationLocal):
        records = Path(loc.path)
        data_location = {"kind": "local", "records_path": str(records.resolve())}
    else:
        raise OpalError("Unknown data location kind.", ExitCodes.BAD_ARGS)

    return RecordsStore(
        kind=data_location["kind"],
        records_path=records,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.representation_column_name,
        y_col=cfg.data.label_source_column_name,
        rep_transform_name=cfg.data.representation_transform.name,
        rep_transform_params=cfg.data.representation_transform.params,
    )


def json_out(obj) -> None:
    print_stdout(json.dumps(obj, indent=2))


def internal_error(ctx: str, e: Exception) -> None:
    if str(os.getenv("OPAL_DEBUG", "")).strip().lower() in ("1", "true", "yes", "on"):
        import traceback

        tb = traceback.format_exc()
        print_stderr(f"Internal error during {ctx}: {e}\n{tb}")
    else:
        print_stderr(
            f"Internal error during {ctx}: {e}\n(Hint: set OPAL_DEBUG=1 for a full traceback)"
        )
