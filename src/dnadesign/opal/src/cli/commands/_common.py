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

    CAND_NAMES = tuple(
        (
            os.getenv("OPAL_CONFIG_NAMES")
            or "campaign.yaml,campaign.yml,opal.yaml,opal.yml"
        ).split(",")
    )
    if opt:
        p = Path(opt).expanduser()
        p = p if p.is_absolute() else (Path.cwd() / p)
        p = p.resolve()
        if not p.exists():
            raise OpalError(
                f"Config path not found: {p}. "
                "Tip: from a campaign folder run `opal <cmd>` or pass `-c campaign.yaml`.",
                ExitCodes.BAD_ARGS,
            )
        return p

    env = os.getenv("OPAL_CONFIG")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p.resolve()

    cur = Path.cwd()
    # 2a) Prefer marker: .opal/config
    for base in (cur, *cur.parents):
        marker = base / ".opal" / "config"
        if marker.exists():
            txt = marker.read_text().strip()
            p = Path(txt)
            if not p.is_absolute():
                p = (marker.parent / p).resolve()
            if p.exists():
                return p
    # 2b) Otherwise look for common YAML names, nearest first
    for base in (cur, *cur.parents):
        for name in CAND_NAMES:
            cand = base / name
            if cand.exists():
                return cand.resolve()

    root = Path("src/dnadesign/opal/campaigns")
    if root.exists():
        found = list(root.glob("*/campaign.yaml"))
        if len(found) == 1:
            return found[0].resolve()

    # 3) Fallback: unique campaign under repo campaigns/ (resolve relative to this file)
    try:
        pkg_root = Path(__file__).resolve().parents[4]
    except Exception:
        pkg_root = Path.cwd()
    root = pkg_root / "src" / "dnadesign" / "opal" / "campaigns"
    if root.exists():
        found = []
        for name in CAND_NAMES:
            found.extend(root.glob(f"*/{name}"))
        if len(found) == 1:
            return found[0].resolve()

    raise OpalError(
        "No campaign config found. Use --config, set $OPAL_CONFIG, or run inside a campaign folder.",
        ExitCodes.BAD_ARGS,
    )


def store_from_cfg(cfg: RootConfig) -> RecordsStore:
    loc = cfg.data.location
    if isinstance(loc, LocationUSR):
        records = Path(loc.path) / loc.dataset / "records.parquet"
        data_location = {
            "kind": "usr",
            "dataset": loc.dataset,
            "path": str(Path(loc.path).resolve()),
            "records_path": str(records.resolve()),
        }
    elif isinstance(loc, LocationLocal):
        records = Path(loc.path)
        data_location = {
            "kind": "local",
            "path": str(Path(loc.path).resolve()),
            "records_path": str(records.resolve()),
        }
    else:
        raise OpalError("Unknown data location kind.", ExitCodes.BAD_ARGS)

    return RecordsStore(
        kind=data_location["kind"],
        records_path=records,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params=cfg.data.transforms_x.params,
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
