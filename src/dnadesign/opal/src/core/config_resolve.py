"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/core/config_resolve.py

Core runtime primitives for config resolve OPAL core.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .utils import ExitCodes, OpalError

_CAMPAIGN_CONFIG_RELATIVE_PATH = Path("configs/campaign.yaml")


def _resolve_path(value: Path | str) -> Path:
    p = Path(value).expanduser()
    p = p if p.is_absolute() else (Path.cwd() / p)
    return p.resolve()


def resolve_campaign_root(cfg_path: Path) -> Path:
    """
    Resolve the campaign root directory from a config file path.
    """
    p = Path(cfg_path).resolve()
    if p.is_dir():
        return p
    if p.parent.name == "configs":
        return p.parent.parent
    return p.parent


def _find_campaign_yaml_in_dir(path: Path) -> Path:
    candidate = path / _CAMPAIGN_CONFIG_RELATIVE_PATH
    if candidate.is_file():
        return candidate.resolve()
    raise OpalError(
        f"Campaign directory must contain {_CAMPAIGN_CONFIG_RELATIVE_PATH}: {path}",
        ExitCodes.BAD_ARGS,
    )


def resolve_campaign_config_path(opt: Optional[Path], *, allow_dir: bool = False) -> Path:
    """
    Resolve a campaign YAML path from explicit args or OPAL_CONFIG.
    """
    env = os.getenv("OPAL_CONFIG")
    env_path: Optional[Path] = _resolve_path(env) if env else None

    if opt:
        p = _resolve_path(opt)
        if not p.exists():
            if env_path is not None and env_path == p:
                raise OpalError(f"$OPAL_CONFIG points to a missing path: {p}", ExitCodes.BAD_ARGS)
            raise OpalError(
                f"Config path not found: {p}. Pass `-c configs/campaign.yaml` from the campaign directory.",
                ExitCodes.BAD_ARGS,
            )
        if p.is_dir():
            if not allow_dir:
                if env_path is not None and env_path == p:
                    msg = f"$OPAL_CONFIG points to a directory (expected campaign YAML): {p}"
                    raise OpalError(msg, ExitCodes.BAD_ARGS)
                raise OpalError(
                    f"Config path is a directory: {p}. Expected configs/campaign.yaml.",
                    ExitCodes.BAD_ARGS,
                )
            return _find_campaign_yaml_in_dir(p)
        return p

    if env:
        p = env_path or _resolve_path(env)
        if not p.exists():
            raise OpalError(f"$OPAL_CONFIG points to a missing path: {p}", ExitCodes.BAD_ARGS)
        if p.is_dir():
            if allow_dir:
                return _find_campaign_yaml_in_dir(p)
            msg = f"$OPAL_CONFIG points to a directory (expected campaign YAML): {p}"
            raise OpalError(msg, ExitCodes.BAD_ARGS)
        return p

    raise OpalError(
        "No config provided. Pass --config <campaign.yaml> or set $OPAL_CONFIG.",
        ExitCodes.BAD_ARGS,
    )
