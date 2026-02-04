# ABOUTME: Resolve OPAL campaign config paths from CLI/env/markers.
# ABOUTME: Supports configs/ layout and campaign root discovery.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/core/config_resolve.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from .utils import ExitCodes, OpalError


def _candidate_names() -> tuple[str, ...]:
    return tuple((os.getenv("OPAL_CONFIG_NAMES") or "campaign.yaml,campaign.yml,opal.yaml,opal.yml").split(","))


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


def _iter_candidate_paths(path: Path, *, names: Iterable[str]) -> Iterable[Path]:
    for name in names:
        yield path / name
    configs_dir = path / "configs"
    for name in names:
        yield configs_dir / name


def _find_campaign_yaml_in_dir(path: Path, *, names: Iterable[str]) -> Path:
    for cand in _iter_candidate_paths(path, names=names):
        if cand.exists():
            return cand.resolve()
    raise OpalError(f"No campaign YAML found in directory: {path}", ExitCodes.BAD_ARGS)


def resolve_campaign_config_path(opt: Optional[Path], *, allow_dir: bool = False) -> Path:
    """
    Resolve a campaign YAML path from explicit args, env, marker files, or cwd.
    """
    names = _candidate_names()
    env = os.getenv("OPAL_CONFIG")
    env_path: Optional[Path] = _resolve_path(env) if env else None

    if opt:
        p = _resolve_path(opt)
        if not p.exists():
            if env_path is not None and env_path == p:
                raise OpalError(f"$OPAL_CONFIG points to a missing path: {p}", ExitCodes.BAD_ARGS)
            raise OpalError(
                f"Config path not found: {p}. Tip: from a campaign folder run `opal <cmd>` or pass `-c campaign.yaml`.",
                ExitCodes.BAD_ARGS,
            )
        if p.is_dir():
            if not allow_dir:
                if env_path is not None and env_path == p:
                    msg = f"$OPAL_CONFIG points to a directory (expected campaign YAML): {p}"
                    raise OpalError(msg, ExitCodes.BAD_ARGS)
                raise OpalError(
                    f"Config path is a directory: {p}. Expected a campaign YAML (e.g., campaign.yaml).",
                    ExitCodes.BAD_ARGS,
                )
            return _find_campaign_yaml_in_dir(p, names=names)
        return p

    if env:
        p = env_path or _resolve_path(env)
        if not p.exists():
            raise OpalError(f"$OPAL_CONFIG points to a missing path: {p}", ExitCodes.BAD_ARGS)
        if p.is_dir():
            if allow_dir:
                return _find_campaign_yaml_in_dir(p, names=names)
            msg = f"$OPAL_CONFIG points to a directory (expected campaign YAML): {p}"
            raise OpalError(msg, ExitCodes.BAD_ARGS)
        return p

    cur = Path.cwd()
    # Prefer marker: .opal/config
    for base in (cur, *cur.parents):
        marker = base / ".opal" / "config"
        if marker.exists():
            txt = marker.read_text().strip()
            if not txt:
                raise OpalError(f"Marker file is empty: {marker}", ExitCodes.BAD_ARGS)
            p = Path(txt)
            if not p.is_absolute():
                # Marker paths are defined relative to the campaign workdir
                p = (marker.parent.parent / p).resolve()
            if not p.exists():
                raise OpalError(
                    f"Marker points to missing config: {p} (from {marker}).",
                    ExitCodes.BAD_ARGS,
                )
            if p.is_dir():
                raise OpalError(
                    f"Marker points to a directory (expected campaign YAML): {p}",
                    ExitCodes.BAD_ARGS,
                )
            return p
    # Otherwise look for common YAML names, nearest first
    for base in (cur, *cur.parents):
        for cand in _iter_candidate_paths(base, names=names):
            if cand.exists():
                return cand.resolve()

    root = Path("src/dnadesign/opal/campaigns")
    if root.exists():
        found = []
        for name in names:
            found.extend(root.glob(f"*/{name}"))
            found.extend(root.glob(f"*/configs/{name}"))
        if len(found) == 1:
            return found[0].resolve()

    # Fallback: unique campaign under repo campaigns/ (resolve relative to this file)
    try:
        pkg_root = Path(__file__).resolve().parents[4]
    except Exception:
        pkg_root = Path.cwd()
    root = pkg_root / "src" / "dnadesign" / "opal" / "campaigns"
    if root.exists():
        found = []
        for name in names:
            found.extend(root.glob(f"*/{name}"))
            found.extend(root.glob(f"*/configs/{name}"))
        if len(found) == 1:
            return found[0].resolve()

    raise OpalError(
        "No campaign config found. Use --config, set $OPAL_CONFIG, or run inside a campaign folder.",
        ExitCodes.BAD_ARGS,
    )
