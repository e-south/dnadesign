"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/_common.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import dataclasses as _dc
import json
import os
from pathlib import Path
from pathlib import Path as _Path
from typing import Optional

import typer

from ...utils import ExitCodes, OpalError, print_stderr, print_stdout

try:
    import numpy as _np
except Exception:
    _np = None

from ...config import LocationLocal, LocationUSR, RootConfig
from ...data_access import RecordsStore


def resolve_config_path(opt: Optional[Path], *, allow_dir: bool = False) -> Path:
    CAND_NAMES = tuple((os.getenv("OPAL_CONFIG_NAMES") or "campaign.yaml,campaign.yml,opal.yaml,opal.yml").split(","))
    env = os.getenv("OPAL_CONFIG")
    env_path: Optional[Path] = None
    if env:
        env_path = Path(env).expanduser()
        env_path = env_path if env_path.is_absolute() else (Path.cwd() / env_path)
        env_path = env_path.resolve()

    if opt:
        p = Path(opt).expanduser()
        p = p if p.is_absolute() else (Path.cwd() / p)
        p = p.resolve()
        if not p.exists():
            if env_path is not None and env_path == p:
                raise OpalError(f"$OPAL_CONFIG points to a missing path: {p}", ExitCodes.BAD_ARGS)
            raise OpalError(
                f"Config path not found: {p}. Tip: from a campaign folder run `opal <cmd>` or pass `-c campaign.yaml`.",
                ExitCodes.BAD_ARGS,
            )
        if p.is_dir() and not allow_dir:
            if env_path is not None and env_path == p:
                raise OpalError(f"$OPAL_CONFIG points to a directory (expected campaign YAML): {p}", ExitCodes.BAD_ARGS)
            raise OpalError(
                f"Config path is a directory: {p}. Expected a campaign YAML (e.g., campaign.yaml).",
                ExitCodes.BAD_ARGS,
            )
        return p

    if env:
        p = env_path or Path(env).expanduser().resolve()
        if not p.exists():
            raise OpalError(f"$OPAL_CONFIG points to a missing path: {p}", ExitCodes.BAD_ARGS)
        if p.is_dir():
            raise OpalError(f"$OPAL_CONFIG points to a directory (expected campaign YAML): {p}", ExitCodes.BAD_ARGS)
        return p

    cur = Path.cwd()
    # 2a) Prefer marker: .opal/config
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
                raise OpalError(f"Marker points to a directory (expected campaign YAML): {p}", ExitCodes.BAD_ARGS)
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


def _format_validation_error(e, cfg_path: Path) -> str:
    """
    Turn a Pydantic ValidationError into an actionable, human-first message,
    keeping strict validation (no fallbacks).
    """
    try:
        from pydantic import ValidationError  # type: ignore
    except Exception:
        # If pydantic isn't present for some reason, fall back to repr
        return f"Invalid configuration in {cfg_path}:\n{e!r}"

    if not isinstance(e, ValidationError):
        return f"Invalid configuration in {cfg_path}:\n{e!r}"

    lines = [f"Config schema error: {cfg_path}"]
    errs = getattr(e, "errors", None)
    errs = errs() if callable(errs) else (errs or [])
    for err in errs:
        loc = err.get("loc", [])
        loc_str = ".".join(str(x) for x in loc) if loc else "(root)"
        typ = err.get("type", "")
        msg = err.get("msg", "")
        lines.append(f"  - at: {loc_str}")
        lines.append(f"    type: {typ}")
        lines.append(f"    detail: {msg}")

        # Assertive, plugin-agnostic hint for unknown model params
        if typ == "extra_forbidden" and loc_str.startswith("model.params."):
            lines.append(
                "    hint: Unknown model parameter. Check the configured model plugin schema "
                "(see config/plugin_schemas.py or the model's README) and remove or rename this key."
            )

    return "\n".join(lines)


def load_cli_config(config_opt: Optional[Path]) -> RootConfig:
    """
    Strict config loader for CLI commands. Converts Pydantic ValidationError into
    an OpalError with a friendly, path-aware message.
    """
    cfg_path = resolve_config_path(config_opt)
    try:
        from ...config import load_config as _load_config

        return _load_config(cfg_path)
    except Exception as e:
        # Prefer a precise message if it's a Pydantic validation error
        try:
            from pydantic import ValidationError  # type: ignore

            cause = getattr(e, "__cause__", None)
            ve = cause if isinstance(cause, ValidationError) else (e if isinstance(e, ValidationError) else None)
            if ve is not None:
                raise OpalError(_format_validation_error(ve, cfg_path), ExitCodes.BAD_ARGS)
        except Exception:
            pass
        # Otherwise, preserve the original exception semantics
        raise


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


def print_config_context(cfg_path: Path, *, cfg: RootConfig | None = None, records_path: Path | None = None) -> None:
    """
    Human output helper: echo resolved config + workdir (and records path if known).
    Avoid in JSON output to keep streams machine-readable.
    """
    parts = [f"Config: {Path(cfg_path).resolve()}"]
    if cfg is not None:
        parts.append(f"Workdir: {Path(cfg.campaign.workdir).resolve()}")
    if records_path is not None:
        parts.append(f"Records: {Path(records_path).resolve()}")
    print_stdout(" | ".join(parts))


def _json_default(o):
    if _dc.is_dataclass(o):
        return _dc.asdict(o)
    if isinstance(o, _Path):
        return str(o)
    if _np is not None:
        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, (_np.ndarray,)):
            return o.tolist()
    return str(o)


def json_out(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, default=_json_default))


def internal_error(ctx: str, e: Exception) -> None:
    if str(os.getenv("OPAL_DEBUG", "")).strip().lower() in ("1", "true", "yes", "on"):
        import traceback as _tb

        tb = _tb.format_exc()
        print_stderr(f"[bold red]Internal error during [white]{ctx}[/]:[/] {e}\n[dim]{tb}[/]")
    else:
        print_stderr(
            f"[bold red]Internal error during [white]{ctx}[/]:[/] {e}\n[dim](Hint: set OPAL_DEBUG=1 for full traceback)[/]"  # noqa
        )


def opal_error(ctx: str, e: OpalError) -> None:
    """When OPAL_DEBUG=1, include a traceback for OpalError too."""
    if str(os.getenv("OPAL_DEBUG", "")).strip().lower() in ("1", "true", "yes", "on"):
        import traceback as _tb

        print_stderr(f"[bold red]OpalError during [white]{ctx}[/]:[/] {e}\n[dim]{_tb.format_exc()}[/]")
    else:
        print_stderr(str(e))
