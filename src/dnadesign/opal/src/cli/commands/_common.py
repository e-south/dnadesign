# ABOUTME: Shared CLI helpers for OPAL commands (path resolution and output).
# ABOUTME: Centralizes validation for config, table, and json file inputs.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/_common.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import dataclasses as _dc
import json
import os
import sys
from pathlib import Path
from pathlib import Path as _Path
from typing import Optional

import typer

from ...core.utils import ExitCodes, OpalError, print_stderr, print_stdout

try:
    import numpy as _np
except Exception:
    _np = None

from ...config import RootConfig
from ...core.config_resolve import resolve_campaign_config_path
from ...storage.data_access import RecordsStore
from ...storage.store_factory import records_store_from_config


def prompt_confirm(prompt: str, *, non_interactive_hint: str) -> bool:
    """
    Prompt for a yes/no confirmation. Raises OpalError if stdin is not interactive.
    """
    if not sys.stdin.isatty():
        raise OpalError(non_interactive_hint, ExitCodes.BAD_ARGS)
    try:
        resp = input(prompt).strip().lower()
    except EOFError as e:
        raise OpalError(non_interactive_hint, ExitCodes.BAD_ARGS) from e
    return resp in ("y", "yes")


def resolve_config_path(opt: Optional[Path], *, allow_dir: bool = False) -> Path:
    return resolve_campaign_config_path(opt, allow_dir=allow_dir)


_TABLE_EXTS = {".csv", ".parquet", ".pq"}
_XLSX_EXTS = {".xlsx"}
_JSON_EXTS = {".json"}


def resolve_table_path(path: Path, *, label: str, must_exist: bool, allow_xlsx: bool = False) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    suffix = p.suffix.lower()
    exts = set(_TABLE_EXTS)
    if allow_xlsx:
        exts |= _XLSX_EXTS
    if suffix not in exts:
        if exts == _TABLE_EXTS:
            msg = f"{label} must be a CSV or Parquet file with extension {', '.join(sorted(_TABLE_EXTS))}; got '{p}'."
        else:
            msg = f"{label} must be a table file with extension {', '.join(sorted(exts))}; got '{p}'."
        raise OpalError(msg)
    if must_exist:
        if not p.exists():
            raise OpalError(f"{label} not found: {p}")
        if p.is_dir():
            raise OpalError(f"{label} must be a file, got directory: {p}")
    else:
        if p.exists() and p.is_dir():
            raise OpalError(f"{label} must be a file, got directory: {p}")
        if not p.parent.exists():
            raise OpalError(f"{label} parent directory does not exist: {p.parent}")
    return p


def resolve_json_path(path: Path, *, label: str, must_exist: bool) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    suffix = p.suffix.lower()
    if suffix not in _JSON_EXTS:
        raise OpalError(f"{label} must be a JSON file with extension {', '.join(sorted(_JSON_EXTS))}; got '{p}'.")
    if must_exist:
        if not p.exists():
            raise OpalError(f"{label} not found: {p}")
        if p.is_dir():
            raise OpalError(f"{label} must be a file, got directory: {p}")
    else:
        if p.exists() and p.is_dir():
            raise OpalError(f"{label} must be a file, got directory: {p}")
        if not p.parent.exists():
            raise OpalError(f"{label} parent directory does not exist: {p.parent}")
    return p


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
    return records_store_from_config(cfg)


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
