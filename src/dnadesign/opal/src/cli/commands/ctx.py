# ABOUTME: CLI commands for inspecting round_ctx runtime carriers.
# ABOUTME: Loads and filters round_ctx snapshots from round metadata.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/ctx.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from ...core.rounds import resolve_round_index_from_state
from ...core.utils import ExitCodes, OpalError, print_stdout
from ...storage.workspace import CampaignWorkspace
from ..registry import cli_group
from ._common import (
    internal_error,
    json_out,
    load_cli_config,
    opal_error,
    print_config_context,
    resolve_config_path,
)

ctx_app = typer.Typer(no_args_is_help=True, help="Inspect runtime carriers (round_ctx.json).")


def _load_round_ctx(ws: CampaignWorkspace, round_sel: Optional[str]) -> Dict[str, Any]:
    r = resolve_round_index_from_state(ws.state_path, round_sel)
    path = ws.round_metadata_dir(r) / "round_ctx.json"
    if not path.exists():
        raise OpalError(f"round_ctx.json not found for round {r} at {path}")
    return json.loads(path.read_text())


def _filter_keys(snapshot: Dict[str, Any], prefixes: Optional[List[str]]) -> Dict[str, Any]:
    if not prefixes:
        return dict(snapshot)
    keep = []
    for p in prefixes:
        p = str(p).strip()
        if p:
            keep.append(p)
    if not keep:
        return dict(snapshot)
    return {k: v for k, v in snapshot.items() if any(k.startswith(p) for p in keep)}


def _extract_contracts(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in snapshot.items():
        if not k.startswith("core/contracts/"):
            continue
        parts = k.split("/")
        if len(parts) < 5:
            continue
        _, _, category, plugin, kind = parts[:5]
        cat = out.setdefault(category, {})
        entry = cat.setdefault(plugin, {"consumed": [], "produced": []})
        if kind in ("consumed", "produced"):
            entry[kind] = list(v) if isinstance(v, list) else [v]
    return out


cli_group("ctx", help="Inspect round_ctx.json (runtime carriers).")(ctx_app)


@ctx_app.command("show", help="Show round_ctx.json (optionally filtered by key prefix).")
def ctx_show(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: Optional[str] = typer.Option("latest", "--round", "-r"),
    keys: Optional[List[str]] = typer.Option(None, "--keys", help="Filter by key prefix (repeatable)."),
    json_out_flag: bool = typer.Option(False, "--json/--human", help="Output format (default: human)."),
) -> None:
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        ws = CampaignWorkspace.from_config(cfg, cfg_path)
        snap = _load_round_ctx(ws, round)
        snap = _filter_keys(snap, keys)
        if json_out_flag:
            json_out(snap)
        else:
            print_config_context(cfg_path, cfg=cfg)
            for k in sorted(snap.keys()):
                print_stdout(f"{k}: {snap[k]}")
    except OpalError as e:
        opal_error("ctx show", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("ctx show", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


@ctx_app.command("audit", help="Show per-plugin consumed/produced keys from round_ctx.json.")
def ctx_audit(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: Optional[str] = typer.Option("latest", "--round", "-r"),
    json_out_flag: bool = typer.Option(False, "--json/--human", help="Output format (default: human)."),
) -> None:
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        ws = CampaignWorkspace.from_config(cfg, cfg_path)
        snap = _load_round_ctx(ws, round)
        audit = _extract_contracts(snap)
        if json_out_flag:
            json_out(audit)
        else:
            print_config_context(cfg_path, cfg=cfg)
            for category in sorted(audit.keys()):
                print_stdout(f"[{category}]")
                for plugin in sorted(audit[category].keys()):
                    entry = audit[category][plugin]
                    consumed = entry.get("consumed", [])
                    produced = entry.get("produced", [])
                    print_stdout(f"  {plugin}: consumed={len(consumed)} produced={len(produced)}")
                    if consumed:
                        print_stdout(f"    consumed: {consumed}")
                    if produced:
                        print_stdout(f"    produced: {produced}")
    except OpalError as e:
        opal_error("ctx audit", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("ctx audit", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


@ctx_app.command("diff", help="Diff two round_ctx.json snapshots.")
def ctx_diff(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round_a: str = typer.Option(..., "--round-a", help="First round (int or 'latest')."),
    round_b: str = typer.Option(..., "--round-b", help="Second round (int or 'latest')."),
    keys: Optional[List[str]] = typer.Option(None, "--keys", help="Filter by key prefix (repeatable)."),
    json_out_flag: bool = typer.Option(False, "--json/--human", help="Output format (default: human)."),
) -> None:
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        ws = CampaignWorkspace.from_config(cfg, cfg_path)
        snap_a = _filter_keys(_load_round_ctx(ws, round_a), keys)
        snap_b = _filter_keys(_load_round_ctx(ws, round_b), keys)

        added = {k: v for k, v in snap_b.items() if k not in snap_a}
        removed = {k: v for k, v in snap_a.items() if k not in snap_b}
        changed = {
            k: {"from": snap_a[k], "to": snap_b[k]} for k in snap_a.keys() & snap_b.keys() if snap_a[k] != snap_b[k]
        }
        out = {"added": added, "removed": removed, "changed": changed}
        if json_out_flag:
            json_out(out)
        else:
            print_config_context(cfg_path, cfg=cfg)
            print_stdout(f"added={len(added)} removed={len(removed)} changed={len(changed)}")
            if added:
                print_stdout(f"added keys: {sorted(added.keys())}")
            if removed:
                print_stdout(f"removed keys: {sorted(removed.keys())}")
            if changed:
                print_stdout(f"changed keys: {sorted(changed.keys())}")
    except OpalError as e:
        opal_error("ctx diff", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("ctx diff", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
