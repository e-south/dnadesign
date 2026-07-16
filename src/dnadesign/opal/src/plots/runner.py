"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/runner.py

Executes configured plots for OPAL campaigns using PlotContext. Owns plot.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
import typer

from ..analysis.ledger import RoundSelector, available_rounds, parse_round_selector, round_suffix
from ..core.utils import ExitCodes, OpalError, now_iso
from ..plots._context import PlotContext
from ..plots._mpl_utils import ensure_mpl_config_dir
from ..plots.manifests import (
    build_plot_manifest,
    load_plot_manifest_index,
    refresh_plot_manifest_freshness,
    write_plot_manifest,
    write_plot_manifest_index,
)
from ..registries.plots import describe_plot_kind, get_plot
from ..storage.data_access import RecordsStore
from ..storage.workspace import CampaignWorkspace
from .config import apply_data_entries, parse_enabled, parse_tags, validate_plot_entry


@dataclass(frozen=True)
class PlotRequest:
    plots_cfg: List[Dict[str, Any]]
    plot_defaults: Dict[str, Any]
    plot_presets: Dict[str, Dict[str, Any]]
    plot_cfg_dir: Path
    campaign_dir: Path
    workspace: CampaignWorkspace
    store: RecordsStore
    rounds_sel: RoundSelector
    run_id: Optional[str]
    selection_view_id: str
    objective_name: str
    objective_family: str
    multi_view_campaign: bool
    round_suffix: str
    name_filter: Optional[str]
    tag_filters: List[str]
    emit_status: bool = True


def _validate_plot_objective_compatibility(
    *,
    plot_name: str,
    plot_kind: str,
    plot_family: str,
    selection_view_id: str,
    objective_name: str,
    objective_family: str,
) -> None:
    if plot_family == "generic":
        return
    if plot_family == "unknown":
        raise OpalError(
            f"[plot] Plot {plot_name!r} (kind={plot_kind!r}) does not declare an objective family. "
            "Register explicit PlotMeta with objective_family='generic' or the exact specialized family.",
            ExitCodes.BAD_ARGS,
        )
    if plot_family == objective_family:
        return
    raise OpalError(
        f"[plot] Plot {plot_name!r} (kind={plot_kind!r}) requires objective family "
        f"{plot_family!r}, but selection view {selection_view_id!r} uses objective "
        f"{objective_name!r} in family {objective_family!r}. Select a compatible "
        "specialized plot or a generic plot.",
        ExitCodes.BAD_ARGS,
    )


def resolve_run_round(runs_df: pl.DataFrame, run_id: str) -> int:
    if runs_df.is_empty():
        raise ValueError("[plot] outputs/ledger/runs.parquet is empty; cannot resolve run_id.")
    if "run_id" not in runs_df.columns or "as_of_round" not in runs_df.columns:
        raise ValueError("[plot] outputs/ledger/runs.parquet missing required columns (run_id, as_of_round).")
    df = runs_df.filter(pl.col("run_id") == str(run_id)).select(pl.col("as_of_round").drop_nulls().unique())
    if df.is_empty():
        raise ValueError(f"[plot] run_id not found in outputs/ledger/runs.parquet: {run_id!r}.")
    rounds = sorted({int(x) for x in df.to_series().to_list()})
    if len(rounds) > 1:
        raise ValueError(f"[plot] run_id {run_id!r} appears in multiple rounds {rounds}.")
    return rounds[0]


def _unique_run_id_for_scope(
    runs_df: pl.DataFrame | None,
    *,
    rounds_sel: RoundSelector,
    explicit_run_id: str | None,
) -> str | None:
    """Resolve a singular run only when the concrete plot scope is unambiguous."""

    if explicit_run_id is not None:
        return str(explicit_run_id)
    if runs_df is None or runs_df.is_empty() or not {"as_of_round", "run_id"} <= set(runs_df.columns):
        return None
    available = available_rounds(runs_df)
    if not available:
        return None
    if isinstance(rounds_sel, str) and rounds_sel in {"latest", "unspecified"}:
        scoped_rounds = [available[-1]]
    elif rounds_sel == "all":
        scoped_rounds = available
    elif isinstance(rounds_sel, list):
        scoped_rounds = sorted({int(value) for value in rounds_sel})
    else:
        return None
    if len(scoped_rounds) != 1:
        return None
    run_ids = (
        runs_df.filter(pl.col("as_of_round") == scoped_rounds[0])
        .select(pl.col("run_id").drop_nulls().cast(pl.Utf8).unique())
        .to_series()
        .to_list()
    )
    unique = sorted({str(value) for value in run_ids if str(value).strip()})
    return unique[0] if len(unique) == 1 else None


def _resolve_output_dir(
    out_cfg: dict,
    *,
    campaign_dir: Path,
    workspace: CampaignWorkspace,
    plot_name: str,
    plot_kind: str,
    round_suffix: str,
) -> Path:
    out_dir_tpl = out_cfg.get("dir")
    if out_dir_tpl:
        out_dir_str = str(out_dir_tpl).format(
            campaign=str(campaign_dir),
            workdir=str(workspace.workdir),
            name=plot_name,
            kind=plot_kind,
            round_suffix=round_suffix,
        )
        out_dir = Path(out_dir_str)
        if not out_dir.is_absolute():
            out_dir = (campaign_dir / out_dir).resolve()
        else:
            out_dir = out_dir.resolve()
    else:
        out_dir = (campaign_dir / "outputs" / "plots").resolve()
    return out_dir


def _entry_round_scope(
    *,
    req: PlotRequest,
    entry: Dict[str, Any],
    preset: Dict[str, Any],
    plot_name: str,
) -> tuple[RoundSelector, str]:
    if req.run_id:
        return req.rounds_sel, req.round_suffix

    raw = entry.get("round_selector", preset.get("round_selector"))
    if raw is None:
        return req.rounds_sel, req.round_suffix
    if isinstance(raw, list):
        raise ValueError(
            f"[plot] plot '{plot_name}' round_selector must be a scalar selector "
            "('latest', 'all', '3', '1,3', or '2-5'), not a YAML list."
        )
    rounds_sel = parse_round_selector(str(raw))
    return rounds_sel, round_suffix(rounds_sel)


def _entry_round_scopes(
    *,
    req: PlotRequest,
    entry: Dict[str, Any],
    preset: Dict[str, Any],
    plot_name: str,
    plot_kind: str,
) -> list[tuple[RoundSelector, str]]:
    """Resolve one or more concrete plot output scopes for a configured entry."""

    base_scope = _entry_round_scope(req=req, entry=entry, preset=preset, plot_name=plot_name)
    raw_variants = entry.get("round_variants", preset.get("round_variants"))
    if raw_variants is None:
        return [base_scope]
    if req.run_id:
        raise ValueError(
            f"[plot] plot '{plot_name}' uses round_variants, but --run-id is single-run. "
            "Remove round_variants or omit --run-id."
        )

    variants = _parse_round_variants(raw_variants, plot_name=plot_name)
    _reject_inherent_history_fanout(variants=variants, plot_name=plot_name, plot_kind=plot_kind)
    scopes: list[tuple[RoundSelector, str]] = []
    for variant in variants:
        variant_key = variant.strip().lower()
        if variant_key in {"configured", "default", "base"}:
            scopes.append(base_scope)
            continue
        if variant_key == "each":
            for round_index in _available_variant_rounds(req=req, plot_name=plot_name):
                round_scope: RoundSelector = [int(round_index)]
                scopes.append((round_scope, round_suffix(round_scope)))
            continue
        parsed = parse_round_selector(variant_key)
        if parsed == "unspecified":
            raise ValueError(
                f"[plot] plot '{plot_name}' round_variants must not contain an empty or unspecified selector."
            )
        scopes.append((parsed, round_suffix(parsed)))

    seen_suffixes: set[str] = set()
    unique: list[tuple[RoundSelector, str]] = []
    for rounds_sel, suffix in scopes:
        if suffix in seen_suffixes:
            raise ValueError(
                f"[plot] plot '{plot_name}' round_variants produced duplicate output suffix {suffix!r}. "
                "Use distinct selectors."
            )
        seen_suffixes.add(suffix)
        unique.append((rounds_sel, suffix))
    return unique


def _parse_round_variants(value: Any, *, plot_name: str) -> list[str]:
    if isinstance(value, bool):
        if value:
            return ["configured", "each"]
        raise ValueError(f"[plot] plot '{plot_name}' round_variants: false is invalid; remove the key instead.")
    if isinstance(value, str):
        variants = [value]
    elif isinstance(value, list):
        variants = value
    else:
        raise ValueError(
            f"[plot] plot '{plot_name}' round_variants must be a string or list of strings "
            f"(got {type(value).__name__})."
        )
    parsed: list[str] = []
    for item in variants:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"[plot] plot '{plot_name}' round_variants entries must be non-empty strings.")
        parsed.append(item.strip())
    return parsed


def _reject_inherent_history_fanout(*, variants: list[str], plot_name: str, plot_kind: str) -> None:
    variant_keys = {variant.strip().lower() for variant in variants}
    if "each" not in variant_keys:
        return
    try:
        metadata = describe_plot_kind(plot_kind)
    except Exception:
        return
    capability = metadata.get("capability")
    if not isinstance(capability, dict):
        return
    if str(capability.get("round_scope") or "") != "round_history":
        return
    raise ValueError(
        f"[plot] plot '{plot_name}' kind '{plot_kind}' declares round_scope=round_history, so "
        "round_variants must not include 'each'. Generate one all-round history artifact, or request "
        "a specific round explicitly when debugging a snapshot."
    )


def _available_variant_rounds(*, req: PlotRequest, plot_name: str) -> list[int]:
    runs_path = req.workspace.ledger_runs_path
    if not runs_path.exists():
        raise ValueError(
            f"[plot] plot '{plot_name}' uses round_variants: each, but no run ledger exists at {runs_path}. "
            "Run OPAL first or remove the per-round fan-out."
        )
    runs_df = pl.read_parquet(runs_path)
    if "as_of_round" not in runs_df.columns:
        raise ValueError(
            f"[plot] plot '{plot_name}' cannot expand round_variants: each because {runs_path} "
            "is missing column 'as_of_round'."
        )
    rounds = available_rounds(runs_df)
    if not rounds:
        raise ValueError(
            f"[plot] plot '{plot_name}' cannot expand round_variants: each because {runs_path} has no rounds."
        )
    return rounds


def run_plots(req: PlotRequest) -> bool:
    ensure_mpl_config_dir(workdir=req.workspace.workdir)

    runs_df = None
    if req.workspace.ledger_runs_path.exists():
        runs_df = pl.read_parquet(req.workspace.ledger_runs_path)

    builtins = {
        "records": Path(req.store.records_path),
        "outputs": req.workspace.outputs_dir,
        "ledger_predictions_dir": req.workspace.ledger_predictions_dir,
        "ledger_runs_parquet": req.workspace.ledger_runs_path,
        "ledger_labels_parquet": req.workspace.ledger_labels_path,
    }
    builtin_resolved = {k: p for k, p in builtins.items() if p.exists()}

    any_fail = False
    manifests_by_dir: dict[Path, list[dict[str, Any]]] = {}

    for entry in req.plots_cfg:
        if not isinstance(entry, dict):
            raise ValueError(f"[plot] Each plot entry must be a mapping (got {type(entry).__name__}).")
        validate_plot_entry(entry, ctx="plot entry")

        preset: Dict[str, Any] = {}
        preset_name = entry.get("preset")
        if preset_name is not None:
            if not isinstance(preset_name, str):
                raise ValueError(f"[plot] plot preset name must be a string (got {type(preset_name).__name__}).")
            if preset_name not in req.plot_presets:
                raise ValueError(f"[plot] Unknown plot preset: {preset_name!r}")
            preset = req.plot_presets.get(preset_name) or {}

        pname = entry.get("name")
        if not pname or not isinstance(pname, str):
            raise ValueError("[plot] Each plot requires a string 'name'.")

        if req.name_filter and pname != req.name_filter:
            continue

        pkind = entry.get("kind") or preset.get("kind")
        if not pkind or not isinstance(pkind, str):
            raise ValueError(f"[plot] Plot '{pname}' is missing 'kind' (or preset kind).")

        enabled = parse_enabled(
            entry.get("enabled") if "enabled" in entry else preset.get("enabled"),
            ctx=pname,
        )
        if not enabled:
            if req.name_filter:
                raise ValueError(f"[plot] Plot '{pname}' is disabled (enabled: false).")
            typer.echo(f"[plot] Skipping disabled plot: {pname}")
            continue

        tags = parse_tags(preset.get("tags"), ctx=f"preset:{preset_name}") + parse_tags(
            entry.get("tags"), ctx=f"plot:{pname}"
        )
        if req.tag_filters:
            if not set(tags).intersection(req.tag_filters):
                if req.name_filter:
                    raise ValueError(f"[plot] Plot '{pname}' does not match tags: {req.tag_filters}")
                continue

        capability = describe_plot_kind(pkind)["capability"]
        _validate_plot_objective_compatibility(
            plot_name=pname,
            plot_kind=pkind,
            plot_family=str(capability["objective_family"]),
            selection_view_id=req.selection_view_id,
            objective_name=req.objective_name,
            objective_family=req.objective_family,
        )

        data_paths = dict(builtin_resolved)
        apply_data_entries(
            data_paths,
            req.plot_defaults.get("data"),
            base_dir=req.plot_cfg_dir,
            ctx="plot_defaults.data",
        )
        if preset:
            apply_data_entries(
                data_paths,
                preset.get("data"),
                base_dir=req.plot_cfg_dir,
                ctx=f"plot_presets.{preset_name}.data",
            )
        apply_data_entries(
            data_paths,
            entry.get("data"),
            base_dir=req.plot_cfg_dir,
            ctx=f"plot '{pname}'.data",
        )

        preset_out = preset.get("output") or {}
        if preset and not isinstance(preset_out, dict):
            raise ValueError(f"[plot] plot_presets.{preset_name}.output must be a mapping.")
        entry_out = entry.get("output")
        if entry_out is None:
            entry_out = {}
        if not isinstance(entry_out, dict):
            raise ValueError(f"[plot] plot '{pname}' output must be a mapping.")
        out_cfg = {
            **(req.plot_defaults.get("output") or {}),
            **preset_out,
            **entry_out,
        }
        entry_round_scopes = _entry_round_scopes(
            req=req,
            entry=entry,
            preset=preset,
            plot_name=pname,
            plot_kind=pkind,
        )
        save_data = bool(out_cfg.get("save_data", False))

        raw_params = entry.get("params", None)
        if raw_params is None:
            if "params" in entry:
                raise ValueError(f"[plot] plot '{pname}' has an empty 'params:' block. Use {{}} or remove it.")
            entry_params = {}
        elif not isinstance(raw_params, dict):
            raise ValueError(f"[plot] plot '{pname}' has a non-mapping 'params' (type={type(raw_params).__name__}).")
        else:
            entry_params = dict(raw_params)

        preset_params = preset.get("params") or {}
        if preset and not isinstance(preset_params, dict):
            raise ValueError(f"[plot] plot_presets.{preset_name}.params must be a mapping.")

        params = {
            **(req.plot_defaults.get("params") or {}),
            **preset_params,
            **entry_params,
        }

        for entry_rounds_sel, entry_round_suffix in entry_round_scopes:
            out_dir = _resolve_output_dir(
                out_cfg,
                campaign_dir=req.campaign_dir,
                workspace=req.workspace,
                plot_name=pname,
                plot_kind=pkind,
                round_suffix=entry_round_suffix,
            )
            if req.multi_view_campaign:
                out_dir = Path(out_dir) / "selection_views" / req.selection_view_id
            fmt = (out_cfg.get("format") or "png").lower()
            dpi = int(out_cfg.get("dpi", 600))
            fname = (out_cfg.get("filename") or "{name}{round_suffix}.png").format(
                name=pname,
                round_suffix=entry_round_suffix,
            )
            if not fname.lower().endswith(f".{fmt}"):
                base = fname.rsplit(".", 1)[0] if "." in fname else fname
                fname = f"{base}.{fmt}"

            logger = _plot_logger(pname, emit_status=req.emit_status)

            ctx = PlotContext(
                campaign_dir=req.campaign_dir,
                workspace=req.workspace,
                rounds=entry_rounds_sel,
                run_id=_unique_run_id_for_scope(
                    runs_df,
                    rounds_sel=entry_rounds_sel,
                    explicit_run_id=req.run_id,
                ),
                selection_view_id=req.selection_view_id,
                data_paths=data_paths,
                output_dir=Path(out_dir),
                filename=fname,
                dpi=dpi,
                format=fmt,
                logger=logger,
                save_data=save_data,
            )

            started_at = now_iso()
            try:
                ctx.output_dir.mkdir(parents=True, exist_ok=True)
                debug = str(os.getenv("OPAL_DEBUG", "")).strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
                if debug and req.emit_status:
                    if isinstance(entry.get("params"), dict):
                        params_preview = {k: entry["params"].get(k) for k in (entry.get("params") or {}).keys()}
                    else:
                        params_preview = "(not a dict)"
                    typer.secho(
                        f"[plot] entry '{pname}': keys={sorted(entry.keys())} "
                        f"params_type={type(entry.get('params')).__name__} "
                        f"params_preview={params_preview}",
                        fg=typer.colors.BLUE,
                    )

                get_plot(pkind)(ctx, params)
                manifest = build_plot_manifest(
                    name=pname,
                    kind=pkind,
                    params=params,
                    context=ctx,
                    status="written",
                    started_at=started_at,
                )
                write_plot_manifest(manifest)
                manifests_by_dir.setdefault(ctx.output_dir, []).append(manifest)
                if manifest.get("status") != "written":
                    any_fail = True
                    _plot_status(req, f"[fail] {pname} ({pkind}) did not write expected media", fg=typer.colors.RED)
                    continue
                _plot_status(
                    req,
                    f"[ok] {pname} ({pkind}) → {ctx.output_dir / ctx.filename}",
                    fg=typer.colors.GREEN,
                )
            except Exception as exc:
                any_fail = True
                manifest = build_plot_manifest(
                    name=pname,
                    kind=pkind,
                    params=params,
                    context=ctx,
                    status="failed",
                    started_at=started_at,
                    error=exc,
                )
                write_plot_manifest(manifest)
                manifests_by_dir.setdefault(ctx.output_dir, []).append(manifest)
                _plot_status(req, f"[fail] {pname} ({pkind})", fg=typer.colors.RED)
                if req.emit_status:
                    traceback.print_exc()
            finally:
                _close_plot_figures()

    merge_existing_index = bool(req.name_filter or req.tag_filters)
    for output_dir, manifests in manifests_by_dir.items():
        write_plot_manifest_index(
            output_dir,
            _merged_manifest_index_rows(output_dir, manifests, merge_existing=merge_existing_index),
        )

    return any_fail


def _merged_manifest_index_rows(
    output_dir: Path,
    manifests: List[dict[str, Any]],
    *,
    merge_existing: bool,
) -> list[dict[str, Any]]:
    """Preserve unrelated manifest-index rows during targeted plot reruns."""

    rows = [dict(row) for row in manifests]
    if not merge_existing or not rows:
        return rows
    index_path = Path(output_dir) / "plot_manifest.json"
    if not index_path.exists():
        return rows
    existing = load_plot_manifest_index(index_path)
    replaced_identities = {identity for row in rows for identity in _manifest_index_identity_values(row)}
    preserved = [
        _refresh_preserved_manifest_row(row)
        for row in existing.get("manifests", [])
        if isinstance(row, dict) and _manifest_index_identity_values(row).isdisjoint(replaced_identities)
    ]
    return [*preserved, *rows]


def _manifest_index_identity_values(row: dict[str, Any]) -> set[tuple[str, str]]:
    identities: set[tuple[str, str]] = set()
    plot_id = str(row.get("plot_id") or "").strip()
    if plot_id:
        identities.add(("plot_id", plot_id))
    manifest_path = str(row.get("manifest_path") or "").strip()
    if manifest_path:
        identities.add(("manifest_path", str(Path(manifest_path))))
    if not identities:
        name = str(row.get("name") or "").strip()
        if name:
            identities.add(("name", name))
    return identities


def _refresh_preserved_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    refreshed = refresh_plot_manifest_freshness(row)
    manifest_path = refreshed.get("manifest_path")
    if manifest_path:
        write_plot_manifest(refreshed)
    return refreshed


def _plot_logger(plot_name: str, *, emit_status: bool) -> logging.Logger:
    if not emit_status:
        logger = logging.Logger(f"opal.plot.{plot_name}.quiet")
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return logger

    logger = logging.getLogger(f"opal.plot.{plot_name}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(rich_tracebacks=False, markup=True, show_path=False, show_time=False)
        except Exception:
            handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        if not isinstance(handler, logging.StreamHandler):
            handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def _plot_status(req: PlotRequest, message: str, *, fg: str) -> None:
    if req.emit_status:
        typer.secho(message, fg=fg)


def _close_plot_figures() -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.close("all")
