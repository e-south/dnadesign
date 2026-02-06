"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/catalog.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import typer
from rich.console import Console
from rich.table import Table

from dnadesign.cruncher.app.campaign_service import select_catalog_entry
from dnadesign.cruncher.app.catalog_service import (
    get_entry,
    list_catalog,
    search_catalog,
)
from dnadesign.cruncher.app.motif_artifacts import (
    artifact_filename,
    build_densegen_artifact,
    build_manifest,
    load_motif_payload,
)
from dnadesign.cruncher.artifacts.layout import RUN_META_DIR, logos_dir_for_run, logos_root, out_root
from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    parse_config_and_value,
    resolve_config_path,
)
from dnadesign.cruncher.cli.paths import render_path
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.core.labels import build_run_name
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.store.catalog_store import CatalogMotifStore
from dnadesign.cruncher.store.motif_store import MotifRef
from dnadesign.cruncher.utils.hashing import sha256_bytes, sha256_lines, sha256_path
from dnadesign.cruncher.utils.paths import resolve_catalog_root
from dnadesign.cruncher.viz.logos import logo_subtitle, logo_title, site_entries_for_logo
from dnadesign.cruncher.viz.mpl import ensure_mpl_cache

app = typer.Typer(no_args_is_help=True, help="Query or inspect cached motifs and binding sites.")
console = Console()
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ResolvedTarget:
    tf_name: str
    ref: MotifRef
    entry: CatalogEntry
    site_entries: list[CatalogEntry]


LOGO_MANIFEST_NAME = "logo_manifest.json"


def _safe_stem(label: str) -> str:
    cleaned = _SAFE_RE.sub("_", label).strip("_")
    return cleaned or "motif"


def _resolve_export_format(out_path: Path, fmt: str | None, *, label: str) -> str:
    resolved = (fmt or "").strip().lower() if fmt is not None else ""
    if resolved:
        if resolved not in {"csv", "parquet"}:
            raise typer.BadParameter(f"{label} must be 'csv' or 'parquet', got: {fmt!r}.")
        return resolved
    suffix = out_path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    raise typer.BadParameter(f"{label} is required when output extension is not .csv/.parquet (got {out_path}).")


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        output.append(value)
        seen.add(value)
    return output


def _densegen_workspaces_root(config_path: Path) -> Path | None:
    for parent in (config_path.parent, *config_path.parents):
        candidate = parent / "src" / "dnadesign" / "densegen" / "workspaces"
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _resolve_densegen_workspace(selector: str, *, config_path: Path) -> Path:
    raw = str(selector or "").strip()
    if not raw:
        raise typer.BadParameter("--densegen-workspace must be a non-empty string.")
    candidate = Path(raw).expanduser()
    looks_like_path = candidate.is_absolute() or any(sep in raw for sep in (os.sep, os.altsep) if sep)
    if looks_like_path or candidate.exists():
        resolved = candidate
        if not resolved.is_absolute():
            resolved = (Path.cwd() / resolved).resolve()
    else:
        root = _densegen_workspaces_root(config_path)
        if root is None:
            raise typer.BadParameter(
                "Unable to locate DenseGen workspaces root relative to the cruncher config. "
                "Pass --densegen-workspace as an absolute path."
            )
        resolved = (root / raw).resolve()
    if not resolved.exists():
        raise typer.BadParameter(f"DenseGen workspace not found: {resolved}")
    if not resolved.is_dir():
        raise typer.BadParameter(f"DenseGen workspace is not a directory: {resolved}")
    config_candidate = resolved / "config.yaml"
    if not config_candidate.is_file():
        raise typer.BadParameter(f"DenseGen workspace missing config.yaml: {config_candidate}")
    inputs_root = resolved / "inputs"
    if not inputs_root.is_dir():
        raise typer.BadParameter(f"DenseGen workspace missing inputs/ directory: {inputs_root}")
    return resolved


def _require_densegen_inputs_path(path: Path, *, inputs_root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(inputs_root.resolve())
    except ValueError as exc:
        raise typer.BadParameter(f"{label} must be under {inputs_root} when --densegen-workspace is set.") from exc
    return resolved


def _remove_existing_artifacts(out_dir: Path, *, tf_names: Sequence[str]) -> int:
    stems = {_safe_stem(name) for name in tf_names if name}
    if not stems:
        return 0
    prefixes = {f"{stem}__" for stem in stems}
    singles = {f"{stem}.json" for stem in stems}
    removed = 0
    for path in out_dir.glob("*.json"):
        if path.name == "artifact_manifest.json":
            continue
        if path.name in singles or any(path.name.startswith(prefix) for prefix in prefixes):
            path.unlink()
            removed += 1
    return removed


def _logo_manifest_path(out_dir: Path) -> Path:
    return out_dir / RUN_META_DIR / LOGO_MANIFEST_NAME


def _load_logo_manifest(out_dir: Path) -> dict | None:
    path = _logo_manifest_path(out_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_logo_manifest(out_dir: Path, payload: dict) -> None:
    path = _logo_manifest_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _matrix_site_count_from_tags(tags: dict[str, object] | None) -> int | None:
    if not tags:
        return None
    for key in ("discovery_nsites", "meme_nsites", "site_count", "nsites"):
        raw = tags.get(key)
        if raw is None:
            continue
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _entry_signature(entry: CatalogEntry) -> dict[str, object]:
    return {
        "source": entry.source,
        "motif_id": entry.motif_id,
        "matrix_source": entry.matrix_source,
        "site_kind": entry.site_kind,
        "site_count": entry.site_count,
        "site_total": entry.site_total,
        "matrix_site_count": _matrix_site_count_from_tags(entry.tags),
    }


def _matrix_checksum(catalog_root: Path, entry: CatalogEntry) -> str:
    motif_path = catalog_root / "normalized" / "motifs" / entry.source / f"{entry.motif_id}.json"
    return sha256_path(motif_path)


def _sites_checksum(catalog_root: Path, entries: list[CatalogEntry]) -> str:
    lines: list[str] = []
    for candidate in sorted(entries, key=lambda e: (e.source, e.motif_id)):
        sites_path = catalog_root / "normalized" / "sites" / candidate.source / f"{candidate.motif_id}.jsonl"
        lines.append(f"{candidate.source}:{candidate.motif_id}:{sha256_path(sites_path)}")
    return sha256_lines(lines)


def _build_logo_signature(
    *,
    cfg,
    catalog_root: Path,
    targets: Sequence[ResolvedTarget],
    bits_mode: str,
    dpi: int,
) -> tuple[str, dict]:
    pwm_config = {
        "pwm_source": cfg.catalog.pwm_source,
        "pwm_window_lengths": cfg.catalog.pwm_window_lengths,
        "pwm_window_strategy": cfg.catalog.pwm_window_strategy,
    }
    if cfg.catalog.pwm_source == "sites":
        pwm_config.update(
            {
                "combine_sites": cfg.catalog.combine_sites,
                "site_kinds": cfg.catalog.site_kinds,
                "site_window_lengths": cfg.catalog.site_window_lengths,
                "site_window_center": cfg.catalog.site_window_center,
                "min_sites_for_pwm": cfg.catalog.min_sites_for_pwm,
                "allow_low_sites": cfg.catalog.allow_low_sites,
                "pseudocounts": cfg.catalog.pseudocounts,
            }
        )
    target_payloads: list[dict[str, object]] = []
    for target in sorted(targets, key=lambda t: (t.tf_name, t.entry.source, t.entry.motif_id)):
        payload: dict[str, object] = {
            "tf_name": target.tf_name,
            "ref": f"{target.entry.source}:{target.entry.motif_id}",
            "entry": _entry_signature(target.entry),
        }
        if cfg.catalog.pwm_source == "matrix":
            payload["matrix_sha256"] = _matrix_checksum(catalog_root, target.entry)
        else:
            entries = target.site_entries
            payload["site_entries"] = [
                _entry_signature(candidate) for candidate in sorted(entries, key=lambda e: (e.source, e.motif_id))
            ]
            payload["sites_sha256"] = _sites_checksum(catalog_root, entries)
        target_payloads.append(payload)
    signature_payload = {
        "render": {"bits_mode": bits_mode, "dpi": dpi},
        "pwm": pwm_config,
        "targets": target_payloads,
    }
    signature = sha256_bytes(json.dumps(signature_payload, sort_keys=True).encode("utf-8"))
    return signature, signature_payload


def _find_existing_logo_run(root_dir: Path, signature: str) -> Path | None:
    if not root_dir.exists():
        return None
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        manifest = _load_logo_manifest(child)
        if manifest and manifest.get("signature") == signature:
            return child
    return None


def _resolve_set_tfs(cfg, set_index: int | None) -> list[str]:
    if set_index is None:
        return []
    if set_index < 1 or set_index > len(cfg.regulator_sets):
        raise typer.BadParameter(f"--set must be between 1 and {len(cfg.regulator_sets)} (got {set_index}).")
    return list(cfg.regulator_sets[set_index - 1])


def _parse_ref(ref: str) -> tuple[str, str]:
    if ":" not in ref:
        raise typer.BadParameter(
            "Expected <source>:<motif_id> reference. Hint: cruncher catalog show regulondb:RDBECOLITFC00214"
        )
    source, motif_id = ref.split(":", 1)
    return source, motif_id


def _ensure_entry_matches_pwm_source(
    entry: CatalogEntry,
    pwm_source: str,
    site_kinds: list[str] | None,
    *,
    tf_name: str,
    ref: str,
) -> None:
    if pwm_source == "matrix":
        if not entry.has_matrix:
            raise ValueError(f"{ref} does not have a cached motif matrix for TF '{tf_name}'.")
        return
    if pwm_source == "sites":
        if not entry.has_sites:
            raise ValueError(f"{ref} does not have cached binding sites for TF '{tf_name}'.")
        if site_kinds is not None and entry.site_kind not in site_kinds:
            raise ValueError(
                f"{ref} site kind '{entry.site_kind}' is not in site_kinds={site_kinds} for TF '{tf_name}'."
            )
        return
    raise ValueError("pwm_source must be 'matrix' or 'sites'")


def _resolve_targets(
    *,
    cfg,
    config_path: Path,
    tfs: Sequence[str],
    refs: Sequence[str],
    set_index: int | None,
    source_filter: str | None,
) -> tuple[list[ResolvedTarget], CatalogIndex]:
    if set_index is not None and (tfs or refs):
        raise typer.BadParameter("--set cannot be combined with --tf or --ref.")
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    catalog = CatalogIndex.load(catalog_root)
    tf_names = list(tfs)
    if set_index is not None:
        tf_names = _resolve_set_tfs(cfg, set_index)
        if not tf_names and not refs:
            raise typer.BadParameter(f"regulator_sets[{set_index}] is empty.")
    if not tf_names and not refs:
        tf_names = [tf for group in cfg.regulator_sets for tf in group]
    tf_names = _dedupe(tf_names)
    refs = _dedupe(refs)
    if not tf_names and not refs:
        raise typer.BadParameter("No targets resolved. Provide --tf, --ref, or --set.")

    targets: list[ResolvedTarget] = []
    seen_keys: set[str] = set()

    for ref in refs:
        source, motif_id = _parse_ref(ref)
        if source_filter and source_filter != source:
            raise typer.BadParameter(f"--source {source_filter} does not match explicit ref {ref}.")
        entry = catalog.entries.get(f"{source}:{motif_id}")
        if entry is None:
            raise ValueError(f"No catalog entry found for {ref}.")
        _ensure_entry_matches_pwm_source(
            entry,
            cfg.catalog.pwm_source,
            cfg.catalog.site_kinds,
            tf_name=entry.tf_name,
            ref=ref,
        )
        site_entries = []
        if cfg.catalog.pwm_source == "sites":
            site_entries = site_entries_for_logo(
                catalog=catalog,
                entry=entry,
                combine_sites=cfg.catalog.combine_sites,
                site_kinds=cfg.catalog.site_kinds,
            )
        key = entry.key
        if key in seen_keys:
            continue
        targets.append(
            ResolvedTarget(
                tf_name=entry.tf_name,
                ref=MotifRef(source=entry.source, motif_id=entry.motif_id),
                entry=entry,
                site_entries=site_entries,
            )
        )
        seen_keys.add(key)

    if tf_names:
        catalog_for_select = catalog
        if source_filter:
            filtered = {k: v for k, v in catalog.entries.items() if v.source == source_filter}
            catalog_for_select = CatalogIndex(entries=filtered)
        for tf_name in tf_names:
            if source_filter:
                all_candidates = catalog.list(tf_name=tf_name, include_synonyms=True)
                if not any(candidate.source == source_filter for candidate in all_candidates):
                    raise ValueError(f"No cached entries for '{tf_name}' in source '{source_filter}'.")
            entry = select_catalog_entry(
                catalog=catalog_for_select,
                tf_name=tf_name,
                pwm_source=cfg.catalog.pwm_source,
                site_kinds=cfg.catalog.site_kinds,
                combine_sites=cfg.catalog.combine_sites,
                source_preference=cfg.catalog.source_preference,
                dataset_preference=cfg.catalog.dataset_preference,
                dataset_map=cfg.catalog.dataset_map,
                allow_ambiguous=cfg.catalog.allow_ambiguous,
            )
            site_entries = []
            if cfg.catalog.pwm_source == "sites":
                site_entries = site_entries_for_logo(
                    catalog=catalog,
                    entry=entry,
                    combine_sites=cfg.catalog.combine_sites,
                    site_kinds=cfg.catalog.site_kinds,
                )
                if not site_entries:
                    raise ValueError(
                        f"No cached site entries available for '{tf_name}' "
                        f"with cruncher.catalog.site_kinds={cfg.catalog.site_kinds}."
                    )
            key = entry.key
            if key in seen_keys:
                continue
            targets.append(
                ResolvedTarget(
                    tf_name=tf_name,
                    ref=MotifRef(source=entry.source, motif_id=entry.motif_id),
                    entry=entry,
                    site_entries=site_entries,
                )
            )
            seen_keys.add(key)

    if not targets:
        raise typer.BadParameter("No catalog entries matched the requested targets.")
    return targets, catalog


def _resolve_site_targets(
    *,
    cfg,
    config_path: Path,
    tfs: Sequence[str],
    refs: Sequence[str],
    set_index: int | None,
    source_filter: str | None,
) -> tuple[list[ResolvedTarget], CatalogIndex]:
    if set_index is not None and (tfs or refs):
        raise typer.BadParameter("--set cannot be combined with --tf or --ref.")
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    catalog = CatalogIndex.load(catalog_root)
    tf_names = list(tfs)
    if set_index is not None:
        tf_names = _resolve_set_tfs(cfg, set_index)
        if not tf_names and not refs:
            raise typer.BadParameter(f"regulator_sets[{set_index}] is empty.")
    if not tf_names and not refs:
        tf_names = [tf for group in cfg.regulator_sets for tf in group]
    tf_names = _dedupe(tf_names)
    refs = _dedupe(refs)
    if not tf_names and not refs:
        raise typer.BadParameter("No targets resolved. Provide --tf, --ref, or --set.")

    targets: list[ResolvedTarget] = []
    seen_keys: set[str] = set()

    for ref in refs:
        source, motif_id = _parse_ref(ref)
        if source_filter and source_filter != source:
            raise typer.BadParameter(f"--source {source_filter} does not match explicit ref {ref}.")
        entry = catalog.entries.get(f"{source}:{motif_id}")
        if entry is None:
            raise ValueError(f"No catalog entry found for {ref}.")
        _ensure_entry_matches_pwm_source(
            entry,
            "sites",
            cfg.catalog.site_kinds,
            tf_name=entry.tf_name,
            ref=ref,
        )
        site_entries = site_entries_for_logo(
            catalog=catalog,
            entry=entry,
            combine_sites=cfg.catalog.combine_sites,
            site_kinds=cfg.catalog.site_kinds,
        )
        if not site_entries:
            raise ValueError(
                "No cached site entries available for "
                f"'{entry.tf_name}' with cruncher.catalog.site_kinds={cfg.catalog.site_kinds}."
            )
        key = entry.key
        if key in seen_keys:
            continue
        targets.append(
            ResolvedTarget(
                tf_name=entry.tf_name,
                ref=MotifRef(source=entry.source, motif_id=entry.motif_id),
                entry=entry,
                site_entries=site_entries,
            )
        )
        seen_keys.add(key)

    if tf_names:
        catalog_for_select = catalog
        if source_filter:
            filtered = {k: v for k, v in catalog.entries.items() if v.source == source_filter}
            catalog_for_select = CatalogIndex(entries=filtered)
        for tf_name in tf_names:
            if source_filter:
                all_candidates = catalog.list(tf_name=tf_name, include_synonyms=True)
                if not any(candidate.source == source_filter for candidate in all_candidates):
                    raise ValueError(f"No cached entries for '{tf_name}' in source '{source_filter}'.")
            entry = select_catalog_entry(
                catalog=catalog_for_select,
                tf_name=tf_name,
                pwm_source="sites",
                site_kinds=cfg.catalog.site_kinds,
                combine_sites=cfg.catalog.combine_sites,
                source_preference=cfg.catalog.source_preference,
                dataset_preference=cfg.catalog.dataset_preference,
                dataset_map=cfg.catalog.dataset_map,
                allow_ambiguous=cfg.catalog.allow_ambiguous,
            )
            site_entries = site_entries_for_logo(
                catalog=catalog,
                entry=entry,
                combine_sites=cfg.catalog.combine_sites,
                site_kinds=cfg.catalog.site_kinds,
            )
            if not site_entries:
                raise ValueError(
                    "No cached site entries available for "
                    f"'{tf_name}' with cruncher.catalog.site_kinds={cfg.catalog.site_kinds}."
                )
            key = entry.key
            if key in seen_keys:
                continue
            targets.append(
                ResolvedTarget(
                    tf_name=tf_name,
                    ref=MotifRef(source=entry.source, motif_id=entry.motif_id),
                    entry=entry,
                    site_entries=site_entries,
                )
            )
            seen_keys.add(key)

    if not targets:
        raise typer.BadParameter("No catalog entries matched the requested targets.")
    return targets, catalog


def _render_pwm_matrix(table_title: str, pwm_matrix: list[list[float]]) -> Table:
    table = Table(title=table_title, header_style="bold")
    table.add_column("Pos", justify="right")
    table.add_column("A")
    table.add_column("C")
    table.add_column("G")
    table.add_column("T")
    for idx, row in enumerate(pwm_matrix, start=1):
        table.add_row(str(idx), *(f"{val:.3f}" for val in row))
    return table


@app.command("list", help="List cached motifs and site sets.")
def list_entries(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (resolved from workspace/CWD if omitted).",
        metavar="CONFIG",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    tf: Optional[str] = typer.Option(None, "--tf", help="Filter by TF name."),
    source: Optional[str] = typer.Option(None, "--source", help="Filter by source adapter."),
    organism: Optional[str] = typer.Option(None, "--organism", help="Filter by organism name or strain."),
    include_synonyms: bool = typer.Option(
        False,
        "--include-synonyms",
        help="Match TF synonyms in addition to tf_name.",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    entries = list_catalog(
        catalog_root,
        tf_name=tf,
        source=source,
        organism=organism,
        include_synonyms=include_synonyms,
    )
    if not entries:
        console.print("No catalog entries found.")
        console.print("Hint: run cruncher fetch motifs --tf <name> <config> to populate the cache.")
        raise typer.Exit(code=1)
    table = Table(title="Catalog", header_style="bold")
    table.add_column("TF")
    table.add_column("Source")
    table.add_column("Motif ID")
    table.add_column("Organism")
    table.add_column("Matrix")
    table.add_column("Sites (cached seq/total)")
    table.add_column("Sites (matrix n)")
    table.add_column("Site kind")
    table.add_column("Dataset")
    table.add_column("Method")
    table.add_column("Mean len")
    table.add_column("Updated")
    for entry in entries:
        organism_label = "-"
        if entry.organism:
            organism_label = entry.organism.get("name") or entry.organism.get("strain") or "-"
        matrix_flag = "yes" if entry.has_matrix else "no"
        if entry.has_matrix and entry.matrix_source:
            matrix_flag = f"{matrix_flag} ({entry.matrix_source})"
        sites_flag = f"{entry.site_count}/{entry.site_total}" if entry.has_sites else "-"
        matrix_sites = _matrix_site_count_from_tags(entry.tags)
        matrix_sites_flag = str(matrix_sites) if matrix_sites is not None else "-"
        mean_len = "-"
        if entry.site_length_mean is not None:
            mean_len = f"{entry.site_length_mean:.1f}"
        table.add_row(
            entry.tf_name,
            entry.source,
            entry.motif_id,
            organism_label,
            matrix_flag,
            sites_flag,
            matrix_sites_flag,
            entry.site_kind or "-",
            entry.dataset_id or "-",
            entry.dataset_method or "-",
            mean_len,
            entry.updated_at.split("T")[0],
        )
    console.print(table)


@app.command("search", help="Search cached motifs by name or motif ID.")
def search_entries(
    args: list[str] = typer.Argument(
        None,
        help="Query (optionally preceded by CONFIG).",
        metavar="ARGS",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    source: Optional[str] = typer.Option(None, "--source", help="Filter by source adapter."),
    organism: Optional[str] = typer.Option(None, "--organism", help="Filter by organism name or strain."),
    regex: bool = typer.Option(False, "--regex", help="Treat query as a regular expression."),
    case_sensitive: bool = typer.Option(False, "--case-sensitive", help="Enable case-sensitive matching."),
    fuzzy: bool = typer.Option(False, "--fuzzy", help="Use Levenshtein ratio to rank approximate matches."),
    min_score: float = typer.Option(0.6, "--min-score", help="Minimum fuzzy score (0-1)."),
    limit: Optional[int] = typer.Option(25, "--limit", help="Limit number of returned entries."),
) -> None:
    try:
        config_path, query = parse_config_and_value(
            args,
            config_option,
            value_label="QUERY",
            command_hint="cruncher catalog search <query>",
        )
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    if fuzzy and regex:
        raise typer.BadParameter(
            "--fuzzy and --regex are mutually exclusive. Hint: use --fuzzy for approximate matches."
        )
    if not (0.0 <= min_score <= 1.0):
        raise typer.BadParameter("--min-score must be between 0 and 1. Hint: try 0.6.")
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    entries = search_catalog(
        catalog_root,
        query=query,
        source=source,
        organism=organism,
        regex=regex,
        case_sensitive=case_sensitive,
        fuzzy=fuzzy,
        min_score=min_score,
        limit=limit,
    )
    if not entries:
        console.print(f"No catalog matches for '{query}'.")
        console.print("Hint: run cruncher catalog list <config> to inspect cached entries.")
        raise typer.Exit(code=1)
    table = Table(title=f"Catalog search: {query}", header_style="bold")
    table.add_column("TF")
    table.add_column("Source")
    table.add_column("Motif ID")
    table.add_column("Organism")
    table.add_column("Matrix")
    table.add_column("Sites (cached seq/total)")
    table.add_column("Sites (matrix n)")
    table.add_column("Site kind")
    table.add_column("Dataset")
    table.add_column("Method")
    table.add_column("Mean len")
    for entry in entries:
        organism_label = "-"
        if entry.organism:
            organism_label = entry.organism.get("name") or entry.organism.get("strain") or "-"
        matrix_flag = "yes" if entry.has_matrix else "no"
        if entry.has_matrix and entry.matrix_source:
            matrix_flag = f"{matrix_flag} ({entry.matrix_source})"
        sites_flag = f"{entry.site_count}/{entry.site_total}" if entry.has_sites else "-"
        matrix_sites = _matrix_site_count_from_tags(entry.tags)
        matrix_sites_flag = str(matrix_sites) if matrix_sites is not None else "-"
        mean_len = "-"
        if entry.site_length_mean is not None:
            mean_len = f"{entry.site_length_mean:.1f}"
        table.add_row(
            entry.tf_name,
            entry.source,
            entry.motif_id,
            organism_label,
            matrix_flag,
            sites_flag,
            matrix_sites_flag,
            entry.site_kind or "-",
            entry.dataset_id or "-",
            entry.dataset_method or "-",
            mean_len,
        )
    console.print(table)


@app.command("resolve", help="Resolve a TF name to cached motif candidates.")
def resolve_tf(
    args: list[str] = typer.Argument(
        None,
        help="TF name (optionally preceded by CONFIG).",
        metavar="ARGS",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    source: Optional[str] = typer.Option(None, "--source", help="Filter by source adapter."),
    organism: Optional[str] = typer.Option(None, "--organism", help="Filter by organism name or strain."),
    include_synonyms: bool = typer.Option(True, "--include-synonyms", help="Include TF synonyms in resolution."),
) -> None:
    try:
        config_path, tf = parse_config_and_value(
            args,
            config_option,
            value_label="TF",
            command_hint="cruncher catalog resolve <tf_name>",
        )
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    entries = list_catalog(
        catalog_root,
        tf_name=tf,
        source=source,
        organism=organism,
        include_synonyms=include_synonyms,
    )
    if not entries:
        console.print(f"No cached entries for TF '{tf}'.")
        console.print("Hint: run cruncher fetch motifs --tf <name> <config> to populate the cache.")
        raise typer.Exit(code=1)
    table = Table(title=f"TF resolve: {tf}", header_style="bold")
    table.add_column("Source")
    table.add_column("Motif ID")
    table.add_column("Organism")
    table.add_column("Matrix")
    table.add_column("Sites (cached seq/total)")
    table.add_column("Sites (matrix n)")
    table.add_column("Site kind")
    table.add_column("Dataset")
    table.add_column("Method")
    table.add_column("Mean len")
    for entry in entries:
        organism_label = "-"
        if entry.organism:
            organism_label = entry.organism.get("name") or entry.organism.get("strain") or "-"
        matrix_flag = "yes" if entry.has_matrix else "no"
        if entry.has_matrix and entry.matrix_source:
            matrix_flag = f"{matrix_flag} ({entry.matrix_source})"
        sites_flag = f"{entry.site_count}/{entry.site_total}" if entry.has_sites else "-"
        matrix_sites = _matrix_site_count_from_tags(entry.tags)
        matrix_sites_flag = str(matrix_sites) if matrix_sites is not None else "-"
        mean_len = "-"
        if entry.site_length_mean is not None:
            mean_len = f"{entry.site_length_mean:.1f}"
        table.add_row(
            entry.source,
            entry.motif_id,
            organism_label,
            matrix_flag,
            sites_flag,
            matrix_sites_flag,
            entry.site_kind or "-",
            entry.dataset_id or "-",
            entry.dataset_method or "-",
            mean_len,
        )
    console.print(table)


@app.command("show", help="Show metadata for a cached motif reference.")
def show(
    args: list[str] = typer.Argument(
        None,
        help="Catalog reference (<source>:<motif_id>), optionally preceded by CONFIG.",
        metavar="ARGS",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
) -> None:
    try:
        config_path, ref = parse_config_and_value(
            args,
            config_option,
            value_label="REF",
            command_hint="cruncher catalog show regulondb:RBM000123",
        )
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    if ":" not in ref:
        raise typer.BadParameter(
            "Expected <source>:<motif_id> reference. Hint: cruncher catalog show regulondb:RBM000123"
        )
    source, motif_id = ref.split(":", 1)
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    entry = get_entry(catalog_root, source=source, motif_id=motif_id)
    if entry is None:
        console.print(f"No catalog entry found for {ref}")
        console.print("Hint: run cruncher catalog list <config> to inspect cached entries.")
        raise typer.Exit(code=1)
    console.print(f"source: {entry.source}")
    console.print(f"motif_id: {entry.motif_id}")
    console.print(f"tf_name: {entry.tf_name}")
    console.print(f"organism: {entry.organism or '-'}")
    console.print(f"kind: {entry.kind}")
    console.print(f"matrix_length: {entry.matrix_length}")
    console.print(f"matrix_source: {entry.matrix_source}")
    console.print(f"matrix_semantics: {entry.matrix_semantics}")
    console.print(f"has_matrix: {entry.has_matrix}")
    console.print(f"has_sites: {entry.has_sites}")
    console.print(f"site_count: {entry.site_count}")
    console.print(f"site_total: {entry.site_total}")
    console.print(f"site_kind: {entry.site_kind or '-'}")
    if entry.site_length_mean is not None:
        console.print(
            f"site_length_mean: {entry.site_length_mean:.2f} "
            f"(min={entry.site_length_min}, max={entry.site_length_max}, n={entry.site_length_count})"
        )
    else:
        console.print("site_length_mean: -")
    console.print(f"site_length_source: {entry.site_length_source or '-'}")
    console.print(f"dataset_id: {entry.dataset_id or '-'}")
    console.print(f"dataset_source: {entry.dataset_source or '-'}")
    console.print(f"dataset_method: {entry.dataset_method or '-'}")
    console.print(f"reference_genome: {entry.reference_genome or '-'}")
    console.print(f"updated_at: {entry.updated_at}")
    synonyms = entry.tags.get("synonyms") if entry.tags else None
    console.print(f"synonyms: {synonyms or '-'}")
    motif_path = catalog_root / "normalized" / "motifs" / entry.source / f"{entry.motif_id}.json"
    sites_path = catalog_root / "normalized" / "sites" / entry.source / f"{entry.motif_id}.jsonl"
    console.print(f"motif_path: {render_path(motif_path, base=config_path.parent) if motif_path.exists() else '-'}")
    console.print(f"sites_path: {render_path(sites_path, base=config_path.parent) if sites_path.exists() else '-'}")


@app.command("pwms", help="Summarize or export cached PWMs for selected TFs or motif refs.")
def pwms(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (resolved from workspace/CWD if omitted).",
        metavar="CONFIG",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    tf: list[str] = typer.Option([], "--tf", help="TF name to include (repeatable)."),
    ref: list[str] = typer.Option([], "--ref", help="Catalog reference (<source>:<motif_id>, repeatable)."),
    set_index: int | None = typer.Option(
        None,
        "--set",
        help="Regulator set index from config (1-based).",
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        help="Limit TF resolution to a single source adapter.",
    ),
    matrix: bool = typer.Option(False, "--matrix", help="Print full PWM matrices after the summary."),
    log_odds: bool = typer.Option(False, "--log-odds", help="Also emit log-odds matrices (table or JSON)."),
    output_format: str = typer.Option(
        "table",
        "--format",
        help="Output format: table or json.",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    if output_format not in {"table", "json"}:
        raise typer.BadParameter("--format must be 'table' or 'json'.")
    if log_odds and output_format == "table" and not matrix:
        raise typer.BadParameter("--log-odds requires --matrix for table output.")
    try:
        targets, _catalog = _resolve_targets(
            cfg=cfg,
            config_path=config_path,
            tfs=tf,
            refs=ref,
            set_index=set_index,
            source_filter=source,
        )
        catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
        store = CatalogMotifStore(
            catalog_root,
            pwm_source=cfg.catalog.pwm_source,
            site_kinds=cfg.catalog.site_kinds,
            combine_sites=cfg.catalog.combine_sites,
            site_window_lengths=cfg.catalog.site_window_lengths,
            site_window_center=cfg.catalog.site_window_center,
            pwm_window_lengths=cfg.catalog.pwm_window_lengths,
            pwm_window_strategy=cfg.catalog.pwm_window_strategy,
            min_sites_for_pwm=cfg.catalog.min_sites_for_pwm,
            allow_low_sites=cfg.catalog.allow_low_sites,
            pseudocounts=cfg.catalog.pseudocounts,
        )
        payloads: list[dict[str, object]] = []
        resolved: list[tuple[ResolvedTarget, object]] = []
        table = Table(title="PWM summary", header_style="bold")
        table.add_column("TF")
        table.add_column("Source")
        table.add_column("Motif ID")
        table.add_column("PWM source")
        table.add_column("Length")
        table.add_column("Window")
        table.add_column("Bits")
        table.add_column("Sites (cached seq/total)")
        table.add_column("Sites (matrix n)")
        table.add_column("Site sets")
        for target in targets:
            pwm = store.get_pwm(target.ref)
            resolved.append((target, pwm))
            info_bits = pwm.information_bits()
            cached_sites = f"{target.entry.site_count}/{target.entry.site_total}" if target.entry.has_sites else "-"
            site_sets = "-" if cfg.catalog.pwm_source == "matrix" else str(len(target.site_entries))
            window = "-"
            if pwm.source_length is not None and pwm.window_start is not None:
                window = f"{pwm.window_start}:{pwm.window_start + pwm.length}/{pwm.source_length}"
            table.add_row(
                target.tf_name,
                target.entry.source,
                target.entry.motif_id,
                cfg.catalog.pwm_source,
                str(pwm.length),
                window,
                f"{info_bits:.2f}",
                cached_sites,
                str(pwm.nsites or "-"),
                site_sets,
            )
            record = {
                "tf_name": target.tf_name,
                "source": target.entry.source,
                "motif_id": target.entry.motif_id,
                "pwm_source": cfg.catalog.pwm_source,
                "length": pwm.length,
                "window_start": pwm.window_start,
                "source_length": pwm.source_length,
                "window_strategy": pwm.window_strategy,
                "window_score": pwm.window_score,
                "info_bits": info_bits,
                "nsites": pwm.nsites,
                "sites_cached": target.entry.site_count if target.entry.has_sites else None,
                "sites_cached_total": target.entry.site_total if target.entry.has_sites else None,
                "site_sets": len(target.site_entries) if cfg.catalog.pwm_source == "sites" else None,
            }
            if output_format == "json":
                record["matrix"] = pwm.matrix.tolist()
                if log_odds:
                    record["log_odds"] = pwm.log_odds().tolist()
            payloads.append(record)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: run cruncher fetch motifs/sites before catalog pwms.")
        raise typer.Exit(code=1)

    if output_format == "json":
        typer.echo(json.dumps(payloads, indent=2))
        return
    console.print(table)
    if matrix:
        for target, pwm in resolved:
            label = f"{target.tf_name} ({target.entry.source}:{target.entry.motif_id})"
            console.print(_render_pwm_matrix(f"PWM: {label}", pwm.matrix.tolist()))
            if log_odds:
                console.print(_render_pwm_matrix(f"Log-odds: {label}", pwm.log_odds().tolist()))


@app.command("export-densegen", help="Export DenseGen motif artifacts (one file per motif).")
def export_densegen(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (resolved from workspace/CWD if omitted).",
        metavar="CONFIG",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    tf: list[str] = typer.Option([], "--tf", help="TF name to include (repeatable)."),
    ref: list[str] = typer.Option([], "--ref", help="Catalog reference (<source>:<motif_id>, repeatable)."),
    set_index: int | None = typer.Option(
        None,
        "--set",
        help="Regulator set index from config (1-based).",
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        help="Limit TF resolution to a single source adapter.",
    ),
    densegen_workspace: str | None = typer.Option(
        None,
        "--densegen-workspace",
        help="DenseGen workspace name or path (defaults --out to inputs/motif_artifacts).",
    ),
    out_dir: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Directory to write DenseGen motif artifacts.",
    ),
    background: str = typer.Option(
        "record",
        "--background",
        help="Background policy: record | uniform | matrix.",
    ),
    pseudocount: float | None = typer.Option(
        None,
        "--pseudocount",
        help="Optional pseudocount for log-odds (>= 0).",
    ),
    producer: str = typer.Option(
        "cruncher",
        "--producer",
        help="Producer label for DenseGen artifacts.",
    ),
    clean: bool = typer.Option(
        True,
        "--clean/--no-clean",
        help="Remove existing motif artifacts for selected TFs before export.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Allow overwriting existing files."),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    if background not in {"record", "uniform", "matrix"}:
        raise typer.BadParameter("--background must be 'record', 'uniform', or 'matrix'.")
    if not producer.strip():
        raise typer.BadParameter("--producer must be a non-empty string.")
    densegen_root = None
    if densegen_workspace:
        densegen_root = _resolve_densegen_workspace(densegen_workspace, config_path=config_path)
        inputs_root = densegen_root / "inputs"
        if out_dir is None:
            out_dir = inputs_root / "motif_artifacts"
        else:
            out_dir = _require_densegen_inputs_path(out_dir, inputs_root=inputs_root, label="--out")
    if out_dir is None:
        raise typer.BadParameter("--out is required when --densegen-workspace is not set.")

    try:
        targets, catalog = _resolve_targets(
            cfg=cfg,
            config_path=config_path,
            tfs=tf,
            refs=ref,
            set_index=set_index,
            source_filter=source,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: run cruncher fetch motifs/sites before catalog export-densegen.")
        raise typer.Exit(code=1)

    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if clean:
        removed = _remove_existing_artifacts(out_dir, tf_names=[target.tf_name for target in targets])
        if removed:
            console.print(f"[dim]Removed {removed} existing artifact(s) for selected TFs.[/]")

    manifest_entries: list[dict[str, object]] = []
    table = Table(title="DenseGen motif artifacts", header_style="bold")
    table.add_column("TF")
    table.add_column("Source")
    table.add_column("Motif ID")
    table.add_column("Artifact")

    for target in sorted(targets, key=lambda t: (t.entry.source, t.entry.motif_id)):
        motif_path = catalog_root / "normalized" / "motifs" / target.entry.source / f"{target.entry.motif_id}.json"
        if not motif_path.exists():
            console.print(f"[red]Missing motif file:[/] {motif_path}")
            raise typer.Exit(code=1)
        payload = load_motif_payload(motif_path)
        artifact = build_densegen_artifact(
            payload,
            producer=producer.strip(),
            background_policy=background,
            pseudocount=pseudocount,
        )
        filename = artifact_filename(
            tf_name=target.tf_name,
            source=target.entry.source,
            motif_id=target.entry.motif_id,
        )
        dest = out_dir / filename
        if dest.exists() and not overwrite:
            console.print(f"[red]Artifact already exists:[/] {dest}")
            raise typer.Exit(code=1)
        dest.write_text(json.dumps(artifact, indent=2, sort_keys=True))
        table.add_row(target.tf_name, target.entry.source, target.entry.motif_id, str(dest))
        manifest_entries.append(
            {
                "tf_name": target.tf_name,
                "source": target.entry.source,
                "motif_id": target.entry.motif_id,
                "path": str(dest),
            }
        )

    manifest = build_manifest(
        producer=producer.strip(),
        entries=manifest_entries,
        config_path=config_path,
        catalog_root=catalog_root,
        background_policy=background,
        pseudocount=pseudocount,
    )
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path = out_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    console.print(table)
    console.print(f"[green]Wrote manifest:[/] {manifest_path}")


@app.command("export-sites", help="Export cached binding sites for DenseGen (CSV/Parquet).")
def export_sites(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (resolved from workspace/CWD if omitted).",
        metavar="CONFIG",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    tf: list[str] = typer.Option([], "--tf", help="TF name to include (repeatable)."),
    ref: list[str] = typer.Option([], "--ref", help="Catalog reference (<source>:<motif_id>, repeatable)."),
    set_index: int | None = typer.Option(
        None,
        "--set",
        help="Regulator set index from config (1-based).",
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        help="Limit TF resolution to a single source adapter.",
    ),
    densegen_workspace: str | None = typer.Option(
        None,
        "--densegen-workspace",
        help="DenseGen workspace name or path (defaults --out to inputs/densegen_sites.parquet).",
    ),
    out_path: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Output file path (.csv or .parquet).",
    ),
    fmt: str | None = typer.Option(
        None,
        "--format",
        help="Output format: csv | parquet (inferred from --out if omitted).",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Allow overwriting existing file."),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    densegen_root = None
    if densegen_workspace:
        densegen_root = _resolve_densegen_workspace(densegen_workspace, config_path=config_path)
        inputs_root = densegen_root / "inputs"
        if out_path is None:
            out_path = inputs_root / "densegen_sites.parquet"
        else:
            out_path = _require_densegen_inputs_path(out_path, inputs_root=inputs_root, label="--out")
    if out_path is None:
        raise typer.BadParameter("--out is required when --densegen-workspace is not set.")

    try:
        targets, catalog = _resolve_site_targets(
            cfg=cfg,
            config_path=config_path,
            tfs=tf,
            refs=ref,
            set_index=set_index,
            source_filter=source,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: run cruncher fetch sites --hydrate before export-sites.")
        raise typer.Exit(code=1)

    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if out_path.is_dir():
            console.print(f"[red]Output path is a directory:[/] {out_path}")
            raise typer.Exit(code=1)
        if not overwrite:
            console.print(f"[red]Output file already exists:[/] {out_path}")
            raise typer.Exit(code=1)

    fmt = _resolve_export_format(out_path, fmt, label="--format")
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)

    rows: list[dict[str, object]] = []
    table = Table(title="DenseGen binding-site export", header_style="bold")
    table.add_column("TF")
    table.add_column("Source")
    table.add_column("Motif ID")
    table.add_column("Sites")

    for target in sorted(targets, key=lambda t: (t.tf_name, t.entry.source, t.entry.motif_id)):
        site_entries = target.site_entries or [target.entry]
        for entry in site_entries:
            sites_path = catalog_root / "normalized" / "sites" / entry.source / f"{entry.motif_id}.jsonl"
            if not sites_path.exists():
                console.print(f"[red]Missing sites file:[/] {sites_path}")
                raise typer.Exit(code=1)
            count = 0
            with sites_path.open() as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    seq = payload.get("sequence")
                    if not seq:
                        site_id = payload.get("site_id") or "unknown"
                        motif_ref = payload.get("motif_ref") or f"{entry.source}:{entry.motif_id}"
                        console.print(f"[red]Missing sequence for site {site_id} in {motif_ref}.[/]")
                        console.print("Hint: re-fetch with `cruncher fetch sites --hydrate` or provide genome FASTA.")
                        raise typer.Exit(code=1)
                    rows.append(
                        {
                            "tf": target.tf_name,
                            "tfbs": seq,
                            "site_id": payload.get("site_id"),
                            "source": payload.get("motif_ref") or f"{entry.source}:{entry.motif_id}",
                        }
                    )
                    count += 1
            table.add_row(target.tf_name, entry.source, entry.motif_id, str(count))

    if not rows:
        console.print("[red]No binding sites found for selected targets.[/]")
        raise typer.Exit(code=1)

    import pandas as pd

    df = pd.DataFrame(rows)
    try:
        if fmt == "csv":
            df.to_csv(out_path, index=False)
        else:
            df.to_parquet(out_path, index=False)
    except Exception as exc:
        console.print(f"[red]Failed to write export:[/] {exc}")
        raise typer.Exit(code=1)

    console.print(table)
    console.print(f"[green]Wrote binding sites:[/] {out_path} ({len(df)} rows)")


@app.command("logos", help="Render PWM logos for selected TFs or motif refs.")
def logos(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (resolved from workspace/CWD if omitted).",
        metavar="CONFIG",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    tf: list[str] = typer.Option([], "--tf", help="TF name to include (repeatable)."),
    ref: list[str] = typer.Option([], "--ref", help="Catalog reference (<source>:<motif_id>, repeatable)."),
    set_index: int | None = typer.Option(
        None,
        "--set",
        help="Regulator set index from config (1-based).",
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        help="Limit TF resolution to a single source adapter.",
    ),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        help="Directory to write logo PNGs (defaults to <out_dir>/logos/catalog/<run>).",
    ),
    bits_mode: str | None = typer.Option(
        None,
        "--bits-mode",
        help="Logo mode: information or probability (default: information).",
    ),
    dpi: int | None = typer.Option(
        None,
        "--dpi",
        help="DPI for logo output (default: 150).",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    try:
        catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
        ensure_mpl_cache(catalog_root)
        targets, catalog = _resolve_targets(
            cfg=cfg,
            config_path=config_path,
            tfs=tf,
            refs=ref,
            set_index=set_index,
            source_filter=source,
        )
        store = CatalogMotifStore(
            catalog_root,
            pwm_source=cfg.catalog.pwm_source,
            site_kinds=cfg.catalog.site_kinds,
            combine_sites=cfg.catalog.combine_sites,
            site_window_lengths=cfg.catalog.site_window_lengths,
            site_window_center=cfg.catalog.site_window_center,
            pwm_window_lengths=cfg.catalog.pwm_window_lengths,
            pwm_window_strategy=cfg.catalog.pwm_window_strategy,
            min_sites_for_pwm=cfg.catalog.min_sites_for_pwm,
            allow_low_sites=cfg.catalog.allow_low_sites,
            pseudocounts=cfg.catalog.pseudocounts,
        )
        resolved_bits_mode = bits_mode or "information"
        resolved_dpi = dpi or 150
        if resolved_bits_mode not in {"information", "probability"}:
            raise typer.BadParameter("--bits-mode must be 'information' or 'probability'.")
        signature, signature_payload = _build_logo_signature(
            cfg=cfg,
            catalog_root=catalog_root,
            targets=targets,
            bits_mode=resolved_bits_mode,
            dpi=resolved_dpi,
        )
        out_base = out_dir
        if out_base is None:
            existing = _find_existing_logo_run(
                logos_root(out_root(config_path, cfg.out_dir)) / "catalog",
                signature,
            )
            if existing is not None:
                console.print(f"Logos already rendered at {render_path(existing, base=config_path.parent)}")
                return
            name_set_index = set_index if len(cfg.regulator_sets) > 1 else None
            run_name = build_run_name(
                "catalog",
                [t.tf_name for t in targets],
                set_index=name_set_index,
                include_stage=False,
            )
            out_base = logos_dir_for_run(
                out_root(config_path, cfg.out_dir),
                "catalog",
                run_name,
            )
        else:
            manifest = _load_logo_manifest(out_base)
            if manifest and manifest.get("signature") == signature:
                console.print(f"Logos already rendered at {render_path(out_base, base=config_path.parent)}")
                return
        out_base.mkdir(parents=True, exist_ok=True)
        from dnadesign.cruncher.viz.pwm import plot_pwm

        table = Table(title="Rendered PWM logos", header_style="bold")
        table.add_column("TF")
        table.add_column("Source")
        table.add_column("Motif ID")
        table.add_column("Length")
        table.add_column("Bits")
        table.add_column("Output")
        outputs: list[str] = []
        for target in targets:
            pwm = store.get_pwm(target.ref)
            info_bits = pwm.information_bits()
            subtitle = logo_subtitle(
                pwm_source=cfg.catalog.pwm_source,
                entry=target.entry,
                catalog=catalog,
                combine_sites=cfg.catalog.combine_sites,
                site_kinds=cfg.catalog.site_kinds,
            )
            title = logo_title(
                tf_name=target.tf_name,
                motif_id=target.entry.motif_id,
                nsites=pwm.nsites,
            )
            stem = _safe_stem(f"{target.tf_name}_{target.entry.source}_{target.entry.motif_id}")
            out_path = out_base / f"{stem}_logo.png"
            plot_pwm(
                pwm,
                mode=resolved_bits_mode,
                out=out_path,
                title=title,
                dpi=resolved_dpi,
                subtitle=f"sites: {subtitle}" if cfg.catalog.pwm_source == "sites" else subtitle,
            )
            outputs.append(str(out_path))
            table.add_row(
                target.tf_name,
                target.entry.source,
                target.entry.motif_id,
                str(pwm.length),
                f"{info_bits:.2f}",
                render_path(out_path, base=config_path.parent),
            )
        console.print(table)
        console.print(f"Logos saved to {render_path(out_base, base=config_path.parent)}")
        _write_logo_manifest(
            out_base,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "signature": signature,
                "render": signature_payload["render"],
                "pwm": signature_payload["pwm"],
                "targets": signature_payload["targets"],
                "outputs": outputs,
            },
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: run cruncher fetch motifs/sites before catalog logos.")
        raise typer.Exit(code=1)
