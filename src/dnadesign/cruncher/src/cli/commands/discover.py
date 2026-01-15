"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/discover.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import typer
from dnadesign.cruncher.app.fetch_service import write_motif_record
from dnadesign.cruncher.cli.commands.catalog import _resolve_targets
from dnadesign.cruncher.cli.config_resolver import ConfigResolutionError, resolve_config_path
from dnadesign.cruncher.cli.paths import render_path
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.core.labels import build_run_name
from dnadesign.cruncher.ingest.models import OrganismRef
from dnadesign.cruncher.ingest.normalize import build_motif_record
from dnadesign.cruncher.ingest.site_windows import resolve_window_length
from dnadesign.cruncher.integrations.meme_suite import (
    check_meme_tools,
    resolve_executable,
    resolve_tool_path,
    tool_version,
)
from dnadesign.cruncher.io.parsers.meme import parse_meme_file
from dnadesign.cruncher.store.catalog_index import CatalogIndex
from dnadesign.cruncher.store.catalog_store import iter_site_sequences
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="Discover motifs from cached binding sites.")
console = Console()


def _safe_id(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("_") or "motif"


def _purge_discovered_entries(
    *,
    catalog_root: Path,
    catalog: CatalogIndex,
    source_id: str,
    tf_name: str,
) -> int:
    tf_norm = tf_name.lower()
    removed: list[str] = []
    for key, entry in list(catalog.entries.items()):
        if entry.source != source_id:
            continue
        if entry.tf_name.lower() != tf_norm:
            continue
        motif_path = catalog_root / "normalized" / "motifs" / entry.source / f"{entry.motif_id}.json"
        if motif_path.exists():
            motif_path.unlink()
        removed.append(key)
        del catalog.entries[key]
    return len(removed)


def _as_organism(entry) -> OrganismRef | None:
    if entry.organism is None:
        return None
    return OrganismRef(
        taxon=entry.organism.get("taxon"),
        name=entry.organism.get("name"),
        strain=entry.organism.get("strain"),
        assembly=entry.organism.get("assembly"),
    )


def _choose_tool(tool: str, *, nseq: int, streme_threshold: int) -> str:
    if tool == "auto":
        return "streme" if nseq >= streme_threshold else "meme"
    return tool


def _run_command(cmd: list[str], *, cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "(no stderr)"
        stdout = result.stdout.strip() or "(no stdout)"
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nstdout:\n{stdout}\nstderr:\n{stderr}")


def _write_fasta(path: Path, sequences: Iterable[str]) -> int:
    count = 0
    with path.open("w") as fh:
        for idx, seq in enumerate(sequences, start=1):
            fh.write(f">site_{idx}\n{seq}\n")
            count += 1
    return count


def _write_discover_manifest(
    *,
    run_dir: Path,
    config_path: Path,
    tool: str,
    exe: Path,
    version: str | None,
    command: list[str],
    tf_name: str,
    motif_source: str,
    window_sites: bool,
    site_window_lengths: dict[str, int] | None,
) -> Path:
    payload = {
        "stage": "discover",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "tool": tool,
        "tool_path": str(exe),
        "tool_version": version,
        "command": command,
        "tf_name": tf_name,
        "motif_source": motif_source,
        "window_sites": window_sites,
    }
    if window_sites:
        payload["site_window_lengths"] = site_window_lengths or {}
    manifest_path = run_dir / "discover_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def _resolve_config_maybe(config: Path | None, config_option: Path | None):
    if config or config_option:
        return resolve_config_path(config_option or config)
    return None


@app.command("check", help="Check MEME Suite availability (streme/meme).")
def check_tools(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (optional).",
        metavar="CONFIG",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    tool: str | None = typer.Option(
        None,
        "--tool",
        help="Tool to check: auto, streme, or meme (defaults to motif_discovery.tool).",
    ),
    tool_path: Path | None = typer.Option(
        None,
        "--tool-path",
        help="Optional path to MEME Suite binary or bin directory (overrides config/MEME_BIN).",
    ),
) -> None:
    config_path = None
    cfg = None
    try:
        config_path = _resolve_config_maybe(config, config_option)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    if config_path is not None:
        cfg = load_config(config_path)

    resolved_tool = (tool or (cfg.motif_discovery.tool if cfg else "auto")).lower()
    resolved_path = resolve_tool_path(
        tool_path or (cfg.motif_discovery.tool_path if cfg else None),
        config_path=config_path,
    )
    if resolved_tool not in {"auto", "streme", "meme"}:
        raise typer.BadParameter("--tool must be auto, streme, or meme.")
    try:
        ok, statuses = check_meme_tools(tool=resolved_tool, tool_path=resolved_path)
    except ValueError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)

    table = Table(title="MEME Suite check", header_style="bold")
    table.add_column("Tool")
    table.add_column("Status")
    table.add_column("Path")
    table.add_column("Version")
    table.add_column("Hint")

    base = config_path.parent if config_path is not None else None
    for status in statuses:
        path_val = status.path
        rendered_path = "-" if path_val in {None, "-"} else render_path(path_val, base=base)
        table.add_row(
            status.tool,
            status.status,
            rendered_path,
            status.version,
            status.hint,
        )

    console.print(table)
    if not ok:
        raise typer.Exit(code=1)


@app.command("motifs", help="Run MEME Suite (STREME/MEME) on cached sites and ingest discovered motifs.")
def discover_motifs(
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
    source_id: str | None = typer.Option(
        None,
        "--source-id",
        help="Catalog source_id for discovered motifs (overrides motif_discovery.source_id).",
    ),
    tool: str | None = typer.Option(
        None,
        "--tool",
        help="Discovery tool: auto, streme, or meme (defaults to motif_discovery.tool).",
    ),
    tool_path: Path | None = typer.Option(
        None,
        "--tool-path",
        help="Optional path to MEME Suite binary or bin directory (overrides config/MEME_BIN).",
    ),
    window_sites: bool | None = typer.Option(
        None,
        "--window-sites/--no-window-sites",
        help="Pre-window binding sites using motif_store.site_window_lengths before discovery.",
    ),
    replace_existing: bool | None = typer.Option(
        None,
        "--replace-existing/--keep-existing",
        help="Replace existing discovered motifs for the same TF/source before writing new ones.",
    ),
    minw: int | None = typer.Option(
        None,
        "--minw",
        help="Minimum motif width (defaults to config; auto from site lengths if unset).",
    ),
    maxw: int | None = typer.Option(
        None,
        "--maxw",
        help="Maximum motif width (defaults to config; auto from site lengths if unset).",
    ),
    nmotifs: int | None = typer.Option(None, "--nmotifs", help="Number of motifs to report per TF."),
    meme_mod: str | None = typer.Option(
        None,
        "--meme-mod",
        help="MEME -mod setting: oops, zoops, or anr (MEME only).",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)

    tool = (tool or cfg.motif_discovery.tool).lower()
    resolved_tool_path = resolve_tool_path(
        tool_path or cfg.motif_discovery.tool_path,
        config_path=config_path,
    )
    minw = minw if minw is not None else cfg.motif_discovery.minw
    maxw = maxw if maxw is not None else cfg.motif_discovery.maxw
    nmotifs = nmotifs or cfg.motif_discovery.nmotifs
    meme_mod = meme_mod or cfg.motif_discovery.meme_mod
    streme_threshold = cfg.motif_discovery.min_sequences_for_streme
    window_sites = cfg.motif_discovery.window_sites if window_sites is None else window_sites
    replace_existing = cfg.motif_discovery.replace_existing if replace_existing is None else replace_existing

    if tool not in {"auto", "streme", "meme"}:
        raise typer.BadParameter("--tool must be auto, streme, or meme.")
    if minw is not None and minw < 1:
        raise typer.BadParameter("--minw must be >= 1.")
    if maxw is not None and maxw < 1:
        raise typer.BadParameter("--maxw must be >= 1.")
    if minw is not None and maxw is not None and maxw < minw:
        raise typer.BadParameter("--maxw must be >= --minw.")
    if nmotifs < 1:
        raise typer.BadParameter("--nmotifs must be >= 1.")
    if meme_mod is not None and meme_mod.lower() not in {"oops", "zoops", "anr"}:
        raise typer.BadParameter("--meme-mod must be oops, zoops, or anr.")
    if meme_mod is not None:
        meme_mod = meme_mod.lower()

    resolved_source_id = source_id or cfg.motif_discovery.source_id
    try:
        cfg_for_sites = cfg.model_copy(deep=True)
        cfg_for_sites.motif_store.pwm_source = "sites"
        targets, _ = _resolve_targets(
            cfg=cfg_for_sites,
            config_path=config_path,
            tfs=tf,
            refs=ref,
            set_index=set_index,
            source_filter=source,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: run cruncher fetch sites before discovery.")
        raise typer.Exit(code=1)

    catalog_root = config_path.parent / cfg.motif_store.catalog_root
    catalog_root.mkdir(parents=True, exist_ok=True)
    out_root = catalog_root / "discoveries"
    out_root.mkdir(parents=True, exist_ok=True)
    catalog = CatalogIndex.load(catalog_root)

    table = Table(title="Motif discovery", header_style="bold")
    table.add_column("TF")
    table.add_column("Tool")
    table.add_column("Motif ID")
    table.add_column("Length")
    table.add_column("Output")

    base = config_path.parent
    for target in targets:
        if not target.site_entries:
            console.print(f"Error: no cached site entries for {target.tf_name}.")
            raise typer.Exit(code=1)
        if window_sites:
            missing = []
            for entry in target.site_entries:
                window_length = resolve_window_length(
                    tf_name=entry.tf_name,
                    dataset_id=entry.dataset_id,
                    window_lengths=cfg.motif_store.site_window_lengths,
                )
                if window_length is None:
                    missing.append(f"{entry.source}:{entry.motif_id}")
            if missing:
                console.print(
                    f"Error: window_sites is enabled but no site_window_lengths entry for TF '{target.tf_name}'."
                )
                console.print(
                    "Hint: set motif_store.site_window_lengths for this TF (or dataset:<id>) "
                    "or run with --no-window-sites."
                )
                raise typer.Exit(code=1)
        run_dir = out_root / build_run_name("discover", [target.tf_name])
        run_dir.mkdir(parents=True, exist_ok=True)
        fasta_path = run_dir / f"{_safe_id(target.tf_name)}_sites.fasta"
        site_window_lengths = cfg.motif_store.site_window_lengths if window_sites else {}
        sequences = list(
            iter_site_sequences(
                root=catalog_root,
                entries=target.site_entries,
                site_window_lengths=site_window_lengths,
                site_window_center=cfg.motif_store.site_window_center,
                allow_variable_lengths=True,
            )
        )
        if not sequences:
            console.print(f"Error: no site sequences available for {target.tf_name}.")
            raise typer.Exit(code=1)
        lengths = [len(seq) for seq in sequences]
        min_len = min(lengths)
        max_len = max(lengths)
        resolved_minw = minw if minw is not None else min_len
        resolved_maxw = maxw if maxw is not None else max_len
        if resolved_minw < 1 or resolved_maxw < 1 or resolved_maxw < resolved_minw:
            console.print(
                f"Error: invalid motif width range for {target.tf_name}: minw={resolved_minw}, maxw={resolved_maxw}."
            )
            raise typer.Exit(code=1)
        if minw is None or maxw is None:
            console.print(
                f"INFO {target.tf_name}: using site-length bounds for discovery "
                f"(minw={resolved_minw}, maxw={resolved_maxw}, site lengths {min_len}-{max_len})."
            )
        _write_fasta(fasta_path, sequences)

        chosen_tool = _choose_tool(tool, nseq=len(sequences), streme_threshold=streme_threshold)
        try:
            exe = resolve_executable(chosen_tool, tool_path=resolved_tool_path)
        except FileNotFoundError as exc:
            console.print(f"Error: {exc}")
            raise typer.Exit(code=1)
        if exe is None:
            available = [name for name in ("streme", "meme") if resolve_executable(name, tool_path=None)]
            if tool == "auto" and available:
                hint = f"Auto selected {chosen_tool}. Install it or re-run with --tool {available[0]}."
            elif available:
                hint = f"Available tools on PATH: {', '.join(available)}. Use --tool to select one."
            else:
                hint = "Install MEME Suite and set MEME_BIN or motif_discovery.tool_path."
            console.print(f"Error: {chosen_tool} not available. {hint} (Run `cruncher discover check`.)")
            raise typer.Exit(code=1)

        if chosen_tool == "streme":
            cmd = [
                str(exe),
                "--dna",
                "--p",
                str(fasta_path),
                "--oc",
                str(run_dir),
                "--minw",
                str(resolved_minw),
                "--maxw",
                str(resolved_maxw),
                "--nmotifs",
                str(nmotifs),
            ]
            output_path = run_dir / "streme.txt"
        else:
            cmd = [
                str(exe),
                str(fasta_path),
                "-dna",
                "-oc",
                str(run_dir),
                "-minw",
                str(resolved_minw),
                "-maxw",
                str(resolved_maxw),
                "-nmotifs",
                str(nmotifs),
            ]
            if meme_mod:
                cmd.extend(["-mod", meme_mod])
            output_path = run_dir / "meme.txt"

        version = tool_version(Path(exe))
        try:
            _run_command(cmd, cwd=run_dir)
        except RuntimeError as exc:
            console.print(f"Error: {exc}")
            raise typer.Exit(code=1)
        if not output_path.exists():
            console.print(f"Error: expected MEME output at {output_path}")
            raise typer.Exit(code=1)

        _write_discover_manifest(
            run_dir=run_dir,
            config_path=config_path,
            tool=chosen_tool,
            exe=Path(exe),
            version=version,
            command=cmd,
            tf_name=target.tf_name,
            motif_source=resolved_source_id,
            window_sites=window_sites,
            site_window_lengths=site_window_lengths,
        )

        result = parse_meme_file(output_path)
        raw_payload = output_path.read_text()
        if replace_existing:
            removed = _purge_discovered_entries(
                catalog_root=catalog_root,
                catalog=catalog,
                source_id=resolved_source_id,
                tf_name=target.tf_name,
            )
            if removed:
                console.print(
                    f"INFO {target.tf_name}: replaced {removed} cached motif(s) in source '{resolved_source_id}'."
                )
        for motif in result.motifs[:nmotifs]:
            motif_id = f"{_safe_id(target.tf_name)}_{_safe_id(motif.motif_id)}"
            tags = {
                "matrix_source": chosen_tool,
                "discovery_tool": chosen_tool,
                "discovery_tool_path": str(exe),
                "discovery_tool_version": version or "unknown",
                "discovery_run": run_dir.name,
                "discovery_nsites": str(len(sequences)),
                "discovery_minw": str(resolved_minw),
                "discovery_maxw": str(resolved_maxw),
                "discovery_nmotifs": str(nmotifs),
            }
            record = build_motif_record(
                source=resolved_source_id,
                motif_id=motif_id,
                tf_name=target.tf_name,
                matrix=motif.prob_matrix,
                log_odds_matrix=motif.log_odds_matrix,
                matrix_semantics="probabilities",
                organism=_as_organism(target.entry),
                raw_payload=raw_payload,
                source_version=None,
                source_url=None,
                raw_artifact_paths=[str(output_path)],
                tags=tags,
                background=result.meta.background_freqs,
            )
            write_motif_record(catalog_root, record)
            catalog.upsert_from_record(record)
            table.add_row(
                target.tf_name,
                chosen_tool,
                f"{resolved_source_id}:{motif_id}",
                str(len(motif.prob_matrix)),
                render_path(output_path, base=base),
            )

    catalog.save(catalog_root)
    console.print(table)
