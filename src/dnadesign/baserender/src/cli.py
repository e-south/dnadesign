"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/cli.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import itertools
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, TypeVar

import typer
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .api import read_records
from .config.job_v2 import (
    find_redundant_overrides,
    job_to_minimal_dict,
    load_job_v2,
    strip_redundant_overrides,
)
from .contracts import BaseRenderError, SchemaError, ensure
from .job_runner import run_job
from .legend import legend_entries_for_record
from .model import Guide, SeqRecord
from .palette import Palette
from .presets.loader import load_job as load_job_v1
from .presets.loader import resolve_job_path
from .presets.style_presets import (
    effective_style_mapping,
    list_style_presets,
    resolve_style,
)
from .render import render_figure
from .utils import safe_stem, unique_stem

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Render biological sequences with k-mer annotations.\n\n"
        "\b\n"
        "Default project layout (monorepo):\n"
        "  jobs/      — YAML presets (sibling of src/)\n"
        "  results/   — outputs per job name (sibling of src/)\n"
        "  src/       — package code\n\n"
        "Use `baserender job run <name-or-path>` for preset-driven runs.\n"
        "Use `render` or `video` for ad-hoc, direct dataset runs."
    ),
)
console = Console()


# ---- style inspection commands ----------------------------------------------

style_app = typer.Typer(no_args_is_help=True, help="Inspect and print effective style presets.")
app.add_typer(style_app, name="style")


@style_app.command(
    "show",
    help=(
        "Print the effective style as YAML (default) or JSON.\n\n"
        "Examples:\n"
        "  baserender style show\n"
        "  baserender style show --preset presentation_default\n"
        "  baserender style show --job jobs/foo.yml\n"
    ),
)
def style_show(
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help="Optional preset name/path applied on top of presentation_default.",
    ),
    job: Optional[str] = typer.Option(
        None,
        "--job",
        help="Job name/path. If provided, shows the merged style for that job.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON instead of YAML."),
) -> None:
    import json as _json

    import yaml as _yaml

    if job:
        job_path = resolve_job_path(job)
        version = _detect_job_version(job_path)
        if version == 2:
            cfg = load_job_v2(job_path)
            mapping = effective_style_mapping(preset=cfg.style.preset, overrides=cfg.style.overrides)
        else:
            cfg = load_job_v1(job_path)
            mapping = effective_style_mapping(preset=cfg.style_preset, overrides=cfg.style)
    else:
        mapping = effective_style_mapping(preset=preset, overrides=None)

    if as_json:
        typer.echo(_json.dumps(mapping, indent=2, sort_keys=True))
    else:
        typer.echo(_yaml.safe_dump(mapping, sort_keys=False))


@style_app.command("list", help="List available style presets from styles/.")
def style_list() -> None:
    presets = list_style_presets()
    if not presets:
        typer.echo("No style presets found.")
        return
    for name in presets:
        typer.echo(name)


# ---- job v2 commands ---------------------------------------------------------

job_app = typer.Typer(no_args_is_help=True, help="Run and manage Job v2 configs.")
app.add_typer(job_app, name="job")


def _resolve_job_path_or_fail(spec: str) -> Path:
    try:
        return resolve_job_path(spec)
    except FileNotFoundError as e:
        raise typer.BadParameter(str(e)) from e


def _detect_job_version(path: Path) -> Optional[int]:
    data = yaml.safe_load(Path(path).read_text())
    if isinstance(data, dict):
        v = data.get("version", None)
        return int(v) if v is not None else None
    return None


def _run_job_v2(job_path: Path) -> None:
    job_cfg = load_job_v2(job_path)
    if job_cfg.output.video is None:
        report = run_job(job_cfg)
    else:
        with _progress() as progress:
            task = progress.add_task("Rendering frames", total=None)
            progress.refresh()

            def report_cb(event: str, payload: dict) -> None:
                if event == "prepare":
                    pass
                elif event == "start":
                    progress.update(task, total=payload.get("total_frames"))
                    console.log(
                        f"Writing MP4 {payload.get('width')}x{payload.get('height')} @ {payload.get('fps')} fps"
                    )
                elif event == "frame":
                    progress.advance(task)
                elif event == "finish":
                    progress.update(task, advance=0)
                    console.log(f"Wrote: {payload.get('out')}")

            report = run_job(job_cfg, report_cb=report_cb)
    if report.outputs:
        for k, v in report.outputs.items():
            console.log(f"{k}: {v}")


@job_app.callback(invoke_without_command=True)
def job_group(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        raise typer.Exit(code=0)


@job_app.command("run", help="Run a Job v2 config (video/images as configured).")
def job_run_cmd(
    job: str = typer.Argument(..., help="Job name (in jobs/) or YAML path."),
    allow_v1: bool = typer.Option(
        False,
        "--allow-v1",
        help="Allow running legacy v1 jobs (unsupported).",
    ),
) -> None:
    job_path = _resolve_job_path_or_fail(job)
    version = _detect_job_version(job_path)
    try:
        if version == 2:
            _run_job_v2(job_path)
        else:
            if not allow_v1:
                raise typer.BadParameter(
                    "Job v1 is no longer supported by default. Run `baserender job upgrade` or pass --allow-v1."
                )
            _run_job_v1(job=job)
    except BaseRenderError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(code=2)


@job_app.command("validate", help="Validate a Job v2 config (no rendering).")
def job_validate(job: str = typer.Argument(..., help="Job name (in jobs/) or YAML path.")) -> None:
    job_path = _resolve_job_path_or_fail(job)
    try:
        cfg = load_job_v2(job_path)
    except BaseRenderError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(code=2)
    console.log(f"job: {cfg.name}")
    console.log(f"input.path: {cfg.input.path}")
    console.log(f"output.results_dir: {cfg.output.results_dir}")
    console.log(f"style.preset: {cfg.style.preset}")
    console.log(f"plugins: {[p.name for p in cfg.pipeline.plugins]}")
    if cfg.selection:
        console.log(f"selection.match_on: {cfg.selection.match_on}")
    console.print("[green]OK[/]")


@job_app.command("lint", help="Lint a Job v2 config and report redundant style overrides.")
def job_lint(job: str = typer.Argument(..., help="Job name (in jobs/) or YAML path.")) -> None:
    job_path = _resolve_job_path_or_fail(job)
    try:
        cfg = load_job_v2(job_path)
    except BaseRenderError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(code=2)
    redundant = find_redundant_overrides(cfg.style.preset, cfg.style.overrides)
    if redundant:
        console.print("[yellow]Redundant style overrides[/]")
        for k in redundant:
            console.print(f"- {k}")
    else:
        console.print("[green]No redundant style overrides found.[/]")
    # Print a minimal YAML suggestion
    minimal = job_to_minimal_dict(cfg)
    if "style" in minimal and "overrides" in minimal["style"]:
        cleaned = strip_redundant_overrides(cfg.style.preset, cfg.style.overrides)
        if cleaned:
            minimal["style"]["overrides"] = cleaned
        else:
            minimal["style"].pop("overrides", None)
    typer.echo(yaml.safe_dump(minimal, sort_keys=False))


@job_app.command("normalize", help="Write a canonical minimal Job v2 YAML.")
def job_normalize(
    job: str = typer.Argument(..., help="Job name (in jobs/) or YAML path."),
    out: Path = typer.Option(..., "--out", help="Output path for normalized YAML."),
) -> None:
    job_path = _resolve_job_path_or_fail(job)
    try:
        cfg = load_job_v2(job_path)
    except BaseRenderError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(code=2)
    minimal = job_to_minimal_dict(cfg)
    if "style" in minimal and "overrides" in minimal["style"]:
        cleaned = strip_redundant_overrides(cfg.style.preset, cfg.style.overrides)
        if cleaned:
            minimal["style"]["overrides"] = cleaned
        else:
            minimal["style"].pop("overrides", None)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(minimal, sort_keys=False))
    console.print(f"[green]Wrote[/] {out}")


def _upgrade_v1_mapping(raw: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise SchemaError("Job YAML must be a mapping/dict")
    input_raw = raw.get("input")
    ensure(isinstance(input_raw, Mapping), "input section is required", SchemaError)
    input_m = input_raw
    columns = input_m.get("columns") or {}
    ensure(isinstance(columns, Mapping), "input.columns must be mapping", SchemaError)

    v2: dict[str, object] = {"version": 2}
    v2_input: dict[str, object] = {
        "path": input_m.get("path"),
        "format": input_m.get("format", "parquet"),
        "columns": {
            "sequence": columns.get("sequence"),
            "annotations": columns.get("annotations"),
        },
        "alphabet": input_m.get("alphabet", "DNA"),
    }
    if "id" in columns:
        v2_input["columns"]["id"] = columns.get("id")
    if "details" in columns:
        v2_input["columns"]["details"] = columns.get("details")

    limit = input_m.get("limit", None)
    sample_seed = input_m.get("sample_seed") or input_m.get("seed")
    if sample_seed is not None:
        n = int(limit) if limit is not None else 500
        v2_input["sample"] = {
            "mode": "random_rows",
            "n": int(n),
            "seed": int(sample_seed),
        }
    elif limit is not None:
        v2_input["limit"] = limit

    if "annotations" in input_m:
        v2_input["annotations"] = input_m.get("annotations")

    v2["input"] = v2_input

    plugins = raw.get("plugins", None)
    if plugins is not None:
        v2["pipeline"] = {"plugins": plugins}

    selection = raw.get("selection", None)
    if selection is not None:
        sel = dict(selection)
        if "csv" in sel and "path" not in sel:
            sel["path"] = sel.pop("csv")
        v2["selection"] = sel

    style_preset = raw.get("style_preset", None)
    style_overrides = raw.get("style", None)
    if style_preset is not None or style_overrides is not None:
        style_block: dict[str, object] = {}
        if style_preset is not None:
            style_block["preset"] = style_preset
        if style_overrides is not None:
            style_block["overrides"] = style_overrides
        v2["style"] = style_block

    output = raw.get("output", None)
    ensure(isinstance(output, Mapping), "output section is required", SchemaError)
    out_block: dict[str, object] = dict(output)
    video = out_block.get("video", None)
    if isinstance(video, Mapping):
        video = dict(video)
        if "seconds_per_seq" in video and "frames_per_record" not in video:
            fps = int(video.get("fps", 2))
            sec = float(video["seconds_per_seq"])
            frames = max(1, int(round(sec * fps)))
            video["frames_per_record"] = frames
            video.pop("seconds_per_seq", None)
        out_block["video"] = video
    v2["output"] = out_block
    return v2


@job_app.command("upgrade", help="Upgrade a v1 job YAML to Job v2.")
def job_upgrade(
    job: str = typer.Argument(..., help="Job name (in jobs/) or YAML path."),
    out: Path = typer.Option(..., "--out", help="Output path for v2 YAML."),
) -> None:
    job_path = _resolve_job_path_or_fail(job)
    raw = yaml.safe_load(Path(job_path).read_text())
    if isinstance(raw, Mapping) and raw.get("version", None) == 2:
        raise typer.BadParameter("Job is already version 2.")
    try:
        upgraded = _upgrade_v1_mapping(raw)
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(upgraded, sort_keys=False))
        console.print(f"[green]Wrote[/] {out}")
    except BaseRenderError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(code=2)


# ---- internal helpers --------------------------------------------------------

T = TypeVar("T")


def _split_plugins(spec: Optional[str]) -> tuple[str, ...]:
    if not spec:
        return ()
    return tuple(p.strip() for p in spec.replace(",", " ").split() if p.strip())


def _parquet_row_count(path: Path) -> int:
    from pyarrow import parquet as pq

    pf = pq.ParquetFile(path)
    return pf.metadata.num_rows


def _apply_limit(it: Iterable[T], limit: Optional[int]) -> Iterable[T]:
    # Iterables: limit>0 → islice; else passthrough
    if limit is None or limit <= 0:
        return it
    return itertools.islice(it, int(limit))


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def _read_csv_keys_and_overlay(
    path: Path,
    key_col: str,
    overlay_col: Optional[str],
):
    """
    Read a headered CSV and return:
      - keys: list[str]           (values from key_col, in file order, blanks skipped)
      - overlays: list[Optional[str]]  (aligned to keys; None when blank/missing)
      - used_overlay: Optional[str]    (the overlay column we actually used)
      - overlay_by_key: dict[str, str] (non-blank overlay by key; last one wins)
    If overlay_col is None, auto-detect a column literally named 'details' if present.
    """
    import csv

    keys: list[str] = []
    overlays: list[Optional[str]] = []
    overlay_by_key: dict[str, str] = {}
    used_overlay: Optional[str] = overlay_col
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or key_col not in reader.fieldnames:
            raise typer.BadParameter(f"CSV '{path}' must contain column '{key_col}' (found: {reader.fieldnames or []})")
        # Auto-detect 'details' if overlay not specified
        if used_overlay is None and "details" in reader.fieldnames:
            used_overlay = "details"
        # Warn if explicitly requested overlay is missing
        if overlay_col and overlay_col not in reader.fieldnames:
            console.print(
                f"[yellow]WARN[/] selection.overlay_column='{overlay_col}' not found in CSV. "
                f"Available columns: {reader.fieldnames}. Falling back to default behavior."
            )
            used_overlay = None
        for row in reader:
            raw_key = row.get(key_col)
            if raw_key is None:
                continue
            k = str(raw_key).strip()
            if k == "":
                continue
            keys.append(k)
            text: Optional[str] = None
            if used_overlay:
                txt_raw = row.get(used_overlay)
                if txt_raw is not None:
                    s = str(txt_raw).strip()
                    if s != "":
                        text = s
                        overlay_by_key[k] = s  # last non-blank wins
            overlays.append(text)
    if not keys:
        raise typer.BadParameter(f"CSV '{path}' column '{key_col}' contains no non-blank values.")
    return keys, overlays, used_overlay, overlay_by_key


def _row_label(rec, fallback: Optional[int] = None) -> Optional[str]:
    ri = getattr(rec, "row_index", None)
    if ri is not None:
        return f"row={ri}"
    if fallback is not None:
        return f"row={fallback}"
    return None


def _join_label(*parts: Optional[str]) -> str:
    return "  ".join([p for p in parts if p])


def _select_records_by_row_index(records: Iterable["SeqRecord"], idxs: Sequence[int]) -> dict[int, "SeqRecord"]:
    idxset = set(idxs)
    found: dict[int, SeqRecord] = {}
    for r in records:
        ri = getattr(r, "row_index", None)
        if ri is None:
            raise typer.BadParameter("Row selection requires row_index (Parquet input only).")
        if ri in idxset:
            found[ri] = r
            if len(found) == len(idxset):
                break
    return found


def _sample_records_by_row_index(
    records: Iterable["SeqRecord"], *, total_rows: int, k: int, seed: int
) -> tuple[list["SeqRecord"], list[int]]:
    import random

    rng = random.Random(int(seed))
    idxs = sorted(rng.sample(range(total_rows), k))
    found = _select_records_by_row_index(records, idxs)
    ordered = [found[i] for i in idxs if i in found]
    return ordered, idxs


# ---- direct (ad-hoc) commands ------------------------------------------------


@app.command(
    help=(
        "Render per-record images directly from a Parquet dataset (no preset).\n\n"
        "\b\n"
        "This writes one image per record. For large datasets this can create many files—\n"
        "prefer `baserender job` (video-only) for exploratory runs."
    )
)
def render(
    path: Path = typer.Argument(..., help="Path to Parquet dataset."),
    in_fmt: str = typer.Option("parquet", "--format", help="Input format (parquet)."),
    seq_col: str = typer.Option("sequence", help="Sequence column name."),
    ann_col: str = typer.Option("densegen__used_tfbs_detail", help="Annotations column name."),
    id_col: Optional[str] = typer.Option("id", help="Record ID column."),
    alphabet: str = typer.Option("DNA", help="Alphabet (DNA|RNA|PROTEIN)."),
    out_dir: Path = typer.Option(..., "--out-dir", help="Output directory for images."),
    img_fmt: str = typer.Option("png", "--img-fmt", help="Image format (png|pdf|svg)."),
    plugin: Optional[str] = typer.Option(
        None,
        "--plugin",
        "-p",
        help="Plugins to apply (comma/space-separated), e.g. -p 'sigma70 pkg.mod:Class'",
    ),
    limit: int = typer.Option(500, "--limit", help="Max records to process (default 500). Use 0 for all."),
) -> None:
    console.rule("[bold]Render images")
    out_dir.mkdir(parents=True, exist_ok=True)
    total_rows = _parquet_row_count(path) if in_fmt == "parquet" else None
    n_total = min(total_rows, limit) if (total_rows is not None and limit > 0) else (total_rows or None)
    console.log(f"Dataset: {path}")
    if total_rows is not None:
        console.log(f"Rows in dataset: {total_rows}")
    if limit and limit > 0:
        console.log(f"Limit: {limit}")

    recs = read_records(
        path,
        format=in_fmt,
        sequence_col=seq_col,
        annotations_col=ann_col,
        id_col=id_col,
        alphabet=alphabet,
        plugins=_split_plugins(plugin),
    )
    recs = _apply_limit(recs, None if limit <= 0 else limit)

    style = resolve_style()
    pal = Palette(style.palette)
    used: set[str] = set()

    with _progress() as progress:
        task = progress.add_task("Rendering images", total=n_total)
        progress.refresh()
        for i, rec in enumerate(recs):
            base = (
                rec.id
                if rec.id
                else (f"row_{rec.row_index}" if getattr(rec, "row_index", None) is not None else f"record_{i}")
            )
            name = unique_stem(safe_stem(base), used)
            out_path = out_dir / f"{name}.{img_fmt}"
            fig = render_figure(
                rec,
                style=style,
                palette=pal,
                out_path=str(out_path),
                fmt=img_fmt,
                legend_entries=legend_entries_for_record(rec),
            )
            import matplotlib.pyplot as plt

            plt.close(fig)
            progress.advance(task)
    console.print(f"[green]Wrote images to {out_dir}[/]")


@app.command(
    help=(
        "Render a single video directly from a Parquet dataset (no preset).\n\n"
        "\b\n"
        "This mirrors the preset `job` command but without YAML. For richer control\n"
        "(pauses, sizing, duration), prefer `baserender job`."
    )
)
def video(
    path: Path = typer.Argument(..., help="Path to Parquet dataset."),
    out: Path = typer.Option(..., "--out", help="Path to output video (mp4)."),
    fps: int = typer.Option(2, help="Frames per second."),
    in_fmt: str = typer.Option("parquet", "--format", help="Input format (parquet)."),
    seq_col: str = typer.Option("sequence", help="Sequence column name."),
    ann_col: str = typer.Option("densegen__used_tfbs_detail", help="Annotations column name."),
    id_col: Optional[str] = typer.Option("id", help="Record ID column."),
    alphabet: str = typer.Option("DNA", help="Alphabet (DNA|RNA|PROTEIN)."),
    plugin: Optional[str] = typer.Option(
        None,
        "--plugin",
        "-p",
        help="Plugins to apply (comma/space-separated), e.g. -p 'sigma70 pkg.mod:Class'",
    ),
    limit: int = typer.Option(500, "--limit", help="Max records to process (default 500). Use 0 for all."),
) -> None:
    console.rule("[bold]Render video")
    console.log(f"Dataset: {path}")
    total_rows = _parquet_row_count(path) if in_fmt == "parquet" else None
    if total_rows is not None:
        console.log(f"Rows in dataset: {total_rows}")
    if limit and limit > 0:
        console.log(f"Limit: {limit}")

    recs = read_records(
        path,
        format=in_fmt,
        sequence_col=seq_col,
        annotations_col=ann_col,
        id_col=id_col,
        alphabet=alphabet,
        plugins=_split_plugins(plugin),
    )
    recs = _apply_limit(recs, None if limit <= 0 else limit)

    from .video import render_video as _render_video

    style = resolve_style()
    pal = Palette(style.palette)

    with _progress() as progress:
        task = progress.add_task("Rendering frames", total=None)
        progress.refresh()

        def report(event: str, payload: dict) -> None:
            if event == "prepare":
                # keep spinner alive; nothing to update numerically yet
                pass
            elif event == "start":
                progress.update(task, total=payload.get("total_frames"))
                console.log(f"Writing MP4 {payload.get('width')}x{payload.get('height')} @ {payload.get('fps')} fps")
            elif event == "frame":
                progress.advance(task)
            elif event == "finish":
                progress.update(task, advance=0)
                console.log(f"Wrote: {payload.get('out')}")

        _render_video(
            recs,
            out_path=out,
            fps=fps,
            style=style,
            palette=pal,
            report=report,
        )


# ---- preset-driven v1 runner (legacy, opt-in) --------------------------------


def _run_job_v1(
    job: str,
    rec_id: Optional[str] = None,
    row: Optional[int] = None,
    fmt: str = "png",
) -> None:
    """
    examples:

      baserender job CpxR_LexA
      baserender job CpxR_LexA --rec-id <some-id> --fmt pdf
      baserender job CpxR_LexA --row 0 --fmt png
    """
    job_path = resolve_job_path(job)
    cfg = load_job_v1(job_path)
    console.rule(f"[bold]Job: {cfg.name}")
    console.log(f"Dataset: {cfg.input_path}")
    if cfg.limit:
        console.log(f"Limit: {cfg.limit}")

    # Single-record still (non-default): do not limit selection
    sel_limit = None if (rec_id is not None or row is not None) else cfg.limit

    # Lazily construct the base record stream (no pre-materialization).
    def _base_records():
        return read_records(
            cfg.input_path,
            format=cfg.format,
            sequence_col=cfg.seq_col,
            annotations_col=cfg.ann_col,
            id_col=cfg.id_col,
            details_col=cfg.details_col,
            alphabet=cfg.alphabet,
            plugins=tuple(cfg.plugins),
            ann_policy=cfg.ann_policy,
        )

    # ---- single still path
    if rec_id is not None or row is not None:
        target = None
        if rec_id is not None:
            for r in _base_records():
                if r.id == rec_id:
                    target = r
                    break
        else:
            for r in _base_records():
                ri = getattr(r, "row_index", None)
                if ri is None:
                    raise typer.BadParameter("Row selection requires row_index (Parquet input only).")
                if ri == row:
                    target = r
                    break

        if target is None:
            console.print("[red]No matching record found.[/]")
            raise typer.Exit(code=2)

        out_dir = cfg.results_dir / cfg.name / "single"
        out_dir.mkdir(parents=True, exist_ok=True)
        leaf_raw = rec_id if rec_id is not None else f"row_{row}"
        leaf = safe_stem(leaf_raw)
        out_file = out_dir / f"{leaf}.{fmt}"

        style = resolve_style(preset=cfg.style_preset, overrides=cfg.style)
        pal = Palette(style.palette)
        fig = render_figure(
            target,
            style=style,
            palette=pal,
            out_path=str(out_file),
            fmt=fmt,
            legend_entries=legend_entries_for_record(target),
        )
        import matplotlib.pyplot as plt

        plt.close(fig)
        console.print(f"[green]Wrote still to {out_file}[/]")
        return

    # ---- video + (optional) stills path
    from .video import render_video as _render_video

    # Build the final list of records to render, optionally honoring explicit selection.
    recs_list = []  # type: list
    overlay_kind = "overlay_label"

    if getattr(cfg, "selection", None) is not None:
        sel = cfg.selection  # type: ignore[attr-defined]
        console.log(
            f"Selection CSV: {sel.path}  (match_on={sel.match_on}, column={sel.column}"
            + (f", overlay_column={sel.overlay_column}" if sel.overlay_column else "")
            + ")"
        )
        # Fast path for 'id' on parquet: use Arrow dataset filter to avoid full-table scans.
        if sel.match_on == "id" and cfg.format == "parquet" and cfg.id_col:
            from .io.parquet import (
                canonicalize_id_strings_for_parquet,
                read_parquet_records_by_ids,
                resolve_present_ids,
            )
            from .plugins.registry import load_plugins

            raw_keys, overlays, used_overlay, overlay_by_key = _read_csv_keys_and_overlay(
                sel.path, sel.column, sel.overlay_column
            )
            canonical_keys, _ = canonicalize_id_strings_for_parquet(cfg.input_path, id_col=cfg.id_col, raw_ids=raw_keys)
            raw_to_canonical = dict(zip(raw_keys, canonical_keys))
            overlay_by_key = {raw_to_canonical.get(k, k): v for k, v in overlay_by_key.items()}
            if used_overlay:
                console.log(
                    f"Using overlay text from CSV column '{used_overlay}' ({sum(1 for x in overlays if x)} non-blank)"
                )
            # Assert id column is provided
            if not cfg.id_col:
                raise typer.BadParameter("selection.match_on=id requires an 'id' column in the input.columns.")
            # Presence pass (dataset membership only; no policy/gating)
            present = resolve_present_ids(cfg.input_path, id_col=cfg.id_col, ids=canonical_keys)
            missing_pairs = [(raw, canon) for raw, canon in zip(raw_keys, canonical_keys) if canon not in present]
            if missing_pairs:
                msg = f"{len(missing_pairs)} id(s) from selection CSV are not present in the dataset."
                examples = []
                for raw, canon in missing_pairs[:5]:
                    if raw != canon:
                        examples.append(f"{raw} -> {canon}")
                    else:
                        examples.append(raw)
                if sel.on_missing == "error":
                    raise typer.BadParameter(msg + f" Examples: {examples}")
                elif sel.on_missing == "warn":
                    console.print(f"[yellow]WARN[/] {msg} Examples: {examples}")
            # Load only present ids (dedup for efficiency, keep original order later)
            key_set = set(present)
            base_iter = read_parquet_records_by_ids(
                cfg.input_path,
                ids=key_set,
                sequence_col=cfg.seq_col,
                annotations_col=cfg.ann_col,
                details_col=cfg.details_col,
                id_col=cfg.id_col,
                alphabet=cfg.alphabet,
                ann_policy=cfg.ann_policy,
            )
            # Apply plugins (same as read_records)
            plugins = load_plugins(tuple(cfg.plugins))

            def _apply_plugins():
                for rec in base_iter:
                    for p in plugins:
                        rec = p.apply(rec)
                    yield rec

            # Map by id for quick lookup
            by_id = {r.id: r for r in _apply_plugins()}
            # Policy drop accounting (present → parsed record)
            dropped_by_policy = sorted(list(set(present) - set(by_id.keys())))
            if dropped_by_policy:
                console.print(
                    f"[yellow]WARN[/] {len(dropped_by_policy)} id(s) exist in the dataset but were dropped by "
                    f"annotation policy (e.g., require_non_empty, ambiguous=drop). Examples: {dropped_by_policy[:5]}"
                )
            for j, k in enumerate(canonical_keys):
                r = by_id.get(k)
                if r is None:
                    continue  # missing or dropped-by-policy (already reported)
                csv_label = overlays[j] if j < len(overlays) else None
                row_txt = _row_label(r)
                if csv_label:
                    # Replace any existing overlay_label to ensure CSV wins.
                    base_guides = [g for g in r.guides if getattr(g, "kind", "") != overlay_kind]
                    r = _dc_replace(r, guides=tuple(base_guides)).validate()
                    label = _join_label(csv_label, row_txt, f"id={r.id}")
                    r = r.with_extra(guides=[Guide(kind=overlay_kind, start=0, end=0, label=label)])
                else:
                    # If Parquet already supplied an overlay, keep it; otherwise add the fallback.
                    has_overlay = any(getattr(g, "kind", "") == overlay_kind and g.label for g in r.guides)
                    if not has_overlay:
                        label = _join_label(f"sel_row={j}", row_txt, f"id={r.id}")
                        r = r.with_extra(guides=[Guide(kind=overlay_kind, start=0, end=0, label=label)])
                recs_list.append(r)
            if not recs_list:
                console.print(
                    "[red]No selected records survived policy gating and/or were missing.[/] "
                    "Check input.annotations settings (require_non_empty, min_per_record, ambiguous)."
                )
                raise typer.Exit(code=2)
            # ignore limit/sample_seed when selection is explicit
        elif sel.match_on == "row":
            # Interpret CSV values as 0-based row indices into the dataset.
            idx_vals, overlays, used_overlay, overlay_by_key = _read_csv_keys_and_overlay(
                sel.path, sel.column, sel.overlay_column
            )
            if used_overlay:
                console.log(
                    f"Using overlay text from CSV column '{used_overlay}' ({sum(1 for x in overlays if x)} non-blank)"
                )
            try:
                idxs = [int(x) for x in idx_vals]
            except Exception:
                raise typer.BadParameter("Row indices in selection CSV must be integers (0-based).")
            found = _select_records_by_row_index(_base_records(), idxs)
            missing = [i for i in idxs if i not in found]
            if missing:
                msg = f"{len(missing)} row index/indices not present in dataset."
                if sel.on_missing == "error":
                    raise typer.BadParameter(msg + f" Examples: {missing[:5]}")
                elif sel.on_missing == "warn":
                    console.print(f"[yellow]WARN[/] {msg} Examples: {missing[:5]}")
            # CSV order if requested
            ordered = idxs if sel.keep_order else sorted(found.keys())
            for j, i in enumerate(ordered):
                r = found.get(i)
                if r is None:
                    continue
                csv_label = overlays[j] if sel.keep_order else overlay_by_key.get(str(i))
                row_txt = _row_label(r)
                if csv_label:
                    base_guides = [g for g in r.guides if getattr(g, "kind", "") != overlay_kind]
                    r = _dc_replace(r, guides=tuple(base_guides)).validate()
                    label = _join_label(csv_label, row_txt, f"id={r.id}")
                    r = r.with_extra(guides=[Guide(kind=overlay_kind, start=0, end=0, label=label)])
                else:
                    has_overlay = any(getattr(g, "kind", "") == overlay_kind and g.label for g in r.guides)
                    if not has_overlay:
                        label = _join_label(f"sel_row={j}", row_txt, f"id={r.id}")
                        r = r.with_extra(guides=[Guide(kind=overlay_kind, start=0, end=0, label=label)])
                recs_list.append(r)
        else:
            # Fallback: match_on 'sequence' (or 'id' without fast path) by scanning the dataset.
            keys, overlays, used_overlay, overlay_by_key = _read_csv_keys_and_overlay(
                sel.path, sel.column, sel.overlay_column
            )
            if used_overlay:
                console.log(
                    f"Using overlay text from CSV column '{used_overlay}' ({sum(1 for x in overlays if x)} non-blank)"
                )
            key_attr = "sequence" if sel.match_on == "sequence" else "id"
            want = set(keys) if not sel.keep_order else None
            found_map = {}
            for i, r in enumerate(_base_records()):
                k = getattr(r, key_attr)
                if sel.keep_order:
                    if k in found_map:
                        continue
                    if k in keys:
                        found_map[k] = r
                        if len(found_map) == len(keys):
                            break
                else:
                    if k in want:
                        found_map[k] = r
                        if len(found_map) == len(want):
                            break
            missing = [k for k in keys if k not in found_map]
            if missing:
                msg = f"{len(missing)} value(s) from selection CSV not found."
                if sel.on_missing == "error":
                    raise typer.BadParameter(msg + f" Examples: {missing[:5]}")
                elif sel.on_missing == "warn":
                    console.print(f"[yellow]WARN[/] {msg} Examples: {missing[:5]}")
            ordered = keys if sel.keep_order else sorted(found_map.keys())
            for j, k in enumerate(ordered):
                r = found_map.get(k)
                if r is None:
                    continue
                csv_label = overlays[j] if sel.keep_order else overlay_by_key.get(k)
                row_txt = _row_label(r)
                if csv_label:
                    base_guides = [g for g in r.guides if getattr(g, "kind", "") != overlay_kind]
                    r = _dc_replace(r, guides=tuple(base_guides)).validate()
                    label = _join_label(csv_label, row_txt, f"id={r.id}")
                    r = r.with_extra(guides=[Guide(kind=overlay_kind, start=0, end=0, label=label)])
                else:
                    has_overlay = any(getattr(g, "kind", "") == overlay_kind and g.label for g in r.guides)
                    if not has_overlay:
                        label = _join_label(f"sel_row={j}", row_txt, f"id={r.id}")
                        r = r.with_extra(guides=[Guide(kind=overlay_kind, start=0, end=0, label=label)])
                recs_list.append(r)
    else:
        # No explicit selection: honor limit/sample_seed and annotate with dataset row index.
        if sel_limit is not None and cfg.sample_seed is not None:
            total_rows = _parquet_row_count(cfg.input_path) if cfg.format == "parquet" else None
            if total_rows is None:
                # Format guard — explicit (no fallback)
                raise typer.Exit(code=2)
            k = min(sel_limit, total_rows)
            sampled, _idxs = _sample_records_by_row_index(
                _base_records(), total_rows=total_rows, k=k, seed=int(cfg.sample_seed)
            )
            for r in sampled:
                label = _row_label(r) or f"row={r.row_index}"
                r = r.with_extra(guides=[Guide(kind=overlay_kind, start=0, end=0, label=label)])
                recs_list.append(r)
        else:
            # Take first N (if limited) in order
            count = 0
            for i, r in enumerate(_base_records()):
                if sel_limit is not None and count >= sel_limit:
                    break
                row_txt = _row_label(r, fallback=i)
                label = row_txt if row_txt is not None else f"row={i}"
                r = r.with_extra(guides=[Guide(kind=overlay_kind, start=0, end=0, label=label)])
                recs_list.append(r)
                count += 1

    # Optional stills export (honor images block if present)
    if getattr(cfg, "images", None) is not None and rec_id is None and row is None:
        img_cfg = cfg.images  # type: ignore[attr-defined]
        img_cfg.dir.parent.mkdir(parents=True, exist_ok=True)
        img_cfg.dir.mkdir(parents=True, exist_ok=True)
        console.log(f"Writing stills to {img_cfg.dir} ({img_cfg.fmt})")
        from .api import render_images

        style_for_images = resolve_style(preset=cfg.style_preset, overrides=cfg.style)
        render_images(recs_list, out_dir=img_cfg.dir, fmt=img_cfg.fmt, style=style_for_images)

    out_video = cfg.video.out_path
    out_video.parent.mkdir(parents=True, exist_ok=True)

    style = resolve_style(preset=cfg.style_preset, overrides=cfg.style)
    pal = Palette(style.palette)

    with _progress() as progress:
        task = progress.add_task("Rendering frames", total=None)
        # Force an immediate draw so the bar is visible even while the video layer
        # pre-enumerates records and sizes.
        progress.refresh()

        def report(event: str, payload: dict) -> None:
            if event == "prepare":
                # keep spinner alive during enumeration/sizing
                pass
            elif event == "start":
                progress.update(task, total=payload.get("total_frames"))
                console.log(f"Writing MP4 {payload.get('width')}x{payload.get('height')} @ {payload.get('fps')} fps")
            elif event == "frame":
                progress.advance(task)
            elif event == "finish":
                progress.update(task, advance=0)
                console.log(f"Wrote: {payload.get('out')}")

        _render_video(
            recs_list,
            out_path=out_video,
            fps=cfg.video.fps,
            style=style,
            palette=pal,
            fmt=cfg.video.fmt,
            frames_per_record=cfg.video.frames_per_record,
            pauses=cfg.video.pauses,
            width_px=cfg.video.width_px,
            height_px=cfg.video.height_px,
            aspect_ratio=cfg.video.aspect_ratio,
            total_duration=cfg.video.total_duration,
            report=report,
        )
    console.print(f"[green]Wrote video to {out_video}[/]")


# ---- environment preflight ---------------------------------------------------


@app.command(
    help=(
        "Check environment and (optionally) a job file for readiness.\n\n"
        "\b\n"
        "Validates: Python importability (pyarrow, matplotlib), ffmpeg availability,\n"
        "job YAML readability, dataset path existence, and results directory."
    )
)
def doctor(job: Optional[str] = typer.Argument(None, help="Optional job name/path to verify.")) -> None:
    ok = True

    def _pass(msg: str) -> None:
        console.print(f"[green]PASS[/] {msg}")

    def _fail(msg: str) -> None:
        nonlocal ok
        ok = False
        console.print(f"[red]FAIL[/] {msg}")

    # Python deps
    try:
        import pyarrow  # noqa: F401

        _pass("pyarrow importable")
    except Exception as e:  # pragma: no cover
        _fail(f"pyarrow not importable: {e!r}")

    try:
        import matplotlib  # noqa: F401
        from matplotlib import animation

        if animation.writers.is_available("ffmpeg"):
            _pass("matplotlib + ffmpeg writer available")
        else:
            _fail("matplotlib found, but ffmpeg writer is NOT available")
    except Exception as e:  # pragma: no cover
        _fail(f"matplotlib not importable or misconfigured: {e!r}")

    if job:
        try:
            job_path = resolve_job_path(job)
            version = _detect_job_version(job_path)
            if version == 2:
                cfg = load_job_v2(job_path)
            else:
                cfg = load_job_v1(job_path)
            _pass(f"job resolved: {job_path}")
            input_path = cfg.input.path if hasattr(cfg, "input") else cfg.input_path
            if input_path.exists():
                _pass(f"dataset exists: {input_path}")
            else:
                _fail(f"dataset NOT found: {input_path}")
            results_root = cfg.output.results_dir if hasattr(cfg, "output") else cfg.results_dir
            try:
                results_root.mkdir(parents=True, exist_ok=True)
                _pass(f"results dir ready: {results_root}")
            except Exception as e:  # pragma: no cover
                _fail(f"cannot create results dir {results_root}: {e!r}")
            try:
                if hasattr(cfg, "style"):
                    _ = resolve_style(preset=cfg.style.preset, overrides=cfg.style.overrides)
                else:
                    _ = resolve_style(preset=cfg.style_preset, overrides=cfg.style)
                _pass("style config valid (merged presets + overrides)")
            except Exception as e:
                _fail(f"style config invalid: {e}")
        except Exception as e:  # pragma: no cover
            _fail(f"Cannot load job '{job}': {e!r}")

    if not ok:
        raise typer.Exit(code=2)
