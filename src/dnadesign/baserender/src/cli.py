"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/cli.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import itertools
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Iterable, Optional, TypeVar

import typer
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
from .model import Guide
from .palette import Palette
from .presets.loader import load_job, resolve_job_path
from .render import render_figure
from .style import Style

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
        "Use `baserender job <name-or-path>` for preset-driven runs.\n"
        "Use `render` or `video` for ad-hoc, direct dataset runs."
    ),
)
console = Console()


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


def _read_csv_column(path: Path, column: str) -> list[str]:
    """Return values from a named column in a headered CSV."""
    import csv

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise typer.BadParameter(f"CSV '{path}' must contain column '{column}' (found: {reader.fieldnames or []})")
        out: list[str] = []
        for row in reader:
            raw = row.get(column)
            if raw is None:
                continue
            s = str(raw).strip()
            if s != "":
                out.append(s)
        if not out:
            raise typer.BadParameter(f"CSV '{path}' column '{column}' contains no non-blank values.")
        return out


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

    style = Style()
    pal = Palette(style.palette)

    with _progress() as progress:
        task = progress.add_task("Rendering images", total=n_total)
        progress.refresh()
        for i, rec in enumerate(recs):
            name = rec.id if rec.id else f"record_{i}"
            out_path = out_dir / f"{name}.{img_fmt}"
            render_figure(rec, style=style, palette=pal, out_path=str(out_path), fmt=img_fmt)
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
            report=report,
        )


# ---- preset-driven command ---------------------------------------------------


@app.command(
    "job",
    help=(
        "Run a preset job by *name* (looked up in jobs/) or by explicit YAML path.\n\n"
        "\b\n"
        "Default: render a video only (no per-record stills) into results/<job>/<job>.mp4.\n"
        "Use --rec-id or --row plus --fmt to render a single still instead."
    ),
)
def job_run(
    job: str = typer.Argument(..., help="Job name (in jobs/) or full YAML path."),
    rec_id: Optional[str] = typer.Option(None, "--rec-id", help="Render a single record by id (writes one still)."),
    row: Optional[int] = typer.Option(
        None,
        "--row",
        help="Render a single record by 0-based row index (writes one still).",
    ),
    fmt: str = typer.Option(
        "png",
        "--fmt",
        help="Image format when rendering a single record (png|pdf|svg).",
    ),
) -> None:
    """
    examples:

      baserender job CpxR_LexA
      baserender job CpxR_LexA --rec-id <some-id> --fmt pdf
      baserender job CpxR_LexA --row 0 --fmt png
    """
    job_path = resolve_job_path(job)
    cfg = load_job(job_path)
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
            for i, r in enumerate(_base_records()):
                if i == row:
                    target = r
                    break

        if target is None:
            console.print("[red]No matching record found.[/]")
            raise typer.Exit(code=2)

        out_dir = cfg.results_dir / cfg.name / "single"
        out_dir.mkdir(parents=True, exist_ok=True)
        leaf = rec_id if rec_id is not None else f"row_{row}"
        out_file = out_dir / f"{leaf}.{fmt}"

        style = Style.from_mapping(cfg.style)
        pal = Palette(style.palette)
        render_figure(target, style=style, palette=pal, out_path=str(out_file), fmt=fmt)
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
            from .io.parquet import read_parquet_records_by_ids, resolve_present_ids
            from .plugins.registry import load_plugins

            keys, overlays, used_overlay, overlay_by_key = _read_csv_keys_and_overlay(
                sel.path, sel.column, sel.overlay_column
            )
            if used_overlay:
                console.log(
                    f"Using overlay text from CSV column '{used_overlay}' ({sum(1 for x in overlays if x)} non-blank)"
                )
            # Assert id column is provided
            if not cfg.id_col:
                raise typer.BadParameter("selection.match_on=id requires an 'id' column in the input.columns.")
            # Presence pass (dataset membership only; no policy/gating)
            present = resolve_present_ids(cfg.input_path, id_col=cfg.id_col, ids=keys)
            missing = [k for k in keys if k not in present]
            if missing:
                msg = f"{len(missing)} id(s) from selection CSV are not present in the dataset."
                if sel.on_missing == "error":
                    raise typer.BadParameter(msg + f" Examples: {missing[:5]}")
                elif sel.on_missing == "warn":
                    console.print(f"[yellow]WARN[/] {msg} Examples: {missing[:5]}")
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
            for j, k in enumerate(keys):
                r = by_id.get(k)
                if r is None:
                    continue  # missing or dropped-by-policy (already reported)
                csv_label = overlays[j] if j < len(overlays) else None
                if csv_label:
                    # Replace any existing overlay_label to ensure CSV wins.
                    base_guides = [g for g in r.guides if getattr(g, "kind", "") != overlay_kind]
                    r = _dc_replace(r, guides=tuple(base_guides)).validate()
                    label = f"{csv_label}  id={r.id}"
                    r = r.with_extra(guides=[Guide(kind=overlay_kind, start=0, end=0, label=label)])
                else:
                    # If Parquet already supplied an overlay, keep it; otherwise add the fallback.
                    has_overlay = any(getattr(g, "kind", "") == overlay_kind and g.label for g in r.guides)
                    if not has_overlay:
                        label = f"sel_row={j}  id={r.id}"
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
            idxset = set(idxs)
            found = {}
            for i, r in enumerate(_base_records()):
                if i in idxset:
                    found[i] = r
                    if len(found) == len(idxset):
                        break
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
                if csv_label:
                    base_guides = [g for g in r.guides if getattr(g, "kind", "") != overlay_kind]
                    r = _dc_replace(r, guides=tuple(base_guides)).validate()
                    label = f"{csv_label}  id={r.id}"
                    r = r.with_extra(guides=[Guide(kind=overlay_kind, start=0, end=0, label=label)])
                else:
                    has_overlay = any(getattr(g, "kind", "") == overlay_kind and g.label for g in r.guides)
                    if not has_overlay:
                        label = f"row={i}  sel_row={j}"
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
                if csv_label:
                    base_guides = [g for g in r.guides if getattr(g, "kind", "") != overlay_kind]
                    r = _dc_replace(r, guides=tuple(base_guides)).validate()
                    label = f"{csv_label}  id={r.id}"
                    r = r.with_extra(guides=[Guide(kind=overlay_kind, start=0, end=0, label=label)])
                else:
                    has_overlay = any(getattr(g, "kind", "") == overlay_kind and g.label for g in r.guides)
                    if not has_overlay:
                        label = f"sel_row={j}"
                        r = r.with_extra(guides=[Guide(kind=overlay_kind, start=0, end=0, label=label)])
                recs_list.append(r)
    else:
        # No explicit selection: honor limit/sample_seed and annotate with dataset row index.
        if sel_limit is not None and cfg.sample_seed is not None:
            import random

            total_rows = _parquet_row_count(cfg.input_path) if cfg.format == "parquet" else None
            if total_rows is None:
                # Format guard — explicit (no fallback)
                raise typer.Exit(code=2)
            k = min(sel_limit, total_rows)
            rng = random.Random(int(cfg.sample_seed))
            idxs = sorted(rng.sample(range(total_rows), k))
            idxset = set(idxs)
            for i, r in enumerate(_base_records()):
                if i in idxset:
                    label = f"row={i}"
                    r = r.with_extra(guides=[Guide(kind=overlay_kind, start=0, end=0, label=label)])
                    recs_list.append(r)
        else:
            # Take first N (if limited) in order
            count = 0
            for i, r in enumerate(_base_records()):
                if sel_limit is not None and count >= sel_limit:
                    break
                label = f"row={i}"
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

        render_images(recs_list, out_dir=img_cfg.dir, fmt=img_cfg.fmt, style=cfg.style)

    out_video = cfg.video.out_path
    out_video.parent.mkdir(parents=True, exist_ok=True)

    style = Style.from_mapping(cfg.style)
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
            cfg = load_job(job_path)
            _pass(f"job resolved: {job_path}")
            if cfg.input_path.exists():
                _pass(f"dataset exists: {cfg.input_path}")
            else:
                _fail(f"dataset NOT found: {cfg.input_path}")
            results_root = cfg.results_dir
            try:
                results_root.mkdir(parents=True, exist_ok=True)
                _pass(f"results dir ready: {results_root}")
            except Exception as e:  # pragma: no cover
                _fail(f"cannot create results dir {results_root}: {e!r}")
            # Validate style mapping up front (assertive)
            from .style import Style as _S

            try:
                _ = _S.from_mapping(cfg.style)
                _pass("style config valid")
            except Exception as e:
                _fail(f"style config invalid: {e}")
            # Assert style mapping is valid (fail fast, clear message)
            try:
                _ = Style.from_mapping(cfg.style)
                _pass("style config valid")
            except Exception as e:
                _fail(f"style config invalid: {e}")
        except Exception as e:  # pragma: no cover
            _fail(f"Cannot load job '{job}': {e!r}")

    if not ok:
        raise typer.Exit(code=2)
