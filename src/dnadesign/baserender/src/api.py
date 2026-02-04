"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/api.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Mapping, Optional, Sequence

from .io.parquet import read_parquet_records
from .legend import legend_entries_for_record
from .model import SeqRecord
from .palette import Palette
from .plugins.registry import PluginLike, load_plugins
from .presets.style_presets import PresetSpec, resolve_style
from .render import render_figure
from .style import Style
from .utils import safe_stem, unique_stem
from .video import render_video as _render_video


def _coerce_style(
    style: Style | Mapping[str, object] | None,
    *,
    preset: Optional[PresetSpec] = None,
) -> Style:
    if style is None:
        return resolve_style(preset=preset)
    if isinstance(style, Style):
        return style
    if isinstance(style, Mapping):
        return resolve_style(preset=preset, overrides=style)
    raise TypeError("style must be a Style, a mapping, or None")


def read_records(
    path: Path,
    format: Literal["parquet"] = "parquet",
    *,
    sequence_col: str = "sequence",
    annotations_col: str = "densegen__used_tfbs_detail",
    id_col: Optional[str] = None,
    details_col: Optional[str] = None,
    alphabet: str = "DNA",
    plugins: Sequence[PluginLike] = (),
    ann_policy: Optional[Mapping[str, object]] = None,
) -> Iterable[SeqRecord]:
    if format != "parquet":
        from .contracts import SchemaError

        raise SchemaError("Only parquet format supported in v0.")
    base_iter = read_parquet_records(
        Path(path),
        sequence_col=sequence_col,
        annotations_col=annotations_col,
        id_col=id_col,
        details_col=details_col,
        alphabet=alphabet,
        ann_policy=ann_policy,
    )
    if not plugins:
        return base_iter
    plugin_instances = load_plugins(plugins)

    def _apply():
        for rec in base_iter:
            for p in plugin_instances:
                rec = p.apply(rec)
            yield rec

    return _apply()


def render_image(
    record: SeqRecord,
    *,
    out_path: Optional[Path] = None,
    fmt: Literal["png", "svg", "pdf"] = "png",
    style: Style | Mapping[str, object] | None = None,
):
    s = _coerce_style(style)
    pal = Palette(s.palette)
    # Build a legend for still images too (parity with video frames)
    legend_entries = legend_entries_for_record(record)
    fig = render_figure(
        record,
        style=s,
        palette=pal,
        out_path=str(out_path) if out_path else None,
        fmt=fmt,
        legend_entries=legend_entries,
    )
    return fig


def render_images(
    records: Iterable[SeqRecord],
    *,
    out_dir: Path,
    fmt: Literal["png", "svg", "pdf"] = "png",
    style: Style | Mapping[str, object] | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = _coerce_style(style)
    pal = Palette(s.palette)
    used: set[str] = set()
    for i, rec in enumerate(records):
        base = (
            rec.id
            if rec.id
            else (f"row_{rec.row_index}" if getattr(rec, "row_index", None) is not None else f"record_{i}")
        )
        name = unique_stem(safe_stem(base), used)
        # Per-record legend
        legend_entries = legend_entries_for_record(rec)
        fig = render_figure(
            rec,
            style=s,
            palette=pal,
            out_path=str(out_dir / f"{name}.{fmt}"),
            fmt=fmt,
            legend_entries=legend_entries,
        )
        import matplotlib.pyplot as plt

        plt.close(fig)


def render_video(
    records: Iterable[SeqRecord],
    *,
    out_path: Path,
    fps: int = 2,
    style: Style | Mapping[str, object] | None = None,
) -> None:
    s = _coerce_style(style)
    pal = Palette(s.palette)
    _render_video(records, out_path=Path(out_path), fps=fps, style=s, palette=pal)
