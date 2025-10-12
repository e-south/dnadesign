"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/api.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Mapping, Optional, Sequence

from .io.parquet import read_parquet_records
from .model import SeqRecord
from .palette import Palette
from .plugins.registry import PluginLike, load_plugins
from .render import render_figure
from .style import Style
from .video import render_video as _render_video


def read_records(
    path: Path,
    format: Literal["parquet"] = "parquet",
    *,
    sequence_col: str = "sequence",
    annotations_col: str = "densegen__used_tfbs_detail",
    id_col: Optional[str] = "id",
    alphabet: str = "DNA",
    plugins: Sequence[PluginLike] = (),
    ann_policy: Optional[Mapping[str, object]] = None,
) -> Iterable[SeqRecord]:
    if format != "parquet":
        raise ValueError("Only parquet format supported in v0.")
    base_iter = read_parquet_records(
        Path(path),
        sequence_col=sequence_col,
        annotations_col=annotations_col,
        id_col=id_col,
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
    style: Optional[Mapping[str, object]] = None,
):
    s = Style.from_mapping(style) if isinstance(style, Mapping) else (style or Style())
    pal = Palette(s.palette)
    fig = render_figure(
        record,
        style=s,
        palette=pal,
        out_path=str(out_path) if out_path else None,
        fmt=fmt,
    )
    return fig


def render_images(
    records: Iterable[SeqRecord],
    *,
    out_dir: Path,
    fmt: Literal["png", "svg", "pdf"] = "png",
    style: Optional[Mapping[str, object]] = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = Style.from_mapping(style) if isinstance(style, Mapping) else (style or Style())
    pal = Palette(s.palette)
    for i, rec in enumerate(records):
        name = rec.id if rec.id else f"record_{i}"
        render_figure(
            rec, style=s, palette=pal, out_path=str(out_dir / f"{name}.{fmt}"), fmt=fmt
        )


def render_video(
    records: Iterable[SeqRecord],
    *,
    out_path: Path,
    fps: int = 2,
    style: Optional[Mapping[str, object]] = None,
) -> None:
    s = Style.from_mapping(style) if isinstance(style, Mapping) else (style or Style())
    pal = Palette(s.palette)
    _render_video(records, out_path=Path(out_path), fps=fps, style=s, palette=pal)
