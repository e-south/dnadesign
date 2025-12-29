"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/job_runner.py

Run reporting for baserender jobs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

from .config.job_v2 import JobV2
from .contracts import SchemaError
from .io.parquet import (
    canonicalize_id_strings_for_parquet,
    read_parquet_records,
    read_parquet_records_by_ids,
    resolve_present_ids,
)
from .legend import legend_entries_for_record
from .model import SeqRecord
from .palette import Palette
from .plugins.registry import load_plugins
from .presets.style_presets import resolve_style
from .render import render_figure
from .reporting import RunReport, style_overrides_hash
from .selection import apply_overlay_label, read_selection_csv
from .utils import safe_stem, unique_stem
from .video import render_video

ReportFn = Callable[[str, dict], None]


def _join_label(*parts: Optional[str]) -> str:
    return "  ".join([p for p in parts if p])


def _row_label(rec: SeqRecord) -> Optional[str]:
    ri = getattr(rec, "row_index", None)
    return f"row={ri}" if ri is not None else None


def _parquet_row_count(path: Path) -> int:
    from pyarrow import parquet as pq

    pf = pq.ParquetFile(path)
    return pf.metadata.num_rows


def _base_iter(job: JobV2, *, row_reporter=None) -> Iterable[SeqRecord]:
    return read_parquet_records(
        job.input.path,
        sequence_col=job.input.columns.sequence,
        annotations_col=job.input.columns.annotations,
        id_col=job.input.columns.id,
        details_col=job.input.columns.details,
        alphabet=job.input.alphabet,
        ann_policy={
            "ambiguous": job.input.annotations.ambiguous,
            "offset_mode": job.input.annotations.offset_mode,
            "zero_as_unspecified": job.input.annotations.zero_as_unspecified,
            "on_missing_kmer": job.input.annotations.on_missing_kmer,
            "require_non_empty": job.input.annotations.require_non_empty,
            "min_per_record": job.input.annotations.min_per_record,
            "require_non_null_cols": list(job.input.annotations.require_non_null_cols),
            "on_invalid_row": job.input.annotations.on_invalid_row,
        },
        row_reporter=row_reporter,
    )


def _apply_plugins(records: Iterable[SeqRecord], plugins) -> Iterable[SeqRecord]:
    for r in records:
        for p in plugins:
            r = p.apply(r)
        yield r


def _has_overlay(rec: SeqRecord) -> bool:
    return any(getattr(g, "kind", "") == "overlay_label" and g.label for g in rec.guides)


def _overlay_default_label(rec: SeqRecord, *, sel_row: int) -> str:
    return _join_label(f"sel_row={sel_row}", _row_label(rec), f"id={rec.id}")


def _select_by_row_index(records: Iterable[SeqRecord], idxs: list[int]) -> dict[int, SeqRecord]:
    idxset = set(idxs)
    found: dict[int, SeqRecord] = {}
    for r in records:
        ri = getattr(r, "row_index", None)
        if ri is None:
            raise SchemaError("Row selection requires row_index (Parquet input only).")
        if ri in idxset:
            found[ri] = r
            if len(found) == len(idxset):
                break
    return found


def run_job(job: JobV2, *, report_cb: Optional[ReportFn] = None) -> RunReport:
    report = RunReport(
        job_name=job.name,
        input_path=str(job.input.path),
        selection_path=str(job.selection.path) if job.selection else None,
        style_preset=str(job.style.preset),
        style_overrides_hash=style_overrides_hash(job.style.overrides),
    )

    def row_reporter(reason: str, *_):
        if reason == "seen":
            report.total_rows_seen += 1
        else:
            report.note_skip_row(reason)

    style = resolve_style(preset=job.style.preset, overrides=job.style.overrides)
    palette = Palette(style.palette)
    plugins = load_plugins(job.pipeline.plugins)

    records: list[SeqRecord] = []

    # ---- selection logic
    if job.selection is not None:
        sel = job.selection
        spec = read_selection_csv(sel.path, key_col=sel.column, overlay_col=sel.overlay_column)
        keys = spec.keys
        overlays = spec.overlays

        if sel.match_on == "id":
            if not job.input.columns.id:
                raise SchemaError("selection.match_on=id requires input.columns.id to be set.")
            raw_keys = keys
            canonical_keys, _ = canonicalize_id_strings_for_parquet(
                job.input.path, id_col=job.input.columns.id, raw_ids=raw_keys
            )
            raw_to_canon = dict(zip(raw_keys, canonical_keys))
            overlay_by_key = {raw_to_canon.get(k, k): v for k, v in spec.overlay_by_key.items()}
            present = resolve_present_ids(job.input.path, id_col=job.input.columns.id, ids=canonical_keys)
            missing_raw = [raw for raw, canon in zip(raw_keys, canonical_keys) if canon not in present]
            if missing_raw:
                report.missing_selection_keys = missing_raw
                if sel.on_missing == "error":
                    raise SchemaError(
                        f"{len(missing_raw)} selection id(s) not present in dataset. Examples: {missing_raw[:5]}"
                    )
            base = read_parquet_records_by_ids(
                job.input.path,
                ids=present,
                sequence_col=job.input.columns.sequence,
                annotations_col=job.input.columns.annotations,
                id_col=job.input.columns.id,
                details_col=job.input.columns.details,
                alphabet=job.input.alphabet,
                ann_policy={
                    "ambiguous": job.input.annotations.ambiguous,
                    "offset_mode": job.input.annotations.offset_mode,
                    "zero_as_unspecified": job.input.annotations.zero_as_unspecified,
                    "on_missing_kmer": job.input.annotations.on_missing_kmer,
                    "require_non_empty": job.input.annotations.require_non_empty,
                    "min_per_record": job.input.annotations.min_per_record,
                    "require_non_null_cols": list(job.input.annotations.require_non_null_cols),
                    "on_invalid_row": job.input.annotations.on_invalid_row,
                },
                row_reporter=row_reporter,
            )
            by_id = {r.id: r for r in _apply_plugins(base, plugins)}
            dropped = sorted(list(set(present) - set(by_id.keys())))
            if dropped:
                report.dropped_by_policy = dropped
            ordered_keys = canonical_keys if sel.keep_order else sorted(by_id.keys())
            for j, k in enumerate(ordered_keys):
                r = by_id.get(k)
                if r is None:
                    continue
                csv_label = overlays[j] if sel.keep_order else overlay_by_key.get(k)
                if csv_label:
                    label = _join_label(csv_label, _row_label(r), f"id={r.id}")
                    r = apply_overlay_label(r, label, source="csv")
                else:
                    if not _has_overlay(r):
                        label = _overlay_default_label(r, sel_row=j)
                        r = apply_overlay_label(r, label, source="default")
                records.append(r)
        elif sel.match_on == "row":
            try:
                idxs = [int(x) for x in keys]
            except Exception as e:
                raise SchemaError("selection.match_on=row requires integer row indices") from e
            found = _select_by_row_index(
                _apply_plugins(_base_iter(job, row_reporter=row_reporter), plugins),
                idxs,
            )
            missing = [i for i in idxs if i not in found]
            if missing:
                report.missing_selection_keys = [str(i) for i in missing]
                if sel.on_missing == "error":
                    raise SchemaError(
                        f"{len(missing)} row index/indices not present in dataset. Examples: {missing[:5]}"
                    )
            ordered = idxs if sel.keep_order else sorted(found.keys())
            overlay_by_key = {str(k): v for k, v in spec.overlay_by_key.items()}
            for j, i in enumerate(ordered):
                r = found.get(i)
                if r is None:
                    continue
                csv_label = overlays[j] if sel.keep_order else overlay_by_key.get(str(i))
                if csv_label:
                    label = _join_label(csv_label, _row_label(r), f"id={r.id}")
                    r = apply_overlay_label(r, label, source="csv")
                else:
                    if not _has_overlay(r):
                        label = _overlay_default_label(r, sel_row=j)
                        r = apply_overlay_label(r, label, source="default")
                records.append(r)
        else:  # sequence
            want = set(keys)
            found_map: dict[str, SeqRecord] = {}
            for r in _apply_plugins(_base_iter(job, row_reporter=row_reporter), plugins):
                k = r.sequence
                if k in want and k not in found_map:
                    found_map[k] = r
                    if len(found_map) == len(want):
                        break
            missing = [k for k in keys if k not in found_map]
            if missing:
                report.missing_selection_keys = missing
                if sel.on_missing == "error":
                    raise SchemaError(
                        f"{len(missing)} sequence(s) from selection CSV not found. Examples: {missing[:5]}"
                    )
            ordered = keys if sel.keep_order else sorted(found_map.keys())
            overlay_by_key = spec.overlay_by_key
            for j, k in enumerate(ordered):
                r = found_map.get(k)
                if r is None:
                    continue
                csv_label = overlays[j] if sel.keep_order else overlay_by_key.get(k)
                if csv_label:
                    label = _join_label(csv_label, _row_label(r), f"id={r.id}")
                    r = apply_overlay_label(r, label, source="csv")
                else:
                    if not _has_overlay(r):
                        label = _overlay_default_label(r, sel_row=j)
                        r = apply_overlay_label(r, label, source="default")
                records.append(r)
    else:
        # No selection: honor sample or limit
        base = _apply_plugins(_base_iter(job, row_reporter=row_reporter), plugins)
        if job.input.sample is not None:
            if job.input.sample.mode == "first_n":
                n = job.input.sample.n
                for i, r in enumerate(base):
                    if i >= n:
                        break
                    if not _has_overlay(r):
                        label = _row_label(r) or f"row={r.row_index}"
                        r = apply_overlay_label(r, label, source="default")
                    records.append(r)
            else:
                total_rows = _parquet_row_count(job.input.path)
                k = min(job.input.sample.n, total_rows)
                import random

                rng = random.Random(int(job.input.sample.seed))
                idxs = sorted(rng.sample(range(total_rows), k))
                found = _select_by_row_index(base, idxs)
                missing = [i for i in idxs if i not in found]
                if missing:
                    for _ in missing:
                        report.note_skip_record("sampled_row_filtered")
                for i in idxs:
                    r = found.get(i)
                    if r is None:
                        continue
                    if not _has_overlay(r):
                        label = _row_label(r) or f"row={r.row_index}"
                        r = apply_overlay_label(r, label, source="default")
                    records.append(r)
        else:
            limit = job.input.limit
            count = 0
            for r in base:
                if limit is not None and count >= limit:
                    break
                if not _has_overlay(r):
                    label = _row_label(r) or f"row={getattr(r, 'row_index', count)}"
                    r = apply_overlay_label(r, label, source="default")
                records.append(r)
                count += 1

    report.yielded_records = len(records)

    if not records:
        raise SchemaError("No records to render after selection and filtering.")

    # ---- outputs
    if job.output.images is not None:
        img_cfg = job.output.images
        img_cfg.dir.mkdir(parents=True, exist_ok=True)
        used: set[str] = set()
        for i, rec in enumerate(records):
            base = rec.id if rec.id else f"record_{i}"
            name = unique_stem(safe_stem(base), used)
            out_path = img_cfg.dir / f"{name}.{img_cfg.fmt}"
            fig = render_figure(
                rec,
                style=style,
                palette=palette,
                out_path=str(out_path),
                fmt=img_cfg.fmt,
                legend_entries=legend_entries_for_record(rec),
            )
            import matplotlib.pyplot as plt

            plt.close(fig)
        report.outputs["images_dir"] = str(img_cfg.dir)

    if job.output.video is not None:
        vid_cfg = job.output.video
        vid_cfg.path.parent.mkdir(parents=True, exist_ok=True)
        render_video(
            records,
            out_path=vid_cfg.path,
            fps=vid_cfg.fps,
            style=style,
            palette=palette,
            fmt=vid_cfg.fmt,
            frames_per_record=vid_cfg.frames_per_record,
            pauses=vid_cfg.pauses,
            width_px=vid_cfg.width_px,
            height_px=vid_cfg.height_px,
            aspect_ratio=vid_cfg.aspect_ratio,
            total_duration=vid_cfg.total_duration,
            report=report_cb,
        )
        report.outputs["video_path"] = str(vid_cfg.path)

    if job.run.emit_report and job.run.report_path is not None:
        report.write(job.run.report_path)
        report.outputs["report_path"] = str(job.run.report_path)

    if job.run.fail_on_skips or job.run.strict:
        if report.has_skips():
            raise SchemaError("Run completed with skipped rows/records; see run report for details.")

    return report
