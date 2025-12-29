"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/config/job_v2.py

Job v2 config schema and loader.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

from ..contracts import SchemaError, ensure
from ..plugins.registry import PluginSpec
from ..presets.style_presets import (
    PresetSpec,
    load_style_preset_mapping,
    resolve_style_preset_path,
)


def _baserender_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _ensure_mapping(obj: Any, *, ctx: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise SchemaError(f"{ctx} must be a mapping/dict")
    return obj


def _reject_unknown_keys(data: Mapping[str, Any], allowed: set[str], *, ctx: str) -> None:
    extra = sorted(set(data.keys()) - allowed)
    if extra:
        raise SchemaError(f"Unknown keys in {ctx}: {extra}")


def _parse_aspect(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val)
    if ":" in s:
        w, _, h = s.partition(":")
        return float(w) / float(h)
    return float(s)


@dataclass(frozen=True)
class InputColumnsV2:
    sequence: str
    annotations: str
    id: Optional[str]
    details: Optional[str]


@dataclass(frozen=True)
class AnnotationPolicyV2:
    ambiguous: str
    offset_mode: str
    zero_as_unspecified: bool
    on_missing_kmer: str
    require_non_empty: bool
    min_per_record: int
    require_non_null_cols: list[str]
    on_invalid_row: str


@dataclass(frozen=True)
class SampleCfgV2:
    mode: str
    n: int
    seed: Optional[int]


@dataclass(frozen=True)
class InputCfgV2:
    path: Path
    format: str
    columns: InputColumnsV2
    alphabet: str
    limit: Optional[int]
    sample: Optional[SampleCfgV2]
    annotations: AnnotationPolicyV2


@dataclass(frozen=True)
class SelectionCfgV2:
    path: Path
    match_on: str
    column: str
    overlay_column: Optional[str]
    keep_order: bool
    on_missing: str


@dataclass(frozen=True)
class PipelineCfgV2:
    plugins: Sequence[PluginSpec]


@dataclass(frozen=True)
class StyleCfgV2:
    preset: PresetSpec
    overrides: Mapping[str, Any]


@dataclass(frozen=True)
class VideoCfgV2:
    path: Path
    fmt: str
    fps: int
    frames_per_record: int
    pauses: Mapping[str, float]
    width_px: Optional[int]
    height_px: Optional[int]
    aspect_ratio: Optional[float]
    total_duration: Optional[float]


@dataclass(frozen=True)
class ImagesCfgV2:
    dir: Path
    fmt: str


@dataclass(frozen=True)
class OutputCfgV2:
    results_dir: Path
    video: Optional[VideoCfgV2]
    images: Optional[ImagesCfgV2]


@dataclass(frozen=True)
class RunCfgV2:
    strict: bool
    fail_on_skips: bool
    emit_report: bool
    report_path: Optional[Path]


@dataclass(frozen=True)
class JobV2:
    version: int
    name: str
    input: InputCfgV2
    pipeline: PipelineCfgV2
    selection: Optional[SelectionCfgV2]
    style: StyleCfgV2
    output: OutputCfgV2
    run: RunCfgV2


def _parse_plugins(raw: Any) -> Sequence[PluginSpec]:
    out: list[PluginSpec] = []
    if not raw:
        return out
    if not isinstance(raw, (list, tuple)):
        raise SchemaError("pipeline.plugins must be a list")
    for item in raw:
        if isinstance(item, str):
            out.append(PluginSpec(name=item, params={}))
        elif isinstance(item, Mapping):
            if len(item) != 1:
                raise SchemaError(f"plugin mapping must have single key, got {item}")
            name, params = next(iter(item.items()))
            if not isinstance(params, Mapping):
                raise SchemaError(f"plugin params must be a mapping for '{name}'")
            out.append(PluginSpec(name=str(name), params=dict(params)))
        else:
            raise SchemaError(f"unsupported plugin spec: {item!r}")
    return out


def _input_base_dir(job_path: Path) -> Path:
    root = _baserender_root()
    jobs_dir = root / "jobs"
    if jobs_dir in job_path.parents:
        return root
    return job_path.parent


def _resolve_input_path(job_path: Path, raw: str, *, field: str) -> Path:
    p = Path(raw)
    base = _input_base_dir(job_path)
    p = p if p.is_absolute() else (base / p)
    if not p.exists():
        raise SchemaError(f"{field} does not exist: {p}")
    return p


def _resolve_results_dir(job_dir: Path, raw: Optional[str]) -> Path:
    if raw is None:
        return _baserender_root() / "results"
    p = Path(raw)
    return p if p.is_absolute() else (job_dir / p)


def _resolve_output_path(job_output_dir: Path, raw: Optional[str], default: Path) -> Path:
    if raw is None:
        return default
    p = Path(raw)
    return p if p.is_absolute() else (job_output_dir / p)


def _parse_columns(raw: Any) -> InputColumnsV2:
    m = _ensure_mapping(raw, ctx="input.columns")
    _reject_unknown_keys(m, {"sequence", "annotations", "id", "details"}, ctx="input.columns")
    seq = str(m.get("sequence", "")).strip()
    ann = str(m.get("annotations", "")).strip()
    ensure(seq != "", "input.columns.sequence is required", SchemaError)
    ensure(ann != "", "input.columns.annotations is required", SchemaError)
    raw_id = m.get("id", None)
    if raw_id is None:
        id_col = None
    else:
        id_col = str(raw_id).strip()
        ensure(
            id_col != "",
            "input.columns.id must be a non-empty string or null",
            SchemaError,
        )
    raw_det = m.get("details", None)
    if raw_det is None:
        det_col = None
    else:
        det_col = str(raw_det).strip()
        ensure(
            det_col != "",
            "input.columns.details must be a non-empty string or null",
            SchemaError,
        )
    return InputColumnsV2(sequence=seq, annotations=ann, id=id_col, details=det_col)


def _parse_annotations(raw: Any) -> AnnotationPolicyV2:
    m = _ensure_mapping(raw or {}, ctx="input.annotations")
    _reject_unknown_keys(
        m,
        {
            "ambiguous",
            "offset_mode",
            "zero_as_unspecified",
            "on_missing_kmer",
            "require_non_empty",
            "min_per_record",
            "require_non_null_cols",
            "on_invalid_row",
        },
        ctx="input.annotations",
    )
    ambiguous = str(m.get("ambiguous", "error")).lower()
    if ambiguous not in {"error", "first", "last", "drop"}:
        raise SchemaError("input.annotations.ambiguous must be one of error|first|last|drop")
    offset_mode = str(m.get("offset_mode", "auto")).lower()
    if offset_mode not in {"auto", "zero_based", "one_based"}:
        raise SchemaError("input.annotations.offset_mode must be auto|zero_based|one_based")
    zero_unspec = bool(m.get("zero_as_unspecified", True))
    on_missing_kmer = str(m.get("on_missing_kmer", "error")).lower()
    if on_missing_kmer not in {"error", "skip_entry"}:
        raise SchemaError("input.annotations.on_missing_kmer must be error|skip_entry")
    require_non_empty = bool(m.get("require_non_empty", False))
    min_per_record = int(m.get("min_per_record", 0))
    if require_non_empty and min_per_record < 1:
        min_per_record = 1
    req_cols_raw = m.get("require_non_null_cols", [])
    if req_cols_raw is None:
        req_cols = []
    elif isinstance(req_cols_raw, (list, tuple)):
        req_cols = [str(c) for c in req_cols_raw]
    else:
        raise SchemaError("input.annotations.require_non_null_cols must be a list of strings")
    on_invalid_row = str(m.get("on_invalid_row", "skip")).lower()
    if on_invalid_row not in {"skip", "error"}:
        raise SchemaError("input.annotations.on_invalid_row must be skip|error")
    return AnnotationPolicyV2(
        ambiguous=ambiguous,
        offset_mode=offset_mode,
        zero_as_unspecified=zero_unspec,
        on_missing_kmer=on_missing_kmer,
        require_non_empty=require_non_empty,
        min_per_record=min_per_record,
        require_non_null_cols=req_cols,
        on_invalid_row=on_invalid_row,
    )


def _parse_sample(raw: Any) -> SampleCfgV2:
    m = _ensure_mapping(raw, ctx="input.sample")
    _reject_unknown_keys(m, {"mode", "n", "seed"}, ctx="input.sample")
    mode = str(m.get("mode", "")).lower()
    if mode not in {"first_n", "random_rows"}:
        raise SchemaError("input.sample.mode must be first_n|random_rows")
    n = int(m.get("n", 0))
    ensure(n > 0, "input.sample.n must be > 0", SchemaError)
    seed = m.get("seed", None)
    if mode == "random_rows":
        if seed is None:
            raise SchemaError("input.sample.seed is required for mode=random_rows")
        seed = int(seed)
    else:
        seed = None
    return SampleCfgV2(mode=mode, n=n, seed=seed)


def _parse_selection(job_path: Path, raw: Any) -> SelectionCfgV2:
    m = _ensure_mapping(raw, ctx="selection")
    _reject_unknown_keys(
        m,
        {"path", "match_on", "column", "overlay_column", "keep_order", "on_missing"},
        ctx="selection",
    )
    raw_path = str(m.get("path", "")).strip()
    ensure(raw_path != "", "selection.path is required", SchemaError)
    path = _resolve_input_path(job_path, raw_path, field="selection.path")
    match_on = str(m.get("match_on", "id")).lower()
    ensure(
        match_on in {"id", "sequence", "row"},
        "selection.match_on must be id|sequence|row",
        SchemaError,
    )
    default_col = "row" if match_on == "row" else match_on
    column = str(m.get("column", default_col))
    overlay_raw = m.get("overlay_column", None)
    overlay_column = None if overlay_raw is None else str(overlay_raw)
    keep_order = bool(m.get("keep_order", True))
    on_missing = str(m.get("on_missing", "warn")).lower()
    ensure(
        on_missing in {"skip", "warn", "error"},
        "selection.on_missing must be skip|warn|error",
        SchemaError,
    )
    return SelectionCfgV2(
        path=path,
        match_on=match_on,
        column=column,
        overlay_column=overlay_column,
        keep_order=keep_order,
        on_missing=on_missing,
    )


def _parse_style(raw: Any) -> StyleCfgV2:
    m = _ensure_mapping(raw or {}, ctx="style")
    _reject_unknown_keys(m, {"preset", "overrides"}, ctx="style")
    preset = m.get("preset", "presentation_default")
    if preset is None or str(preset).strip() == "":
        preset = "presentation_default"
    overrides = m.get("overrides", {}) or {}
    ensure(isinstance(overrides, Mapping), "style.overrides must be a mapping", SchemaError)
    return StyleCfgV2(preset=preset, overrides=overrides)


def _parse_output(job_dir: Path, name: str, raw: Any) -> OutputCfgV2:
    m = _ensure_mapping(raw, ctx="output")
    _reject_unknown_keys(m, {"results_dir", "video", "images"}, ctx="output")
    results_dir = _resolve_results_dir(job_dir, m.get("results_dir", None))
    job_out_dir = results_dir / name

    video_raw = m.get("video", {})
    if video_raw is None:
        video_cfg = None
    else:
        vm = _ensure_mapping(video_raw, ctx="output.video")
        _reject_unknown_keys(
            vm,
            {
                "path",
                "fmt",
                "fps",
                "frames_per_record",
                "pauses",
                "width_px",
                "height_px",
                "aspect",
                "total_duration",
            },
            ctx="output.video",
        )
        fmt = str(vm.get("fmt", "mp4")).lower()
        ensure(fmt == "mp4", "output.video.fmt must be mp4", SchemaError)
        fps = int(vm.get("fps", 2))
        ensure(fps >= 1, "output.video.fps must be >= 1", SchemaError)
        frames_per_record = int(vm.get("frames_per_record", 1))
        ensure(
            frames_per_record >= 1,
            "output.video.frames_per_record must be >= 1",
            SchemaError,
        )
        pauses = {str(k): float(v) for (k, v) in (vm.get("pauses") or {}).items()}
        width_px = vm.get("width_px", None)
        height_px = vm.get("height_px", None)
        aspect_ratio = _parse_aspect(vm.get("aspect", None))
        total_duration = vm.get("total_duration", None)
        total_duration = float(total_duration) if total_duration is not None else None
        default_path = job_out_dir / f"{name}.mp4"
        path = _resolve_output_path(job_out_dir, vm.get("path", None), default_path)
        video_cfg = VideoCfgV2(
            path=path,
            fmt=fmt,
            fps=fps,
            frames_per_record=frames_per_record,
            pauses=pauses,
            width_px=int(width_px) if width_px is not None else None,
            height_px=int(height_px) if height_px is not None else None,
            aspect_ratio=aspect_ratio,
            total_duration=total_duration,
        )

    images_raw = m.get("images", None)
    if images_raw is None:
        images_cfg = None
    else:
        im = _ensure_mapping(images_raw, ctx="output.images")
        _reject_unknown_keys(im, {"dir", "fmt"}, ctx="output.images")
        fmt = str(im.get("fmt", "png")).lower()
        ensure(
            fmt in {"png", "svg", "pdf"},
            "output.images.fmt must be png|svg|pdf",
            SchemaError,
        )
        default_dir = job_out_dir / "images"
        out_dir = _resolve_output_path(job_out_dir, im.get("dir", None), default_dir)
        images_cfg = ImagesCfgV2(dir=out_dir, fmt=fmt)

    if video_cfg is None and images_cfg is None:
        raise SchemaError("output must define video and/or images (both cannot be null)")

    return OutputCfgV2(results_dir=results_dir, video=video_cfg, images=images_cfg)


def _parse_run(job_dir: Path, name: str, raw: Any, results_dir: Path) -> RunCfgV2:
    m = _ensure_mapping(raw or {}, ctx="run")
    _reject_unknown_keys(m, {"strict", "fail_on_skips", "emit_report", "report_path"}, ctx="run")
    strict = bool(m.get("strict", False))
    fail_on_skips = bool(m.get("fail_on_skips", False))
    emit_report = bool(m.get("emit_report", True))
    report_raw = m.get("report_path", None)
    if report_raw is None:
        report_path = (results_dir / name / "run_report.json") if emit_report else None
    else:
        job_out_dir = results_dir / name
        report_path = _resolve_output_path(job_out_dir, report_raw, job_out_dir / "run_report.json")
    return RunCfgV2(
        strict=strict,
        fail_on_skips=fail_on_skips,
        emit_report=emit_report,
        report_path=report_path,
    )


def load_job_v2(path: Path) -> JobV2:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, Mapping):
        raise SchemaError("Job YAML must be a mapping/dict at the top level")
    _reject_unknown_keys(
        data,
        {"version", "input", "pipeline", "selection", "style", "output", "run"},
        ctx="top-level",
    )
    version = data.get("version", None)
    ensure(version == 2, "Job YAML must specify version: 2", SchemaError)

    job_dir = path.parent
    name = path.stem

    input_raw = data.get("input", None)
    ensure(input_raw is not None, "input section is required", SchemaError)
    input_m = _ensure_mapping(input_raw, ctx="input")
    _reject_unknown_keys(
        input_m,
        {"path", "format", "columns", "alphabet", "limit", "sample", "annotations"},
        ctx="input",
    )
    raw_path = str(input_m.get("path", "")).strip()
    ensure(raw_path != "", "input.path is required", SchemaError)
    input_path = _resolve_input_path(path, raw_path, field="input.path")
    fmt = str(input_m.get("format", "parquet")).lower()
    ensure(fmt == "parquet", "input.format must be 'parquet'", SchemaError)
    columns = _parse_columns(input_m.get("columns", None))
    alphabet = str(input_m.get("alphabet", "DNA")).upper()
    ensure(
        alphabet in {"DNA", "RNA", "PROTEIN"},
        "input.alphabet must be DNA|RNA|PROTEIN",
        SchemaError,
    )

    sample_raw = input_m.get("sample", None)
    sample_cfg = None
    if sample_raw is not None:
        if "limit" in input_m:
            raise SchemaError("input.sample cannot be used with input.limit")
        sample_cfg = _parse_sample(sample_raw)

    limit = None
    if sample_cfg is None:
        raw_limit = input_m.get("limit", 500)
        limit_val = int(raw_limit)
        limit = None if limit_val <= 0 else limit_val

    ann_policy = _parse_annotations(input_m.get("annotations", None))

    input_cfg = InputCfgV2(
        path=input_path,
        format=fmt,
        columns=columns,
        alphabet=alphabet,
        limit=limit,
        sample=sample_cfg,
        annotations=ann_policy,
    )

    pipeline_raw = data.get("pipeline", None)
    if pipeline_raw is None:
        pipeline_cfg = PipelineCfgV2(plugins=())
    else:
        pm = _ensure_mapping(pipeline_raw, ctx="pipeline")
        _reject_unknown_keys(pm, {"plugins"}, ctx="pipeline")
        pipeline_cfg = PipelineCfgV2(plugins=_parse_plugins(pm.get("plugins", [])))

    selection_raw = data.get("selection", None)
    selection_cfg = _parse_selection(path, selection_raw) if selection_raw is not None else None

    style_cfg = _parse_style(data.get("style", None))

    output_raw = data.get("output", None)
    ensure(output_raw is not None, "output section is required", SchemaError)
    output_cfg = _parse_output(job_dir, name, output_raw)

    run_cfg = _parse_run(job_dir, name, data.get("run", None), output_cfg.results_dir)

    return JobV2(
        version=2,
        name=name,
        input=input_cfg,
        pipeline=pipeline_cfg,
        selection=selection_cfg,
        style=style_cfg,
        output=output_cfg,
        run=run_cfg,
    )


def job_to_dict(job: JobV2) -> dict[str, Any]:
    data: dict[str, Any] = {
        "version": 2,
        "input": {
            "path": str(job.input.path),
            "format": job.input.format,
            "columns": {
                "sequence": job.input.columns.sequence,
                "annotations": job.input.columns.annotations,
                "id": job.input.columns.id,
                "details": job.input.columns.details,
            },
            "alphabet": job.input.alphabet,
            "limit": job.input.limit,
            "sample": (
                None
                if job.input.sample is None
                else {
                    "mode": job.input.sample.mode,
                    "n": job.input.sample.n,
                    "seed": job.input.sample.seed,
                }
            ),
            "annotations": {
                "ambiguous": job.input.annotations.ambiguous,
                "offset_mode": job.input.annotations.offset_mode,
                "zero_as_unspecified": job.input.annotations.zero_as_unspecified,
                "on_missing_kmer": job.input.annotations.on_missing_kmer,
                "require_non_empty": job.input.annotations.require_non_empty,
                "min_per_record": job.input.annotations.min_per_record,
                "require_non_null_cols": list(job.input.annotations.require_non_null_cols),
                "on_invalid_row": job.input.annotations.on_invalid_row,
            },
        },
        "pipeline": {"plugins": [(p.name if not p.params else {p.name: p.params}) for p in job.pipeline.plugins]},
        "selection": (
            None
            if job.selection is None
            else {
                "path": str(job.selection.path),
                "match_on": job.selection.match_on,
                "column": job.selection.column,
                "overlay_column": job.selection.overlay_column,
                "keep_order": job.selection.keep_order,
                "on_missing": job.selection.on_missing,
            }
        ),
        "style": {
            "preset": job.style.preset,
            "overrides": dict(job.style.overrides),
        },
        "output": {
            "results_dir": str(job.output.results_dir),
            "video": (
                None
                if job.output.video is None
                else {
                    "path": str(job.output.video.path),
                    "fmt": job.output.video.fmt,
                    "fps": job.output.video.fps,
                    "frames_per_record": job.output.video.frames_per_record,
                    "pauses": dict(job.output.video.pauses),
                    "width_px": job.output.video.width_px,
                    "height_px": job.output.video.height_px,
                    "aspect": job.output.video.aspect_ratio,
                    "total_duration": job.output.video.total_duration,
                }
            ),
            "images": (
                None
                if job.output.images is None
                else {
                    "dir": str(job.output.images.dir),
                    "fmt": job.output.images.fmt,
                }
            ),
        },
        "run": {
            "strict": job.run.strict,
            "fail_on_skips": job.run.fail_on_skips,
            "emit_report": job.run.emit_report,
            "report_path": str(job.run.report_path) if job.run.report_path else None,
        },
    }
    return data


_DEFAULTS = {
    "input.alphabet": "DNA",
    "input.limit": 500,
    "input.annotations.ambiguous": "error",
    "input.annotations.offset_mode": "auto",
    "input.annotations.zero_as_unspecified": True,
    "input.annotations.on_missing_kmer": "error",
    "input.annotations.require_non_empty": False,
    "input.annotations.min_per_record": 0,
    "input.annotations.require_non_null_cols": (),
    "input.annotations.on_invalid_row": "skip",
    "selection.match_on": "id",
    "selection.keep_order": True,
    "selection.on_missing": "warn",
    "style.preset": "presentation_default",
    "output.video.fmt": "mp4",
    "output.video.fps": 2,
    "output.video.frames_per_record": 1,
    "output.images.fmt": "png",
    "run.strict": False,
    "run.fail_on_skips": False,
    "run.emit_report": True,
}


def _strip_defaults(job: JobV2, data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)

    inp = out.get("input", {})
    cols = inp.get("columns", {})
    if isinstance(cols, Mapping):
        if cols.get("id") is None:
            cols.pop("id", None)
        if cols.get("details") is None:
            cols.pop("details", None)
        inp["columns"] = cols
    if inp.get("sample") is None:
        inp.pop("sample", None)
    if job.input.sample is not None:
        inp.pop("limit", None)
    elif job.input.limit == _DEFAULTS["input.limit"]:
        inp.pop("limit", None)
    if job.input.alphabet == _DEFAULTS["input.alphabet"]:
        inp.pop("alphabet", None)

    ann = inp.get("annotations", {})
    if ann:
        if job.input.annotations.ambiguous == _DEFAULTS["input.annotations.ambiguous"]:
            ann.pop("ambiguous", None)
        if job.input.annotations.offset_mode == _DEFAULTS["input.annotations.offset_mode"]:
            ann.pop("offset_mode", None)
        if job.input.annotations.zero_as_unspecified == _DEFAULTS["input.annotations.zero_as_unspecified"]:
            ann.pop("zero_as_unspecified", None)
        if job.input.annotations.on_missing_kmer == _DEFAULTS["input.annotations.on_missing_kmer"]:
            ann.pop("on_missing_kmer", None)
        if job.input.annotations.require_non_empty == _DEFAULTS["input.annotations.require_non_empty"]:
            ann.pop("require_non_empty", None)
        if job.input.annotations.min_per_record == _DEFAULTS["input.annotations.min_per_record"]:
            ann.pop("min_per_record", None)
        if not job.input.annotations.require_non_null_cols:
            ann.pop("require_non_null_cols", None)
        if job.input.annotations.on_invalid_row == _DEFAULTS["input.annotations.on_invalid_row"]:
            ann.pop("on_invalid_row", None)
        if not ann:
            inp.pop("annotations", None)
    out["input"] = inp

    if out.get("pipeline", {}).get("plugins") in ([], None):
        out.pop("pipeline", None)

    if out.get("selection"):
        sel = out["selection"]
        if job.selection and job.selection.match_on == _DEFAULTS["selection.match_on"]:
            sel.pop("match_on", None)
        if job.selection and job.selection.keep_order == _DEFAULTS["selection.keep_order"]:
            sel.pop("keep_order", None)
        if job.selection and job.selection.on_missing == _DEFAULTS["selection.on_missing"]:
            sel.pop("on_missing", None)
        out["selection"] = sel
    else:
        out.pop("selection", None)

    style = out.get("style", {})
    if job.style.preset == _DEFAULTS["style.preset"]:
        style.pop("preset", None)
    if not style.get("overrides"):
        style.pop("overrides", None)
    if not style:
        out.pop("style", None)
    else:
        out["style"] = style

    output = out.get("output", {})
    if output:
        if output.get("results_dir") == str(_baserender_root() / "results"):
            output.pop("results_dir", None)
        video = output.get("video")
        if isinstance(video, Mapping):
            if video.get("fmt") == _DEFAULTS["output.video.fmt"]:
                video.pop("fmt", None)
            if video.get("fps") == _DEFAULTS["output.video.fps"]:
                video.pop("fps", None)
            if video.get("frames_per_record") == _DEFAULTS["output.video.frames_per_record"]:
                video.pop("frames_per_record", None)
            if not video:
                output["video"] = {}
            else:
                output["video"] = video
        images = output.get("images")
        if isinstance(images, Mapping):
            if images.get("fmt") == _DEFAULTS["output.images.fmt"]:
                images.pop("fmt", None)
            if not images:
                output["images"] = {}
            else:
                output["images"] = images
        out["output"] = output

    run = out.get("run", {})
    if run:
        if job.run.strict == _DEFAULTS["run.strict"]:
            run.pop("strict", None)
        if job.run.fail_on_skips == _DEFAULTS["run.fail_on_skips"]:
            run.pop("fail_on_skips", None)
        if job.run.emit_report == _DEFAULTS["run.emit_report"]:
            run.pop("emit_report", None)
        if not run:
            out.pop("run", None)
        else:
            out["run"] = run

    return out


def job_to_minimal_dict(job: JobV2) -> dict[str, Any]:
    return _strip_defaults(job, job_to_dict(job))


def _redundant_override_paths(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
    prefix: str = "",
) -> list[str]:
    paths: list[str] = []
    for k, v in overrides.items():
        key = f"{prefix}{k}"
        if k in base and isinstance(v, Mapping) and isinstance(base[k], Mapping):
            child_paths = _redundant_override_paths(base[k], v, prefix=f"{key}.")
            paths.extend(child_paths)
        else:
            if k in base and v == base[k]:
                paths.append(key)
    return paths


def find_redundant_overrides(preset: PresetSpec, overrides: Mapping[str, Any]) -> list[str]:
    base = load_style_preset_mapping(resolve_style_preset_path(preset))
    return _redundant_override_paths(base, overrides)


def _strip_redundant_overrides(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in overrides.items():
        if k in base and isinstance(v, Mapping) and isinstance(base[k], Mapping):
            cleaned = _strip_redundant_overrides(base[k], v)
            if cleaned:
                out[k] = cleaned
        else:
            if k not in base or v != base[k]:
                out[k] = v
    return out


def strip_redundant_overrides(preset: PresetSpec, overrides: Mapping[str, Any]) -> dict[str, Any]:
    base = load_style_preset_mapping(resolve_style_preset_path(preset))
    return _strip_redundant_overrides(base, overrides)
