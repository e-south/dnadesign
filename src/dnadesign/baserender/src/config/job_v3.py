"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/config/job_v3.py

Job v3 schema and loader with strict nested key validation and explicit outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..core import (
    Alphabet,
    ContractError,
    SchemaError,
    ensure,
    reject_unknown_keys,
    require_mapping,
    require_one_of,
)
from .adapter_contracts import adapter_contract, adapter_kinds


@dataclass(frozen=True)
class SampleCfg:
    mode: str
    n: int
    seed: int | None


@dataclass(frozen=True)
class AdapterCfg:
    kind: str
    columns: Mapping[str, Any]
    policies: Mapping[str, Any]


@dataclass(frozen=True)
class InputCfg:
    kind: str
    path: Path
    adapter: AdapterCfg
    alphabet: Alphabet
    limit: int | None
    sample: SampleCfg | None


@dataclass(frozen=True)
class SelectionCfg:
    path: Path
    match_on: str
    column: str
    overlay_column: str | None
    keep_order: bool
    on_missing: str


@dataclass(frozen=True)
class PluginSpec:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineCfg:
    plugins: tuple[PluginSpec, ...] = ()


@dataclass(frozen=True)
class RenderCfg:
    renderer: str
    style_preset: str | Path | None
    style_overrides: Mapping[str, Any]


@dataclass(frozen=True)
class ImagesOutputCfg:
    kind: str
    dir: Path
    fmt: str


@dataclass(frozen=True)
class VideoOutputCfg:
    kind: str
    path: Path
    fmt: str
    fps: int
    frames_per_record: int
    pauses: Mapping[str, float]
    width_px: int | None
    height_px: int | None
    aspect_ratio: float | None
    total_duration: float | None


OutputCfg = ImagesOutputCfg | VideoOutputCfg


@dataclass(frozen=True)
class RunCfg:
    strict: bool
    fail_on_skips: bool
    emit_report: bool
    report_path: Path | None


@dataclass(frozen=True)
class JobV3:
    version: int
    name: str
    path: Path
    results_root: Path
    input: InputCfg
    selection: SelectionCfg | None
    pipeline: PipelineCfg
    render: RenderCfg
    outputs: tuple[OutputCfg, ...]
    run: RunCfg


def _baserender_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _workspace_root_from_job_path(job_path: Path) -> Path | None:
    job_abs = job_path.resolve()
    if job_abs.name != "job.yml":
        return None
    workspace_root = job_abs.parent
    if not workspace_root.is_dir():
        return None
    if not (workspace_root / "inputs").is_dir():
        return None
    if not (workspace_root / "outputs").is_dir():
        return None
    return workspace_root


def _default_results_root(job_path: Path, *, caller_root: Path) -> Path:
    workspace_root = _workspace_root_from_job_path(job_path)
    if workspace_root is not None:
        return workspace_root / "outputs"
    return caller_root / "results"


def _job_output_root(job_path: Path, results_root: Path) -> Path:
    workspace_root = _workspace_root_from_job_path(job_path)
    if workspace_root is not None and results_root.resolve() == (workspace_root / "outputs").resolve():
        return results_root.resolve()
    return (results_root / job_path.stem).resolve()


def resolve_job_path(spec: str | Path) -> Path:
    p = Path(spec)
    if p.suffix.lower() in {".yml", ".yaml"}:
        if p.exists():
            return p
        rooted = _baserender_root() / p
        if rooted.exists():
            return rooted
        raise FileNotFoundError(f"Could not resolve job file: {spec}")

    root = _baserender_root()
    for candidate in (root / "jobs" / f"{p}.yml", root / "docs" / "examples" / f"{p}.yml"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve job name '{spec}' in jobs/ or docs/examples/")


def _resolve_path(job_path: Path, raw: str, *, field: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        if not p.exists():
            raise SchemaError(f"{field} does not exist: {p}")
        return p
    candidate = (job_path.parent / p).resolve()
    if candidate.exists():
        return candidate
    raise SchemaError(f"{field} does not exist: {candidate}")


def _parse_aspect(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    sval = str(value)
    if ":" in sval:
        left, _, right = sval.partition(":")
        try:
            return float(left) / float(right)
        except Exception as exc:
            raise SchemaError(f"Invalid aspect ratio: {value!r}") from exc
    try:
        return float(sval)
    except Exception as exc:
        raise SchemaError(f"Invalid aspect ratio: {value!r}") from exc


def _parse_bool(value: Any, *, field: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise SchemaError(f"{field} must be bool")


def _parse_sample(raw: Any) -> SampleCfg:
    data = require_mapping(raw, "input.sample")
    reject_unknown_keys(data, {"mode", "n", "seed"}, "input.sample")

    mode = str(data.get("mode", "")).strip().lower()
    require_one_of(mode, {"first_n", "random_rows"}, "input.sample.mode")

    n = int(data.get("n", 0))
    ensure(n >= 1, "input.sample.n must be >= 1", SchemaError)

    seed_raw = data.get("seed")
    if mode == "random_rows":
        ensure(seed_raw is not None, "input.sample.seed is required when mode=random_rows", SchemaError)
        seed = int(seed_raw)
    else:
        seed = None if seed_raw is None else int(seed_raw)

    return SampleCfg(mode=mode, n=n, seed=seed)


def _parse_plugin_specs(raw: Any) -> tuple[PluginSpec, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise SchemaError("pipeline.plugins must be a list")

    out: list[PluginSpec] = []
    for item in raw:
        if isinstance(item, str):
            out.append(PluginSpec(name=item, params={}))
        elif isinstance(item, Mapping):
            if len(item) != 1:
                raise SchemaError(f"plugin mapping must have a single key, got: {item}")
            name, params = next(iter(item.items()))
            if not isinstance(params, Mapping):
                raise SchemaError(f"plugin params must be a mapping for '{name}'")
            out.append(PluginSpec(name=str(name), params=dict(params)))
        else:
            raise SchemaError(f"Unsupported plugin spec: {item!r}")
    return tuple(out)


def _parse_adapter(job_path: Path, raw: Any) -> AdapterCfg:
    data = require_mapping(raw, "input.adapter")
    reject_unknown_keys(data, {"kind", "columns", "policies"}, "input.adapter")

    kind = str(data.get("kind", "")).strip()
    require_one_of(kind, adapter_kinds(), "input.adapter.kind")
    contract = adapter_contract(kind)

    columns = require_mapping(data.get("columns", {}), "input.adapter.columns")
    policies = require_mapping(data.get("policies", {}), "input.adapter.policies")
    reject_unknown_keys(columns, set(contract.allowed_config_columns), "input.adapter.columns")

    missing = sorted(set(contract.required_config_columns) - set(columns.keys()))
    if missing:
        raise SchemaError(f"input.adapter.columns missing required keys for {kind}: {missing}")

    parsed_columns = dict(columns)
    for key in contract.resolved_path_columns:
        parsed_columns[key] = str(
            _resolve_path(
                job_path,
                str(parsed_columns[key]),
                field=f"input.adapter.columns.{key}",
            )
        )

    reject_unknown_keys(policies, set(contract.allowed_policy_keys), "input.adapter.policies")
    parsed_policies = contract.normalize_policies(policies, "input.adapter.policies")

    return AdapterCfg(kind=kind, columns=parsed_columns, policies=parsed_policies)


def _parse_input(job_path: Path, raw: Any) -> InputCfg:
    data = require_mapping(raw, "input")
    reject_unknown_keys(data, {"kind", "path", "adapter", "alphabet", "limit", "sample"}, "input")

    kind = str(data.get("kind", "")).strip().lower()
    ensure(kind == "parquet", "input.kind must be 'parquet'", SchemaError)

    raw_path = str(data.get("path", "")).strip()
    ensure(raw_path != "", "input.path is required", SchemaError)
    path = _resolve_path(job_path, raw_path, field="input.path")

    adapter = _parse_adapter(job_path, data.get("adapter"))

    alphabet = str(data.get("alphabet", "DNA")).upper()
    require_one_of(alphabet, {"DNA", "RNA", "PROTEIN"}, "input.alphabet")

    sample = data.get("sample")
    sample_cfg = None if sample is None else _parse_sample(sample)

    limit_raw = data.get("limit")
    limit = None if limit_raw is None else int(limit_raw)
    if limit is not None:
        ensure(limit >= 1, "input.limit must be >= 1 when set", SchemaError)
    if sample_cfg is not None and limit is not None:
        raise SchemaError("input.sample cannot be used together with input.limit")

    return InputCfg(
        kind=kind,
        path=path,
        adapter=adapter,
        alphabet=alphabet,
        limit=limit,
        sample=sample_cfg,
    )


def _parse_selection(job_path: Path, raw: Any) -> SelectionCfg:
    data = require_mapping(raw, "selection")
    reject_unknown_keys(
        data,
        {"path", "match_on", "column", "overlay_column", "keep_order", "on_missing"},
        "selection",
    )

    raw_path = str(data.get("path", "")).strip()
    ensure(raw_path != "", "selection.path is required", SchemaError)
    path = _resolve_path(job_path, raw_path, field="selection.path")

    match_on = str(data.get("match_on", "id")).strip().lower()
    require_one_of(match_on, {"id", "sequence", "row"}, "selection.match_on")

    column_default = "row" if match_on == "row" else match_on
    column = str(data.get("column", column_default)).strip()
    ensure(column != "", "selection.column must be a non-empty string", SchemaError)

    overlay_raw = data.get("overlay_column")
    overlay_column = None if overlay_raw is None else str(overlay_raw)
    if overlay_column is not None:
        ensure(overlay_column.strip() != "", "selection.overlay_column must be non-empty when set", SchemaError)

    keep_order = _parse_bool(data.get("keep_order"), field="selection.keep_order", default=True)

    on_missing = str(data.get("on_missing", "warn")).strip().lower()
    require_one_of(on_missing, {"skip", "warn", "error"}, "selection.on_missing")

    return SelectionCfg(
        path=path,
        match_on=match_on,
        column=column,
        overlay_column=overlay_column,
        keep_order=keep_order,
        on_missing=on_missing,
    )


def _parse_render(raw: Any) -> RenderCfg:
    data = require_mapping(raw, "render")
    reject_unknown_keys(data, {"renderer", "style"}, "render")

    renderer = str(data.get("renderer", "")).strip()
    ensure(renderer == "sequence_rows", "render.renderer must be 'sequence_rows'", SchemaError)

    style_raw = require_mapping(data.get("style", {}), "render.style")
    reject_unknown_keys(style_raw, {"preset", "overrides"}, "render.style")

    preset_raw = style_raw.get("preset")
    if preset_raw is None or str(preset_raw).strip() == "":
        style_preset: str | Path | None = None
    else:
        style_preset = str(preset_raw)

    overrides_raw = style_raw.get("overrides", {})
    if overrides_raw is None:
        overrides_raw = {}
    if not isinstance(overrides_raw, Mapping):
        raise SchemaError("render.style.overrides must be a mapping")

    return RenderCfg(renderer=renderer, style_preset=style_preset, style_overrides=dict(overrides_raw))


def _resolve_output_dir(job: Path, results_root: Path, raw_dir: str | None) -> Path:
    root = _job_output_root(job, results_root)
    default = root / "images"
    if raw_dir is None:
        return default
    p = Path(raw_dir)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _resolve_output_file(job: Path, results_root: Path, raw_path: str | None) -> Path:
    job_name = job.stem
    root = _job_output_root(job, results_root)
    default = root / f"{job_name}.mp4"
    if raw_path is None:
        return default
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _parse_outputs(job_path: Path, results_root: Path, raw: Any) -> tuple[OutputCfg, ...]:
    if not isinstance(raw, (list, tuple)):
        raise SchemaError("outputs must be a non-empty list")
    if len(raw) == 0:
        raise SchemaError("outputs must contain at least one output entry")

    outputs: list[OutputCfg] = []
    seen_kinds: set[str] = set()

    for i, item in enumerate(raw):
        data = require_mapping(item, f"outputs[{i}]")
        kind = str(data.get("kind", "")).strip().lower()
        require_one_of(kind, {"images", "video"}, f"outputs[{i}].kind")
        if kind in seen_kinds:
            raise SchemaError(f"outputs contains duplicate kind '{kind}'")
        seen_kinds.add(kind)

        if kind == "images":
            reject_unknown_keys(data, {"kind", "dir", "fmt"}, f"outputs[{i}]")
            fmt = str(data.get("fmt", "png")).strip().lower()
            require_one_of(fmt, {"png", "svg", "pdf"}, f"outputs[{i}].fmt")
            raw_dir = data.get("dir")
            out_dir = _resolve_output_dir(job_path, results_root, None if raw_dir is None else str(raw_dir))
            outputs.append(ImagesOutputCfg(kind="images", dir=out_dir, fmt=fmt))
            continue

        reject_unknown_keys(
            data,
            {
                "kind",
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
            f"outputs[{i}]",
        )
        fmt = str(data.get("fmt", "mp4")).strip().lower()
        ensure(fmt == "mp4", f"outputs[{i}].fmt must be 'mp4'", SchemaError)

        fps = int(data.get("fps", 2))
        ensure(fps >= 1, f"outputs[{i}].fps must be >= 1", SchemaError)

        frames_per_record = int(data.get("frames_per_record", 1))
        ensure(frames_per_record >= 1, f"outputs[{i}].frames_per_record must be >= 1", SchemaError)

        pauses_raw = data.get("pauses", {})
        if pauses_raw is None:
            pauses_raw = {}
        if not isinstance(pauses_raw, Mapping):
            raise SchemaError(f"outputs[{i}].pauses must be a mapping")
        pauses = {str(k): float(v) for k, v in pauses_raw.items()}

        width_raw = data.get("width_px")
        width_px = None if width_raw is None else int(width_raw)
        if width_px is not None:
            ensure(width_px >= 1, f"outputs[{i}].width_px must be >= 1", SchemaError)

        height_raw = data.get("height_px")
        height_px = None if height_raw is None else int(height_raw)
        if height_px is not None:
            ensure(height_px >= 1, f"outputs[{i}].height_px must be >= 1", SchemaError)

        aspect_ratio = _parse_aspect(data.get("aspect"))
        if aspect_ratio is not None:
            ensure(aspect_ratio > 0, f"outputs[{i}].aspect must be > 0", SchemaError)

        total_duration_raw = data.get("total_duration")
        total_duration = None if total_duration_raw is None else float(total_duration_raw)
        if total_duration is not None:
            ensure(total_duration > 0, f"outputs[{i}].total_duration must be > 0", SchemaError)

        raw_path = data.get("path")
        out_path = _resolve_output_file(job_path, results_root, None if raw_path is None else str(raw_path))

        outputs.append(
            VideoOutputCfg(
                kind="video",
                path=out_path,
                fmt=fmt,
                fps=fps,
                frames_per_record=frames_per_record,
                pauses=pauses,
                width_px=width_px,
                height_px=height_px,
                aspect_ratio=aspect_ratio,
                total_duration=total_duration,
            )
        )

    return tuple(outputs)


def _parse_run(job_path: Path, results_root: Path, raw: Any) -> RunCfg:
    if raw is None:
        data = {}
    else:
        data = require_mapping(raw, "run")
    reject_unknown_keys(data, {"strict", "fail_on_skips", "emit_report", "report_path"}, "run")

    strict = _parse_bool(data.get("strict"), field="run.strict", default=False)
    fail_on_skips = _parse_bool(data.get("fail_on_skips"), field="run.fail_on_skips", default=False)
    emit_report = _parse_bool(data.get("emit_report"), field="run.emit_report", default=True)

    raw_report_path = data.get("report_path")
    if raw_report_path is None:
        report_path = (_job_output_root(job_path, results_root) / "run_report.json") if emit_report else None
    else:
        rp = Path(str(raw_report_path))
        report_path = rp if rp.is_absolute() else (_job_output_root(job_path, results_root) / rp).resolve()

    return RunCfg(
        strict=strict,
        fail_on_skips=fail_on_skips,
        emit_report=emit_report,
        report_path=report_path,
    )


def load_job_v3(path: str | Path, *, caller_root: str | Path | None = None) -> JobV3:
    try:
        job_path = resolve_job_path(path)
        caller_scope = Path.cwd().resolve() if caller_root is None else Path(caller_root).expanduser().resolve()
        try:
            raw = yaml.safe_load(job_path.read_text())
        except Exception as exc:
            raise SchemaError(f"Could not parse job YAML: {job_path}") from exc

        data = require_mapping(raw, "top-level")
        reject_unknown_keys(
            data,
            {"version", "results_root", "input", "selection", "pipeline", "render", "outputs", "run"},
            "top-level",
        )

        version = data.get("version")
        ensure(version == 3, "Job YAML must specify version: 3", SchemaError)

        results_root_raw = data.get("results_root")
        if results_root_raw is None:
            results_root = _default_results_root(job_path, caller_root=caller_scope)
        else:
            p = Path(str(results_root_raw))
            results_root = p if p.is_absolute() else (job_path.parent / p).resolve()

        input_cfg = _parse_input(job_path, data.get("input"))

        selection_raw = data.get("selection")
        selection_cfg = None if selection_raw is None else _parse_selection(job_path, selection_raw)

        pipeline_raw = require_mapping(data.get("pipeline", {}), "pipeline")
        reject_unknown_keys(pipeline_raw, {"plugins"}, "pipeline")
        pipeline_cfg = PipelineCfg(plugins=_parse_plugin_specs(pipeline_raw.get("plugins")))

        render_cfg = _parse_render(data.get("render"))

        outputs_cfg = _parse_outputs(job_path, results_root, data.get("outputs"))

        run_cfg = _parse_run(job_path, results_root, data.get("run"))

        return JobV3(
            version=3,
            name=job_path.stem,
            path=job_path,
            results_root=results_root,
            input=input_cfg,
            selection=selection_cfg,
            pipeline=pipeline_cfg,
            render=render_cfg,
            outputs=outputs_cfg,
            run=run_cfg,
        )
    except ContractError as exc:
        raise SchemaError(str(exc)) from exc


def output_kind(job: JobV3, kind: str) -> OutputCfg | None:
    for entry in job.outputs:
        if entry.kind == kind:
            return entry
    return None


def validate_job_v3(path: str | Path, *, caller_root: str | Path | None = None) -> JobV3:
    return load_job_v3(path, caller_root=caller_root)
