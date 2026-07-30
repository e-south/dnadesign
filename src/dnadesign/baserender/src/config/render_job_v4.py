"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/config/render_job_v4.py

Generic Render Job v4 schema and loader with strict nested key validation.

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
from ..workspaces import WORKSPACE_MARKER_FILENAME
from .adapter_contracts import adapter_contract, normalize_adapter_config
from .job_contracts import (
    DEFAULT_RENDER_CONTRACT_KIND,
    render_contract_descriptor,
    validate_render_contract_renderer,
)


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
class RenderContractCfg:
    kind: str


@dataclass(frozen=True)
class ImagesOutputCfg:
    kind: str
    dir: Path | None
    path: Path | None
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
    content_fit: str = "native"
    title_text: str | None = None
    title_font_size: int | None = None
    title_align: str = "center"


OutputCfg = ImagesOutputCfg | VideoOutputCfg


@dataclass(frozen=True)
class RunCfg:
    strict: bool
    fail_on_skips: bool


@dataclass(frozen=True)
class BundleCfg:
    path: Path


@dataclass(frozen=True)
class RenderJobV4:
    version: int
    contract: RenderContractCfg
    name: str
    path: Path
    bundle: BundleCfg
    input: InputCfg
    selection: SelectionCfg | None
    pipeline: PipelineCfg
    render: RenderCfg
    outputs: tuple[OutputCfg, ...]
    run: RunCfg


def _baserender_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _has_packaged_job_examples(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.glob("*.yaml")) or any(path.glob("*.yml"))


def _workspace_root_from_job_path(job_path: Path) -> Path | None:
    job_abs = job_path.resolve()
    if job_abs.name != "job.yaml":
        return None
    workspace_root = job_abs.parent
    if not workspace_root.is_dir():
        return None
    if not (workspace_root / WORKSPACE_MARKER_FILENAME).exists():
        return None
    if not (workspace_root / "inputs").is_dir():
        return None
    if not (workspace_root / "outputs").is_dir():
        return None
    return workspace_root


def _published_job_owner_root_from_job_path(job_path: Path) -> Path | None:
    job_abs = job_path.resolve()
    if job_abs.parent.name != "baserender_jobs":
        return None
    owner_root = job_abs.parent.parent
    if not owner_root.is_dir():
        return None
    return owner_root


def _cassette_run_root_from_job_path(job_path: Path) -> Path | None:
    return _published_job_owner_root_from_job_path(job_path)


def _job_owner_root(job_path: Path, *, caller_scope: Path) -> Path:
    workspace_root = _workspace_root_from_job_path(job_path)
    if workspace_root is not None:
        return workspace_root.resolve()
    cassette_root = _published_job_owner_root_from_job_path(job_path)
    if cassette_root is not None:
        return cassette_root.resolve()
    return caller_scope.resolve()


def _allowed_path_roots(job_path: Path, *, caller_scope: Path) -> tuple[Path, ...]:
    roots: list[Path] = []
    for root in (
        _job_owner_root(job_path, caller_scope=caller_scope),
        job_path.parent.resolve(),
        caller_scope.resolve(),
    ):
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def _append_allowed_root(roots: list[Path], path: Path) -> None:
    resolved = path.resolve()
    if resolved not in roots:
        roots.append(resolved)


def _inline_mapping_allowed_roots(
    mapping: Mapping[str, Any], *, caller_scope: Path, job_path: Path
) -> tuple[Path, ...]:
    roots = list(_allowed_path_roots(job_path, caller_scope=caller_scope))

    def _append_if_absolute(raw: Any) -> None:
        if raw is None:
            return
        text = str(raw).strip()
        if text == "":
            return
        candidate = Path(text).expanduser()
        if candidate.is_absolute():
            _append_allowed_root(roots, candidate)

    bundle_data = mapping.get("bundle")
    if isinstance(bundle_data, Mapping):
        _append_if_absolute(bundle_data.get("path"))

    input_data = mapping.get("input")
    if isinstance(input_data, Mapping):
        _append_if_absolute(input_data.get("path"))
        adapter_data = input_data.get("adapter")
        if isinstance(adapter_data, Mapping):
            columns_data = adapter_data.get("columns")
            if isinstance(columns_data, Mapping):
                for key in ("hits_path", "config_path"):
                    _append_if_absolute(columns_data.get(key))

    selection_data = mapping.get("selection")
    if isinstance(selection_data, Mapping):
        _append_if_absolute(selection_data.get("path"))

    pipeline_data = mapping.get("pipeline")
    if isinstance(pipeline_data, Mapping):
        plugins_data = pipeline_data.get("plugins")
        if isinstance(plugins_data, (list, tuple)):
            for plugin in plugins_data:
                if not isinstance(plugin, Mapping) or len(plugin) != 1:
                    continue
                _, params = next(iter(plugin.items()))
                if not isinstance(params, Mapping):
                    continue
                for key in ("config_path", "library_path", "run_manifest_path", "lockfile_path", "motif_store_root"):
                    _append_if_absolute(params.get(key))

    return tuple(roots)


def _ensure_within_allowed_roots(candidate: Path, *, field: str, allowed_roots: tuple[Path, ...]) -> Path:
    resolved = candidate.resolve()
    for root in allowed_roots:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return resolved
    roots = ", ".join(str(root) for root in allowed_roots)
    raise SchemaError(f"{field} must stay within {roots}: {resolved}")


def _parse_bundle(job_path: Path, raw: Any, *, allowed_roots: tuple[Path, ...]) -> BundleCfg:
    ensure(raw is not None, "bundle.path is required", SchemaError)
    data = require_mapping(raw, "bundle")
    reject_unknown_keys(data, {"path"}, "bundle")
    raw_path = str(data.get("path", "")).strip()
    ensure(raw_path != "", "bundle.path is required", SchemaError)
    path = Path(raw_path)
    resolved = _ensure_within_allowed_roots(
        path if path.is_absolute() else job_path.parent / path,
        field="bundle.path",
        allowed_roots=allowed_roots,
    )
    ensure(resolved != resolved.parent, "bundle.path must name an owned directory", SchemaError)
    return BundleCfg(path=resolved)


def resolve_job_path(spec: str | Path) -> Path:
    p = Path(spec)
    if p.suffix.lower() == ".yaml":
        if p.exists():
            return p
        rooted = _baserender_root() / p
        if rooted.exists():
            return rooted
        raise FileNotFoundError(f"Could not resolve job file: {spec}")

    root = _baserender_root()
    candidates = [root / "docs" / "examples" / f"{p}.yaml"]
    jobs_root = root / "jobs"
    has_packaged_jobs = _has_packaged_job_examples(jobs_root)
    if has_packaged_jobs:
        candidates.insert(0, jobs_root / f"{p}.yaml")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if has_packaged_jobs:
        raise FileNotFoundError(f"Could not resolve job name '{spec}' in jobs/ or docs/examples/")
    raise FileNotFoundError(f"Could not resolve job name '{spec}' in docs/examples/ or as an explicit path")


def _resolve_path(job_path: Path, raw: str, *, field: str, allowed_roots: tuple[Path, ...]) -> Path:
    p = Path(raw)
    if p.is_absolute():
        candidate = _ensure_within_allowed_roots(p, field=field, allowed_roots=allowed_roots)
        if not candidate.exists():
            raise SchemaError(f"{field} does not exist: {candidate}")
        return candidate
    candidate = _ensure_within_allowed_roots(job_path.parent / p, field=field, allowed_roots=allowed_roots)
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


def _parse_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise SchemaError(f"{field} must be int")
    try:
        parsed = int(value)
    except Exception as exc:
        raise SchemaError(f"{field} must be int") from exc
    if isinstance(value, float) and not value.is_integer():
        raise SchemaError(f"{field} must be int")
    return parsed


def _parse_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise SchemaError(f"{field} must be float")
    try:
        return float(value)
    except Exception as exc:
        raise SchemaError(f"{field} must be float") from exc


def _parse_contract(raw: Any) -> RenderContractCfg:
    if raw is None:
        return RenderContractCfg(kind=DEFAULT_RENDER_CONTRACT_KIND)
    data = require_mapping(raw, "contract")
    reject_unknown_keys(data, {"kind"}, "contract")
    kind = str(data.get("kind", "")).strip()
    ensure(kind != "", "contract.kind is required", SchemaError)
    return RenderContractCfg(kind=render_contract_descriptor(kind).kind)


def _parse_sample(raw: Any) -> SampleCfg:
    data = require_mapping(raw, "input.sample")
    reject_unknown_keys(data, {"mode", "n", "seed"}, "input.sample")

    mode = str(data.get("mode", "")).strip().lower()
    require_one_of(mode, {"first_n", "random_rows"}, "input.sample.mode")

    n = _parse_int(data.get("n", 0), field="input.sample.n")
    ensure(n >= 1, "input.sample.n must be >= 1", SchemaError)

    seed_raw = data.get("seed")
    if mode == "random_rows":
        ensure(seed_raw is not None, "input.sample.seed is required when mode=random_rows", SchemaError)
        seed = _parse_int(seed_raw, field="input.sample.seed")
    else:
        seed = None if seed_raw is None else _parse_int(seed_raw, field="input.sample.seed")

    return SampleCfg(mode=mode, n=n, seed=seed)


def _parse_plugin_specs(job_path: Path, raw: Any, *, allowed_roots: tuple[Path, ...]) -> tuple[PluginSpec, ...]:
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
            parsed_params = dict(params)
            plugin_name = str(name)
            if plugin_name == "attach_motifs_from_config":
                config_path_raw = parsed_params.get("config_path")
                if config_path_raw is None:
                    raise SchemaError("pipeline plugin 'attach_motifs_from_config' requires params.config_path")
                parsed_params["config_path"] = str(
                    _resolve_path(
                        job_path,
                        str(config_path_raw),
                        field="pipeline.plugins.attach_motifs_from_config.config_path",
                        allowed_roots=allowed_roots,
                    )
                )
            if plugin_name == "attach_motifs_from_library":
                library_path_raw = parsed_params.get("library_path")
                if library_path_raw is None:
                    raise SchemaError("pipeline plugin 'attach_motifs_from_library' requires params.library_path")
                parsed_params["library_path"] = str(
                    _resolve_path(
                        job_path,
                        str(library_path_raw),
                        field="pipeline.plugins.attach_motifs_from_library.library_path",
                        allowed_roots=allowed_roots,
                    )
                )
            if plugin_name == "attach_motifs_from_cruncher_lockfile":
                for key in ("run_manifest_path", "lockfile_path", "motif_store_root"):
                    value = parsed_params.get(key)
                    if value is not None:
                        parsed_params[key] = str(
                            _resolve_path(
                                job_path,
                                str(value),
                                field=f"pipeline.plugins.attach_motifs_from_cruncher_lockfile.{key}",
                                allowed_roots=allowed_roots,
                            )
                        )
                has_manifest = parsed_params.get("run_manifest_path") is not None
                has_lock_bundle = (
                    parsed_params.get("lockfile_path") is not None and parsed_params.get("motif_store_root") is not None
                )
                if not has_manifest and not has_lock_bundle:
                    raise SchemaError(
                        "pipeline plugin 'attach_motifs_from_cruncher_lockfile' requires "
                        "params.run_manifest_path or both params.lockfile_path + params.motif_store_root"
                    )
            out.append(PluginSpec(name=plugin_name, params=parsed_params))
        else:
            raise SchemaError(f"Unsupported plugin spec: {item!r}")
    return tuple(out)


def _parse_adapter(job_path: Path, raw: Any, *, allowed_roots: tuple[Path, ...]) -> AdapterCfg:
    data = require_mapping(raw, "input.adapter")
    reject_unknown_keys(data, {"kind", "columns", "policies"}, "input.adapter")

    columns = require_mapping(data.get("columns", {}), "input.adapter.columns")
    policies = require_mapping(data.get("policies", {}), "input.adapter.policies")

    def _resolve_adapter_path_column(key: str, value: Any) -> str:
        return str(
            _resolve_path(
                job_path,
                str(value),
                field=f"input.adapter.columns.{key}",
                allowed_roots=allowed_roots,
            )
        )

    kind, parsed_columns, parsed_policies = normalize_adapter_config(
        kind=data.get("kind", ""),
        columns=columns,
        policies=policies,
        resolve_path=_resolve_adapter_path_column,
    )

    return AdapterCfg(kind=kind, columns=parsed_columns, policies=parsed_policies)


def _validate_adapter_compatibility(input_cfg: InputCfg, render_cfg: RenderCfg) -> None:
    contract = adapter_contract(input_cfg.adapter.kind)
    if render_cfg.renderer not in contract.supported_renderers:
        allowed = ", ".join(sorted(contract.supported_renderers))
        raise SchemaError(
            "input.adapter.kind "
            f"{input_cfg.adapter.kind!r} is not compatible with render.renderer {render_cfg.renderer!r}; "
            f"supported render.renderer values: {allowed}"
        )
    if input_cfg.alphabet not in contract.supported_alphabets:
        allowed = ", ".join(sorted(contract.supported_alphabets))
        raise SchemaError(
            "input.adapter.kind "
            f"{input_cfg.adapter.kind!r} is not compatible with input.alphabet {input_cfg.alphabet!r}; "
            f"supported input.alphabet values: {allowed}"
        )


def _parse_input(job_path: Path, raw: Any, *, allowed_roots: tuple[Path, ...]) -> InputCfg:
    data = require_mapping(raw, "input")
    reject_unknown_keys(data, {"kind", "path", "adapter", "alphabet", "limit", "sample"}, "input")

    kind = str(data.get("kind", "")).strip().lower()
    require_one_of(kind, {"parquet", "json", "jsonl"}, "input.kind")

    raw_path = str(data.get("path", "")).strip()
    ensure(raw_path != "", "input.path is required", SchemaError)
    path = _resolve_path(job_path, raw_path, field="input.path", allowed_roots=allowed_roots)

    adapter = _parse_adapter(job_path, data.get("adapter"), allowed_roots=allowed_roots)

    alphabet = str(data.get("alphabet", "DNA")).upper()
    require_one_of(alphabet, {"DNA", "IUPAC_DNA", "RNA", "PROTEIN"}, "input.alphabet")

    sample = data.get("sample")
    sample_cfg = None if sample is None else _parse_sample(sample)

    limit_raw = data.get("limit")
    limit = None if limit_raw is None else _parse_int(limit_raw, field="input.limit")
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


def _parse_selection(job_path: Path, raw: Any, *, allowed_roots: tuple[Path, ...]) -> SelectionCfg:
    data = require_mapping(raw, "selection")
    reject_unknown_keys(
        data,
        {"path", "match_on", "column", "overlay_column", "keep_order", "on_missing"},
        "selection",
    )

    raw_path = str(data.get("path", "")).strip()
    ensure(raw_path != "", "selection.path is required", SchemaError)
    path = _resolve_path(job_path, raw_path, field="selection.path", allowed_roots=allowed_roots)

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
    require_one_of(
        renderer,
        {"sequence_rows", "nucleotide_evidence_map", "hairpin_cartoon", "topology_cartoon", "snapback_map"},
        "render.renderer",
    )

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


def _resolve_output_dir(
    bundle_root: Path,
    raw_dir: str | None,
    *,
    field: str,
) -> Path:
    relative = Path("images" if raw_dir is None else raw_dir)
    ensure(not relative.is_absolute(), f"{field} must be relative to bundle.path", SchemaError)
    resolved = (bundle_root / relative).resolve()
    try:
        resolved.relative_to(bundle_root)
    except ValueError as exc:
        raise SchemaError(f"{field} must stay inside bundle.path: {relative}") from exc
    ensure(resolved != bundle_root / "manifest.json", f"{field} is reserved for the bundle manifest", SchemaError)
    return resolved


def _resolve_output_file(
    job: Path,
    bundle_root: Path,
    raw_path: str | None,
    *,
    field: str,
) -> Path:
    job_name = job.stem
    relative = Path(f"{job_name}.mp4" if raw_path is None else raw_path)
    ensure(not relative.is_absolute(), f"{field} must be relative to bundle.path", SchemaError)
    resolved = (bundle_root / relative).resolve()
    try:
        resolved.relative_to(bundle_root)
    except ValueError as exc:
        raise SchemaError(f"{field} must stay inside bundle.path: {relative}") from exc
    ensure(resolved != bundle_root, f"{field} must name a file inside bundle.path", SchemaError)
    ensure(resolved != bundle_root / "manifest.json", f"{field} is reserved for the bundle manifest", SchemaError)
    return resolved


def _parse_outputs(
    job_path: Path,
    bundle_root: Path,
    raw: Any,
) -> tuple[OutputCfg, ...]:
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
            reject_unknown_keys(data, {"kind", "dir", "path", "fmt"}, f"outputs[{i}]")
            fmt = str(data.get("fmt", "png")).strip().lower()
            require_one_of(fmt, {"png", "svg", "pdf"}, f"outputs[{i}].fmt")
            raw_dir = data.get("dir")
            raw_path = data.get("path")
            if raw_dir is not None and raw_path is not None:
                raise SchemaError(f"outputs[{i}] must define only one of dir or path for images output")
            out_dir = None
            out_path = None
            if raw_dir is not None:
                out_dir = _resolve_output_dir(
                    bundle_root,
                    str(raw_dir),
                    field=f"outputs[{i}].dir",
                )
            elif raw_path is not None:
                out_path = _resolve_output_file(
                    job_path,
                    bundle_root,
                    str(raw_path),
                    field=f"outputs[{i}].path",
                )
            else:
                out_dir = _resolve_output_dir(
                    bundle_root,
                    None,
                    field=f"outputs[{i}].dir",
                )
            outputs.append(ImagesOutputCfg(kind="images", dir=out_dir, path=out_path, fmt=fmt))
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
                "content_fit",
                "title_text",
                "title_font_size",
                "title_align",
            },
            f"outputs[{i}]",
        )
        fmt = str(data.get("fmt", "mp4")).strip().lower()
        ensure(fmt == "mp4", f"outputs[{i}].fmt must be 'mp4'", SchemaError)

        fps = _parse_int(data.get("fps", 2), field=f"outputs[{i}].fps")
        ensure(fps >= 1, f"outputs[{i}].fps must be >= 1", SchemaError)

        frames_per_record = _parse_int(data.get("frames_per_record", 1), field=f"outputs[{i}].frames_per_record")
        ensure(frames_per_record >= 1, f"outputs[{i}].frames_per_record must be >= 1", SchemaError)

        pauses_raw = data.get("pauses", {})
        if pauses_raw is None:
            pauses_raw = {}
        if not isinstance(pauses_raw, Mapping):
            raise SchemaError(f"outputs[{i}].pauses must be a mapping")
        pauses = {str(k): _parse_float(v, field=f"outputs[{i}].pauses.{k}") for k, v in pauses_raw.items()}

        width_raw = data.get("width_px")
        width_px = None if width_raw is None else _parse_int(width_raw, field=f"outputs[{i}].width_px")
        if width_px is not None:
            ensure(width_px >= 1, f"outputs[{i}].width_px must be >= 1", SchemaError)

        height_raw = data.get("height_px")
        height_px = None if height_raw is None else _parse_int(height_raw, field=f"outputs[{i}].height_px")
        if height_px is not None:
            ensure(height_px >= 1, f"outputs[{i}].height_px must be >= 1", SchemaError)

        aspect_ratio = _parse_aspect(data.get("aspect"))
        if aspect_ratio is not None:
            ensure(aspect_ratio > 0, f"outputs[{i}].aspect must be > 0", SchemaError)
        if width_px is not None and height_px is not None and aspect_ratio is not None:
            declared_ratio = float(width_px) / float(height_px)
            if abs(declared_ratio - float(aspect_ratio)) > 1.0e-6:
                raise SchemaError(
                    f"outputs[{i}].aspect conflicts with width_px/height_px ({aspect_ratio!r} != {declared_ratio:.6g})"
                )

        total_duration_raw = data.get("total_duration")
        total_duration = (
            None
            if total_duration_raw is None
            else _parse_float(total_duration_raw, field=f"outputs[{i}].total_duration")
        )
        if total_duration is not None:
            ensure(total_duration > 0, f"outputs[{i}].total_duration must be > 0", SchemaError)

        content_fit = str(data.get("content_fit", "native")).strip().lower()
        require_one_of(
            content_fit,
            {"native", "fill_width", "fill_width_per_frame"},
            f"outputs[{i}].content_fit",
        )

        title_text_raw = data.get("title_text")
        title_text = None if title_text_raw is None else str(title_text_raw).strip()
        if title_text == "":
            title_text = None

        title_font_size_raw = data.get("title_font_size")
        title_font_size = (
            None
            if title_font_size_raw is None
            else _parse_int(title_font_size_raw, field=f"outputs[{i}].title_font_size")
        )
        if title_font_size is not None:
            ensure(title_font_size >= 6, f"outputs[{i}].title_font_size must be >= 6", SchemaError)

        title_align = str(data.get("title_align", "center")).strip().lower()
        require_one_of(title_align, {"left", "center", "right"}, f"outputs[{i}].title_align")

        raw_path = data.get("path")
        out_path = _resolve_output_file(
            job_path,
            bundle_root,
            None if raw_path is None else str(raw_path),
            field=f"outputs[{i}].path",
        )

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
                content_fit=content_fit,
                title_text=title_text,
                title_font_size=title_font_size,
                title_align=title_align,
            )
        )

    destinations = [_output_destination_for_validation(output) for output in outputs]
    ensure(
        len(destinations) == len(set(destinations)),
        "outputs must resolve to distinct bundle paths",
        SchemaError,
    )
    manifest_path = bundle_root / "manifest.json"
    for index, destination in enumerate(destinations):
        if manifest_path == destination or manifest_path in destination.parents:
            raise SchemaError(f"outputs[{index}] must not place an artifact beneath the bundle manifest")
    for left_index, left in enumerate(destinations):
        for right_index, right in enumerate(destinations[left_index + 1 :], start=left_index + 1):
            if left in right.parents or right in left.parents:
                raise SchemaError(
                    f"outputs[{left_index}] and outputs[{right_index}] have an impossible "
                    "file/directory prefix collision"
                )
    return tuple(outputs)


def _output_destination_for_validation(output: OutputCfg) -> Path:
    if isinstance(output, ImagesOutputCfg):
        if output.path is not None:
            return output.path
        assert output.dir is not None
        return output.dir
    return output.path


def _parse_run(raw: Any) -> RunCfg:
    if raw is None:
        data = {}
    else:
        data = require_mapping(raw, "run")
    reject_unknown_keys(data, {"strict", "fail_on_skips"}, "run")

    strict = _parse_bool(data.get("strict"), field="run.strict", default=False)
    fail_on_skips = _parse_bool(data.get("fail_on_skips"), field="run.fail_on_skips", default=False)
    return RunCfg(
        strict=strict,
        fail_on_skips=fail_on_skips,
    )


def _parse_sequence_rows_job_mapping(
    raw_mapping: Any,
    *,
    job_path: Path,
    caller_scope: Path,
    allowed_roots: tuple[Path, ...] | None = None,
) -> RenderJobV4:
    if allowed_roots is None:
        allowed_roots = _allowed_path_roots(job_path, caller_scope=caller_scope)
    data = require_mapping(raw_mapping, "top-level")
    reject_unknown_keys(
        data,
        {"version", "contract", "bundle", "input", "selection", "pipeline", "render", "outputs", "run"},
        "top-level",
    )

    version = data.get("version")
    ensure(version == 4, "Job YAML must specify version: 4", SchemaError)
    contract_cfg = _parse_contract(data.get("contract"))

    bundle_cfg = _parse_bundle(job_path, data.get("bundle"), allowed_roots=allowed_roots)

    input_cfg = _parse_input(job_path, data.get("input"), allowed_roots=allowed_roots)

    selection_raw = data.get("selection")
    selection_cfg = (
        None if selection_raw is None else _parse_selection(job_path, selection_raw, allowed_roots=allowed_roots)
    )

    pipeline_raw = require_mapping(data.get("pipeline", {}), "pipeline")
    reject_unknown_keys(pipeline_raw, {"plugins"}, "pipeline")
    pipeline_cfg = PipelineCfg(
        plugins=_parse_plugin_specs(job_path, pipeline_raw.get("plugins"), allowed_roots=allowed_roots)
    )

    render_cfg = _parse_render(data.get("render"))
    validate_render_contract_renderer(contract_cfg.kind, render_cfg.renderer, field="contract.kind")
    _validate_adapter_compatibility(input_cfg, render_cfg)
    outputs_cfg = _parse_outputs(job_path, bundle_cfg.path, data.get("outputs"))
    run_cfg = _parse_run(data.get("run"))

    return RenderJobV4(
        version=4,
        contract=contract_cfg,
        name=job_path.stem,
        path=job_path,
        bundle=bundle_cfg,
        input=input_cfg,
        selection=selection_cfg,
        pipeline=pipeline_cfg,
        render=render_cfg,
        outputs=outputs_cfg,
        run=run_cfg,
    )


def load_render_job(path: str | Path, *, caller_root: str | Path | None = None) -> RenderJobV4:
    try:
        job_path = resolve_job_path(path)
        if caller_root is None:
            # Default non-workspace outputs to job-local scope.
            caller_scope = job_path.parent.resolve()
        else:
            caller_scope = Path(caller_root).expanduser().resolve()
        try:
            raw = yaml.safe_load(job_path.read_text())
        except Exception as exc:
            raise SchemaError(f"Could not parse job YAML: {job_path}") from exc
        return _parse_sequence_rows_job_mapping(raw, job_path=job_path, caller_scope=caller_scope)
    except ContractError as exc:
        raise SchemaError(str(exc)) from exc


def load_render_job_from_mapping(
    mapping: Mapping[str, Any],
    *,
    caller_root: str | Path | None = None,
    source_name: str = "inline_job.yaml",
) -> RenderJobV4:
    try:
        caller_scope = Path.cwd().resolve() if caller_root is None else Path(caller_root).expanduser().resolve()
        name = str(source_name).strip()
        ensure(name != "", "source_name must be non-empty", SchemaError)
        source_path = Path(name)
        ensure(source_path.is_absolute() is False, "source_name must be relative, not absolute", SchemaError)
        ensure(
            source_path.name == name,
            "source_name must be a simple filename (no directory components)",
            SchemaError,
        )
        job_path = (caller_scope / source_path).resolve()
        mapping_data = require_mapping(mapping, "top-level")
        parsed_mapping = dict(mapping_data)
        allowed_roots = _inline_mapping_allowed_roots(parsed_mapping, caller_scope=caller_scope, job_path=job_path)
        return _parse_sequence_rows_job_mapping(
            parsed_mapping,
            job_path=job_path,
            caller_scope=caller_scope,
            allowed_roots=allowed_roots,
        )
    except ContractError as exc:
        raise SchemaError(str(exc)) from exc


def output_kind(job: RenderJobV4, kind: str) -> OutputCfg | None:
    for entry in job.outputs:
        if entry.kind == kind:
            return entry
    return None


def validate_render_job(path: str | Path, *, caller_root: str | Path | None = None) -> RenderJobV4:
    return load_render_job(path, caller_root=caller_root)


def load_job(path: str | Path, *, caller_root: str | Path | None = None) -> RenderJobV4:
    return load_render_job(path, caller_root=caller_root)


def validate_job(path: str | Path, *, caller_root: str | Path | None = None) -> RenderJobV4:
    return validate_render_job(path, caller_root=caller_root)
