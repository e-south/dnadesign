"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/presets/loader.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

from ..contracts import SchemaError, ensure
from ..plugins.registry import PluginSpec


def _baserender_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_job_path(spec: str | Path) -> Path:
    p = Path(spec)
    if p.suffix.lower() in {".yml", ".yaml"} and p.exists():
        return p
    name = p.stem if p.suffix else str(p)
    candidate = _baserender_root() / "jobs" / f"{name}.yml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not resolve job: {spec}")


@dataclass(frozen=True)
class VideoCfg:
    fmt: str
    fps: int
    frames_per_record: int
    pauses: Dict[str, float]
    width_px: Optional[int]
    height_px: Optional[int]
    aspect_ratio: Optional[float]
    total_duration: Optional[float]
    out_path: Path


@dataclass(frozen=True)
class ImagesCfg:
    dir: Path
    fmt: str


@dataclass(frozen=True)
class SelectionCfg:
    path: Path
    match_on: str  # 'id' | 'sequence' | 'row'
    column: str  # CSV column to read
    overlay_column: Optional[str]
    keep_order: bool
    on_missing: str  # 'skip' | 'warn' | 'error'


@dataclass(frozen=True)
class Job:
    name: str
    input_path: Path
    format: str
    seq_col: str
    ann_col: str
    id_col: Optional[str]
    details_col: Optional[str]
    alphabet: str
    plugins: Sequence[PluginSpec]
    style: Mapping[str, object]
    results_dir: Path
    video: VideoCfg
    limit: Optional[int]  # None or <=0 means unlimited
    sample_seed: Optional[int]  # deterministic random subset when limit is set
    ann_policy: Mapping[str, object]  # explicit annotation parsing policy
    selection: Optional[SelectionCfg]
    images: Optional[ImagesCfg]


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


def _parse_plugins(raw: Any) -> Sequence[PluginSpec]:
    out: list[PluginSpec] = []
    if not raw:
        return out
    if not isinstance(raw, (list, tuple)):
        raise SchemaError("plugins must be a list of strings or mappings")
    for item in raw:
        if isinstance(item, str):
            out.append(PluginSpec(name=item, params={}))
        elif isinstance(item, dict):
            if len(item) != 1:
                raise SchemaError(f"plugin mapping must have single key, got {item}")
            name, params = next(iter(item.items()))
            if not isinstance(params, Mapping):
                raise SchemaError(f"plugin params must be mapping for '{name}'")
            out.append(PluginSpec(name=str(name), params=dict(params)))
        else:
            raise SchemaError(f"unsupported plugin spec: {item!r}")
    return out


def load_job(path: Path) -> Job:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    ensure(
        "input" in data and "output" in data,
        "Job YAML must have 'input' and 'output' sections",
        SchemaError,
    )

    input_ = data["input"]
    output = data["output"]
    style = data.get("style", {})
    plugins = _parse_plugins(data.get("plugins", []))
    name = path.stem
    root = _baserender_root()
    results_dir = root / "results"

    video_cfg = output.get("video", {})
    fmt = str(video_cfg.get("fmt", "mp4"))
    # fps must be a positive integer (Matplotlib/FFmpeg constraint).
    raw_fps = video_cfg.get("fps", 2)
    try:
        fps = int(raw_fps)
    except Exception as e:
        raise SchemaError(f"fps must be an integer, got {raw_fps!r}") from e
    ensure(
        fps >= 1,
        "fps must be >= 1 (use frames_per_record to slow playback)",
        SchemaError,
    )

    # Consolidated duration control: frames_per_record (>=1).
    fpr = video_cfg.get("frames_per_record", None)
    sec = video_cfg.get("seconds_per_seq", None)
    if fpr is not None and sec is not None:
        raise SchemaError("Provide only one of frames_per_record OR seconds_per_seq (not both).")
    if fpr is None and sec is None:
        frames_per_record = 1
    elif fpr is not None:
        frames_per_record = int(fpr)
        ensure(frames_per_record >= 1, "frames_per_record must be >= 1", SchemaError)
    else:
        # Back-compat shim: derive frames from seconds_per_seq × fps
        try:
            frames_per_record = max(1, int(round(float(sec) * fps)))
        except Exception as e:
            raise SchemaError(f"seconds_per_seq must be a number, got {sec!r}") from e
    total_duration = video_cfg.get("total_duration")
    total_duration = float(total_duration) if total_duration is not None else None
    pauses = {str(k): float(v) for (k, v) in (video_cfg.get("pauses") or {}).items()}
    width_px = video_cfg.get("width_px")
    height_px = video_cfg.get("height_px")
    aspect_ratio = _parse_aspect(video_cfg.get("aspect") or video_cfg.get("aspect_ratio"))

    out_path = Path(video_cfg.get("path")) if video_cfg.get("path") else (results_dir / name / f"{name}.{fmt}")
    vc = VideoCfg(
        fmt=fmt,
        fps=fps,
        frames_per_record=frames_per_record,
        pauses=pauses,
        width_px=int(width_px) if width_px is not None else None,
        height_px=int(height_px) if height_px is not None else None,
        aspect_ratio=aspect_ratio,
        total_duration=total_duration,
        out_path=out_path,
    )

    # Optional stills export
    imgs_raw = output.get("images")
    images_cfg: Optional[ImagesCfg] = None
    if imgs_raw is not None:
        fmt_i = str(imgs_raw.get("fmt", "png")).lower()
        ensure(
            fmt_i in {"png", "svg", "pdf"},
            "images.fmt must be png|svg|pdf",
            SchemaError,
        )
        dir_i = imgs_raw.get("dir")
        if dir_i:
            img_dir = Path(dir_i)
            if not img_dir.is_absolute():
                img_dir = results_dir / name / Path(dir_i)
        else:
            img_dir = results_dir / name / "images"
        images_cfg = ImagesCfg(dir=img_dir, fmt=fmt_i)

    # Optional explicit selection from CSV
    sel_raw = data.get("selection")
    selection_cfg: Optional[SelectionCfg] = None
    if sel_raw is not None:
        p = Path(sel_raw.get("path") or sel_raw.get("csv") or "")
        ensure(str(p) != "", "selection.path (or selection.csv) is required", SchemaError)
        if not p.is_absolute():
            p = root / p
        match_on = str(sel_raw.get("match_on", "id")).lower()
        ensure(
            match_on in {"id", "sequence", "row"},
            "selection.match_on must be id|sequence|row",
            SchemaError,
        )
        default_col = "row" if match_on == "row" else match_on
        column = str(sel_raw.get("column", default_col))
        overlay_column = sel_raw.get("overlay_column") or sel_raw.get("details_column")
        overlay_column = str(overlay_column) if overlay_column is not None else None
        keep_order = bool(sel_raw.get("keep_order", True))
        on_missing = str(sel_raw.get("on_missing", "warn")).lower()
        ensure(
            on_missing in {"skip", "warn", "error"},
            "selection.on_missing must be skip|warn|error",
            SchemaError,
        )
        selection_cfg = SelectionCfg(
            path=p,
            match_on=match_on,
            column=column,
            overlay_column=overlay_column,
            keep_order=keep_order,
            on_missing=on_missing,
        )

    # Limit default: 500 (set 0 or negative to process all)
    raw_limit = input_.get("limit", 500)
    limit = int(raw_limit)
    if limit <= 0:
        limit = None
    # Optional deterministic sampling seed (applies when limit is set)
    seed_val = input_.get("sample_seed") or input_.get("seed")
    sample_seed = int(seed_val) if seed_val is not None else None
    # Annotation disambiguation policy (explicit, no silent fallbacks)
    ann_cfg = dict(input_.get("annotations") or {})
    amb = str(ann_cfg.get("ambiguous", "error")).lower()
    if amb not in {"error", "first", "last", "drop"}:
        raise SchemaError(f"input.annotations.ambiguous must be one of ['error','first','last','drop'], got {amb!r}")
    offset_mode = str(ann_cfg.get("offset_mode", "auto")).lower()
    if offset_mode not in {"auto", "zero_based", "one_based"}:
        raise SchemaError(
            f"input.annotations.offset_mode must be one of ['auto','zero_based','one_based'], got {offset_mode!r}"
        )
    zero_unspec = bool(ann_cfg.get("zero_as_unspecified", True))
    # --- row-level gating options ---
    require_non_empty = bool(ann_cfg.get("require_non_empty", False))
    min_per_record = int(ann_cfg.get("min_per_record", 0))
    if require_non_empty and min_per_record < 1:
        min_per_record = 1
    req_cols_raw = ann_cfg.get("require_non_null_cols", [])
    if req_cols_raw is None:
        req_cols = []
    elif isinstance(req_cols_raw, (list, tuple)):
        req_cols = [str(c) for c in req_cols_raw]
    else:
        raise SchemaError("input.annotations.require_non_null_cols must be a list of column names")

    ann_policy: Mapping[str, object] = {
        "ambiguous": amb,
        "offset_mode": offset_mode,
        "zero_as_unspecified": zero_unspec,
        # Pass through new gating options
        "require_non_empty": require_non_empty,
        "min_per_record": min_per_record,
        "require_non_null_cols": req_cols,
    }

    return Job(
        name=name,
        input_path=Path(input_["path"]),
        format=input_.get("format", "parquet"),
        seq_col=input_["columns"]["sequence"],
        ann_col=input_["columns"]["annotations"],
        id_col=input_["columns"].get("id"),
        details_col=input_["columns"].get("details", "details"),
        alphabet=input_.get("alphabet", "DNA"),
        plugins=plugins,
        style=style,
        results_dir=results_dir,
        video=vc,
        limit=limit,
        sample_seed=sample_seed,
        ann_policy=ann_policy,
        selection=selection_cfg,
        images=images_cfg,
    )
