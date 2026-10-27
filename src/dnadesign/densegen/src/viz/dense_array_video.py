"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/dense_array_video.py

Builds workspace-scoped DenseGen showcase videos from sampled accepted outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
from pathlib import Path

import pandas as pd

from dnadesign.baserender import run_job
from dnadesign.densegen.src.config.plots import PlotVideoConfig
from dnadesign.densegen.src.integrations.baserender.notebook_contract import (
    densegen_baserender_title_text,
    densegen_notebook_render_contract,
    densegen_video_subtitle_text,
)
from dnadesign.densegen.src.viz.dense_array_video_source import (
    encode_video_source_annotations,
    order_video_source_rows,
    prepare_video_source_frame,
    sample_video_source_rows,
)

_VIDEO_SUBTITLE_COLUMN = "densegen__video_subtitle"
_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_segment(text: str) -> str:
    cleaned = _SAFE_SEGMENT_RE.sub("_", str(text).strip())
    cleaned = cleaned.strip("._-")
    if cleaned in {"", ".", ".."}:
        return "densegen"
    return cleaned


def _attach_video_display_metadata(
    frame: pd.DataFrame,
    *,
    workspace_name: str | None,
    show_subtitle: bool = True,
) -> tuple[pd.DataFrame, str]:
    enriched = frame.copy()
    if show_subtitle:
        enriched[_VIDEO_SUBTITLE_COLUMN] = [
            densegen_video_subtitle_text(record_id=record_id, plan_name=plan_name)
            for record_id, plan_name in zip(
                enriched["id"].astype(str).tolist(),
                enriched["densegen__plan"].astype(str).tolist(),
                strict=True,
            )
        ]
    else:
        enriched[_VIDEO_SUBTITLE_COLUMN] = ""
    workspace_title = densegen_baserender_title_text(workspace_name=str(workspace_name or "").strip())
    return enriched, workspace_title


def _output_path(out_path: Path, *, video_cfg: PlotVideoConfig) -> Path:
    out_dir = out_path.parent.resolve()
    stage_b_root = (out_dir / "stage_b").resolve()
    if str(video_cfg.mode) == "single_plan_single_video":
        plan_name = str(video_cfg.single_plan_name or "").strip()
        plan_segment = _safe_segment(plan_name)
    else:
        plan_segment = "all_plans"
    candidate = (stage_b_root / plan_segment / str(video_cfg.output_name)).resolve()
    try:
        candidate.relative_to(stage_b_root)
    except ValueError as exc:
        raise ValueError("Dense-array video output path escaped stage_b workspace scope.") from exc
    return candidate


def _write_selection_csv(path: Path, *, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id"])
        writer.writeheader()
        for record_id in ids:
            writer.writerow({"id": str(record_id)})


def _published_video_is_complete(bundle_root: Path, published_file: Path) -> bool:
    manifest_path = bundle_root / "manifest.json"
    if not manifest_path.is_file() or not published_file.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("schema") != "dnadesign.baserender.render_bundle_manifest.v1":
        return False
    inventory = payload.get("artifact_inventory")
    if not isinstance(inventory, dict):
        return False
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list):
        return False
    expected_path = published_file.relative_to(bundle_root).as_posix()
    expected_digest = _sha256_file(published_file)
    return any(
        isinstance(item, dict) and item.get("path") == expected_path and item.get("sha256") == expected_digest
        for item in artifacts
    )


def plot_dense_array_video_showcase(
    dense_arrays_df: pd.DataFrame,
    out_path: Path,
    *,
    video_cfg: PlotVideoConfig,
    workspace_name: str | None = None,
) -> Path:
    if dense_arrays_df is None or dense_arrays_df.empty:
        raise ValueError("Dense-array video requires non-empty accepted output rows.")

    frame = prepare_video_source_frame(dense_arrays_df)
    ordered = order_video_source_rows(frame, video_cfg=video_cfg)

    target_total_frames = int(round(float(video_cfg.playback.target_duration_sec) * float(video_cfg.playback.fps)))
    if target_total_frames < 1:
        raise ValueError("Dense-array video target frame budget must be >= 1.")
    if target_total_frames > int(video_cfg.limits.max_total_frames):
        raise ValueError(
            "Dense-array video target frames exceed plots.video.limits.max_total_frames; "
            "reduce target_duration_sec/fps or increase max_total_frames."
        )
    snapshot_cap = max(1, min(int(video_cfg.sampling.max_snapshots), int(target_total_frames)))
    sampled, effective_stride = sample_video_source_rows(
        ordered,
        stride=int(video_cfg.sampling.stride),
        max_source_rows=int(video_cfg.sampling.max_source_rows),
        max_snapshots=snapshot_cap,
        plan_snapshot_counts=dict(video_cfg.sampling.plan_snapshots or {}),
    )
    sampled = encode_video_source_annotations(sampled)
    sampled, workspace_title = _attach_video_display_metadata(
        sampled,
        workspace_name=workspace_name,
        show_subtitle=bool(video_cfg.show_subtitle),
    )
    if len(sampled) > target_total_frames:
        raise ValueError(
            "Dense-array video snapshot count exceeds playback frame budget; "
            "increase sampling.stride or reduce max_snapshots."
        )
    estimated_render_sec = float(target_total_frames / max(1, int(video_cfg.playback.fps))) * 1.5
    if estimated_render_sec > float(video_cfg.limits.max_estimated_render_sec):
        raise ValueError(
            "Dense-array video estimated render time exceeds plots.video.limits.max_estimated_render_sec; "
            "reduce duration/fps or raise max_estimated_render_sec."
        )

    out_file = _output_path(out_path, video_cfg=video_cfg)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    contract = densegen_notebook_render_contract()
    style_overrides = dict(contract.style_overrides)
    presentation = video_cfg.presentation
    if presentation.palette:
        palette = dict(style_overrides.get("palette") or {})
        palette.update(dict(presentation.palette))
        style_overrides["palette"] = palette
    if presentation.legend_font_size is not None:
        style_overrides["legend_font_size"] = int(presentation.legend_font_size)
    if presentation.legend_height_px is not None:
        style_overrides["legend_height_px"] = float(presentation.legend_height_px)
    title_font_size = int(
        max(
            style_overrides.get("font_size_seq", 18),
            style_overrides.get("font_size_label", 18),
            style_overrides.get("legend_font_size", 18),
        )
    )
    render_identity_payload = {
        "records": sampled.to_json(orient="split", date_format="iso"),
        "video": video_cfg.model_dump(mode="json"),
        "style": style_overrides,
        "workspace_title": workspace_title,
        "contract": {
            "adapter_kind": contract.adapter_kind,
            "adapter_columns": dict(contract.adapter_columns),
            "adapter_policies": dict(contract.adapter_policies),
            "style_preset": contract.style_preset,
        },
    }
    render_identity = hashlib.sha256(
        json.dumps(render_identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    bundle_root = out_file.parent / f"{out_file.stem}.render-v1-{render_identity}"
    published_file = bundle_root / out_file.name
    if bundle_root.exists():
        if _published_video_is_complete(bundle_root, published_file):
            return published_file
        raise ValueError(f"Dense-array video bundle exists but is incomplete or invalid: {bundle_root}")
    with tempfile.TemporaryDirectory(prefix="dense-video-", dir=str(out_path.parent)) as tmpdir:
        tmp_root = Path(tmpdir)
        records_path = tmp_root / "records.parquet"
        selection_path = tmp_root / "selection.csv"
        sampled.to_parquet(records_path, index=False, engine="pyarrow")
        if bool(video_cfg.show_subtitle):
            _write_selection_csv(selection_path, ids=[str(item) for item in sampled["id"].astype(str).tolist()])

        adapter_columns = dict(contract.adapter_columns)
        if bool(video_cfg.show_subtitle):
            adapter_columns["video_subtitle"] = _VIDEO_SUBTITLE_COLUMN

        job_mapping: dict[str, object] = {
            "version": 4,
            "bundle": {"path": str(bundle_root)},
            "input": {
                "kind": "parquet",
                "path": str(records_path),
                "adapter": {
                    "kind": str(contract.adapter_kind),
                    "columns": adapter_columns,
                    "policies": dict(contract.adapter_policies),
                },
                "alphabet": "DNA",
            },
            "render": {
                "renderer": "sequence_rows",
                "style": {
                    "preset": str(contract.style_preset),
                    "overrides": style_overrides,
                },
            },
            "outputs": [
                {
                    "kind": "video",
                    "path": out_file.name,
                    "fmt": "mp4",
                    "fps": int(video_cfg.playback.fps),
                    "frames_per_record": 1,
                    "total_duration": float(video_cfg.playback.target_duration_sec),
                    "content_fit": "fill_width" if not (video_cfg.show_title or video_cfg.show_subtitle) else "native",
                    "title_text": workspace_title if bool(video_cfg.show_title) else None,
                    "title_font_size": title_font_size,
                }
            ],
            "run": {"strict": True, "fail_on_skips": True},
        }
        if bool(video_cfg.show_subtitle):
            job_mapping["selection"] = {
                "path": str(selection_path),
                "match_on": "id",
                "column": "id",
                "keep_order": True,
                "on_missing": "error",
            }
        if "densegen__promoter_detail" in sampled.columns:
            job_mapping["input"]["adapter"]["columns"]["promoter_detail"] = "densegen__promoter_detail"
        run_job(job_mapping, kind="sequence_rows_render_v3", caller_root=tmp_root)

    if not published_file.exists():
        raise ValueError(f"Dense-array video output was not created: {published_file}")
    return published_file
