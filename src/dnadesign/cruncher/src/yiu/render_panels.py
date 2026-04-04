"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/render_panels.py

Shared YIU helpers for loading view contracts and assembling composite renders.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dnadesign.cruncher.yiu.bundle_models import PayloadViewEntry


def adapter_for_view(view: PayloadViewEntry):
    if view.contract_kind == "sequence_evidence_map_v1":
        from dnadesign.baserender.src.adapters.sequence_evidence_map_v1 import SequenceEvidenceMapV1Adapter

        return SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="IUPAC_DNA")
    if view.contract_kind == "yiu_payload_visual_v1":
        from dnadesign.baserender.src.adapters.yiu_payload_visual_v1 import YiuPayloadVisualV1Adapter

        return YiuPayloadVisualV1Adapter(columns={}, policies={}, alphabet="IUPAC_DNA")
    raise ValueError(f"unsupported YIU view contract for rendering: {view.contract_kind}")


def load_contract_rows(contract_path: Path, *, input_kind: str) -> list[dict[str, object]]:
    if input_kind == "jsonl":
        return [json.loads(line) for line in contract_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"render input must decode to a mapping or list: {contract_path}")


def load_view_records(contract_path: Path, *, view: PayloadViewEntry):
    adapter = adapter_for_view(view)
    rows = load_contract_rows(contract_path, input_kind=view.input_kind)
    return [adapter.apply(row, row_index=index) for index, row in enumerate(rows)]


def render_view_panel(
    *,
    baserender_module,
    records,
    renderer_kind: str,
    style_preset: str | None,
    style_overrides: dict[str, object],
):
    record_or_records = records[0] if len(records) == 1 else records
    grid = {"ncols": 2} if len(records) == 2 else None
    return baserender_module.render(
        record_or_records,
        renderer=renderer_kind,
        style={"preset": style_preset, "overrides": style_overrides},
        grid=grid,
    )


def figure_to_rgba_array(fig) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))


def trim_white_border_rgba(arr: np.ndarray, *, threshold: int = 248, pad_px: int = 8) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[2] < 3:
        return arr
    rgb = arr[:, :, :3]
    non_white = np.any(rgb < int(threshold), axis=2)
    ys, xs = np.where(non_white)
    if ys.size == 0 or xs.size == 0:
        return arr
    height, width = arr.shape[:2]
    pad = max(0, int(pad_px))
    top = max(0, int(ys.min()) - pad)
    bottom = min(height - 1, int(ys.max()) + pad)
    left = max(0, int(xs.min()) - pad)
    right = min(width - 1, int(xs.max()) + pad)
    return arr[top : bottom + 1, left : right + 1, :]


def save_composite_render(*, panel_images: list[np.ndarray], render_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not panel_images:
        raise ValueError("YIU composite render requires at least one panel image")
    trimmed_images = [trim_white_border_rgba(image) for image in panel_images]
    height_ratios = [max(1, image.shape[0]) for image in trimmed_images]
    max_width = max(image.shape[1] for image in trimmed_images)
    total_height = sum(height_ratios)
    composite = plt.figure(figsize=(max_width / 180.0, total_height / 180.0), dpi=180)
    try:
        axes = composite.subplots(
            nrows=len(panel_images),
            ncols=1,
            gridspec_kw={"height_ratios": height_ratios, "hspace": 0.03},
        )
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes], dtype=object)
        for axis, image in zip(axes.tolist(), trimmed_images, strict=True):
            axis.imshow(image)
            axis.set_axis_off()
        composite.patch.set_facecolor("white")
        composite.patch.set_alpha(1.0)
        render_path.parent.mkdir(parents=True, exist_ok=True)
        composite.savefig(
            render_path,
            format=render_path.suffix.lstrip(".") or "pdf",
            bbox_inches="tight",
            pad_inches=0.01,
            facecolor="white",
        )
    finally:
        plt.close(composite)


__all__ = [
    "figure_to_rgba_array",
    "load_contract_rows",
    "load_view_records",
    "render_view_panel",
    "save_composite_render",
    "trim_white_border_rgba",
]
