"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/baserender_render.py

Notebook component builders for BaseRender render OPAL analysis notebook components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from io import BytesIO
from typing import Any, Mapping

_BASERENDER_CANVAS_BACKGROUND_RGB = (255, 255, 255)
_BASERENDER_CONTENT_THRESHOLD = 245
_BASERENDER_CONTENT_PAD_PX = 32
_BASERENDER_BLACK_MATTE_THRESHOLD = 24


def render_notebook_baserender_record(
    record_row: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    title: str | None = None,
) -> dict[str, Any]:
    """Render a single record through the public BaseRender API."""

    if not bool(contract.get("available")):
        raise ValueError(str(contract.get("reason") or "BaseRender contract is unavailable."))
    record_id = str(record_row.get("id") or "unknown")
    adapter_kind = str(contract.get("adapter_kind") or "")
    if not adapter_kind:
        raise ValueError("BaseRender contract is missing adapter_kind.")

    baserender = import_module("dnadesign.baserender")
    render_sequence_panel_image = baserender.render_sequence_panel_image

    style_overrides = dict(contract.get("style_overrides") or {})
    adapter_columns = dict(contract.get("adapter_columns") or {})
    try:
        from PIL import Image

        render_route = str(contract.get("render_route") or "figure")
        if render_route == "sequence_panel":
            panel = render_sequence_panel_image(
                dict(record_row),
                adapter_kind=adapter_kind,
                adapter_columns=adapter_columns,
                adapter_policies=dict(contract.get("adapter_policies") or {}),
                style_overrides=style_overrides,
                target_width_px=int(contract.get("target_width_px") or 2600),
                target_height_px=int(contract.get("target_height_px") or 430),
                vertical_anchor=str(contract.get("vertical_anchor") or "center"),
                canvas_top_pad_px=int(contract.get("canvas_top_pad_px") or 0),
                title=title,
            )
            image_bytes = _encode_horizontal_fit_white_png(Image.fromarray(panel.image))
            sequence_length = int(panel.diagnostics.sequence_length_bp)
            feature_count = int(panel.diagnostics.feature_count)
        else:
            record = baserender.adapt_record(
                dict(record_row),
                adapter_kind=adapter_kind,
                adapter_columns=adapter_columns,
                adapter_policies=dict(contract.get("adapter_policies") or {}),
            )
            figure = baserender.render_record_figure(
                record,
                renderer_name=str(contract.get("renderer_name") or "sequence_rows"),
                style_preset=contract.get("style_preset") or "presentation_default",
                style_overrides=style_overrides,
            )
            figure.patch.set_facecolor("white")
            figure.patch.set_alpha(1.0)
            for axis in figure.axes:
                axis.set_facecolor("white")
            buffer = BytesIO()
            figure.savefig(buffer, format="png", facecolor="white", transparent=False, bbox_inches="tight")
            buffer.seek(0)
            image_bytes = _encode_content_fit_white_png(Image.open(buffer))
            sequence_length = len(str(record.sequence))
            feature_count = len(record.features)
            try:
                import matplotlib.pyplot as plt

                plt.close(figure)
            except Exception:
                pass
    except Exception as exc:
        raise ValueError(f"BaseRender image encoding failed for `{record_id}`.") from exc
    if not image_bytes:
        raise ValueError(f"BaseRender image bytes were empty for `{record_id}`.")
    caption = _format_caption(
        contract.get("caption"),
        adapter_kind=adapter_kind,
        sequence_length=sequence_length,
        feature_count=feature_count,
    )
    return {
        "record_id": record_id,
        "image_bytes": image_bytes,
        "caption": caption,
        "alt_text": _format_alt_text(
            contract.get("alt_text_template") or caption,
            record_id=record_id,
            sequence_length=sequence_length,
            feature_count=feature_count,
        ),
        "sequence_length": sequence_length,
        "feature_count": feature_count,
    }


def _format_caption(
    configured: object,
    *,
    adapter_kind: str,
    sequence_length: int,
    feature_count: int,
) -> str:
    defaults = {
        "densegen_tfbs": "DenseGen TFBS annotation",
        "usr_genbank_annotations_v1": "GenBank source annotation",
        "generic_features": "Sequence feature annotation",
    }
    base = str(configured or defaults.get(adapter_kind, "Sequence annotation")).strip().rstrip(".")
    return f"{base} · {sequence_length:,} bp · {feature_count:,} annotated elements"


def _format_alt_text(template: object, *, record_id: str, sequence_length: int, feature_count: int) -> str:
    try:
        return str(template).format(
            record_id=record_id,
            sequence_length=sequence_length,
            feature_count=feature_count,
        )
    except Exception:
        return (
            f"BaseRender sequence diagram for record {record_id}; sequence length "
            f"{sequence_length} bp with {feature_count} annotations."
        )


def _encode_opaque_white_png(image: Any) -> bytes:
    from PIL import Image

    buffer = BytesIO()
    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (*_BASERENDER_CANVAS_BACKGROUND_RGB, 255))
    canvas.alpha_composite(rgba)
    canvas.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _encode_content_fit_white_png(image: Any) -> bytes:
    """Encode a white-canvas PNG whose visible sequence content fills the natural width."""

    import numpy as np
    from PIL import Image

    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (*_BASERENDER_CANVAS_BACKGROUND_RGB, 255))
    white.alpha_composite(rgba)
    white = _normalize_black_border_matte_to_white(white)

    arr = np.asarray(white.convert("RGB"))
    content_mask = (arr < _BASERENDER_CONTENT_THRESHOLD).any(axis=2)
    if not content_mask.any():
        return _encode_opaque_white_png(white)

    ys, xs = np.where(content_mask)
    pad = int(_BASERENDER_CONTENT_PAD_PX)
    left = max(0, int(xs.min()) - pad)
    right = min(white.width, int(xs.max()) + pad + 1)
    top = max(0, int(ys.min()) - pad)
    bottom = min(white.height, int(ys.max()) + pad + 1)
    fitted = white.crop((left, top, right, bottom))
    return _encode_opaque_white_png(fitted)


def _encode_horizontal_fit_white_png(image: Any) -> bytes:
    """Fit visible content horizontally while preserving BaseRender's vertical canvas."""

    import numpy as np
    from PIL import Image

    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (*_BASERENDER_CANVAS_BACKGROUND_RGB, 255))
    white.alpha_composite(rgba)
    white = _normalize_black_border_matte_to_white(white)

    arr = np.asarray(white.convert("RGB"))
    content_mask = (arr < _BASERENDER_CONTENT_THRESHOLD).any(axis=2)
    if not content_mask.any():
        return _encode_opaque_white_png(white)

    _, xs = np.where(content_mask)
    pad = int(_BASERENDER_CONTENT_PAD_PX)
    left = max(0, int(xs.min()) - pad)
    right = min(white.width, int(xs.max()) + pad + 1)
    fitted = white.crop((left, 0, right, white.height))
    return _encode_opaque_white_png(fitted)


def _normalize_black_border_matte_to_white(image: Any) -> Any:
    """Replace a black border-connected matte with white before content fitting."""

    from PIL import ImageDraw

    rgba = image.convert("RGBA")
    white = (*_BASERENDER_CANVAS_BACKGROUND_RGB, 255)
    corners = (
        (0, 0),
        (max(0, rgba.width - 1), 0),
        (0, max(0, rgba.height - 1)),
        (max(0, rgba.width - 1), max(0, rgba.height - 1)),
    )
    for corner in corners:
        pixel = rgba.getpixel(corner)
        if _is_black_matte_pixel(pixel):
            ImageDraw.floodfill(rgba, corner, white, thresh=_BASERENDER_BLACK_MATTE_THRESHOLD)
    return rgba


def _is_black_matte_pixel(pixel: object) -> bool:
    channels = tuple(int(value) for value in pixel)  # type: ignore[arg-type]
    return len(channels) >= 3 and max(channels[:3]) <= _BASERENDER_BLACK_MATTE_THRESHOLD
