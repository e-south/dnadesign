from __future__ import annotations

from importlib import import_module
from io import BytesIO
from typing import Any, Mapping


def render_notebook_baserender_record(record_row: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
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
    try:
        from PIL import Image

        render_route = str(contract.get("render_route") or "figure")
        if render_route == "sequence_panel":
            panel = render_sequence_panel_image(
                dict(record_row),
                adapter_kind=adapter_kind,
                adapter_columns=dict(contract.get("adapter_columns") or {}),
                adapter_policies=dict(contract.get("adapter_policies") or {}),
                style_overrides=style_overrides,
                target_width_px=int(contract.get("target_width_px") or 2600),
                target_height_px=int(contract.get("target_height_px") or 430),
                vertical_anchor=str(contract.get("vertical_anchor") or "top"),
                canvas_top_pad_px=int(contract.get("canvas_top_pad_px") or 8),
            )
            image_bytes = _encode_opaque_white_png(Image.fromarray(panel.image))
            sequence_length = int(panel.diagnostics.sequence_length_bp)
            feature_count = int(panel.diagnostics.feature_count)
        else:
            record = baserender.adapt_record(
                dict(record_row),
                adapter_kind=adapter_kind,
                adapter_columns=dict(contract.get("adapter_columns") or {}),
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
            image_bytes = buffer.getvalue()
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
    caption = str(contract.get("caption") or "BaseRender record view.")
    return {
        "record_id": record_id,
        "image_bytes": image_bytes,
        "caption": f"{caption} Record `{record_id}`.",
        "alt_text": _format_alt_text(
            contract.get("alt_text_template") or caption,
            record_id=record_id,
            sequence_length=sequence_length,
            feature_count=feature_count,
        ),
        "sequence_length": sequence_length,
        "feature_count": feature_count,
    }


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
    canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    canvas.alpha_composite(rgba)
    canvas.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()
