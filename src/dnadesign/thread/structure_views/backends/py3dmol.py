"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/structure_views/backends/py3dmol.py

py3Dmol backend for browser structure views.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import html

from dnadesign.thread.structure_views.models import StructureViewModel, StructureViewSpec


def py3dmol_available() -> bool:
    """Return whether py3Dmol can be imported."""

    try:
        import py3Dmol  # noqa: F401
    except ImportError:
        return False
    return True


def render_py3dmol_structure_view(spec: StructureViewSpec) -> str:
    """Render a complete py3Dmol-backed HTML figure."""

    spec.validate()
    try:
        import py3Dmol
    except ImportError as exc:  # pragma: no cover - exercised when optional dependency is absent
        raise RuntimeError("py3Dmol is required for interactive browser structure views") from exc

    view = py3Dmol.view(width=int(spec.width), height=int(spec.height))
    if hasattr(view, "setBackgroundColor"):
        view.setBackgroundColor(spec.background_color)
    for index, model in enumerate(spec.models):
        view.addModel(model.structure_text, model.structure_format)
        view.setStyle({"model": index}, _style_for_model(spec, model))
    view.zoomTo()
    viewer_html = view._make_html()
    return _wrap_view_html(spec, viewer_html)


def _style_for_model(spec: StructureViewSpec, model: StructureViewModel) -> dict[str, dict[str, object]]:
    style: dict[str, object] = {"color": model.color}
    if model.opacity < 1.0:
        style["opacity"] = float(model.opacity)
    return {spec.style: style}


def _wrap_view_html(spec: StructureViewSpec, viewer_html: str) -> str:
    title = html.escape(spec.title)
    legend = "".join(_legend_item(model) for model in spec.models)
    srcdoc = html.escape(_viewer_document(viewer_html), quote=True)
    iframe_title = html.escape(f"Interactive structure view: {spec.title}", quote=True)
    return f"""
    <figure style="margin:0;">
      <div style="border:1px solid #d8dee4; border-radius:6px; background:#ffffff;
                  padding:0.55rem; width:100%; box-sizing:border-box;">
        <div style="font-weight:650; font-size:1rem; margin:0 0 0.35rem 0; color:#24292f;">{title}</div>
        <iframe title="{iframe_title}" srcdoc="{srcdoc}"
                sandbox="allow-scripts"
                referrerpolicy="no-referrer"
                style="display:block; width:100%; height:{int(spec.height) + 12}px;
                       border:0; background:#ffffff;"></iframe>
        <div style="display:flex; flex-wrap:wrap; gap:0.8rem; align-items:center;
                    margin-top:0.4rem; font-size:0.86rem; color:#57606a;">
          {legend}
        </div>
      </div>
    </figure>
    """


def _viewer_document(viewer_html: str) -> str:
    return f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          html, body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            background: #ffffff;
          }}
        </style>
      </head>
      <body>
        {viewer_html}
      </body>
    </html>
    """


def _legend_item(model: StructureViewModel) -> str:
    label = html.escape(model.label or model.model_id)
    color = html.escape(model.color)
    return (
        f'<span><span style="display:inline-block; width:0.72rem; height:0.72rem; '
        f"background:{color}; border:1px solid #57606a; vertical-align:-0.08rem; "
        f'margin-right:0.25rem;"></span>{label}</span>'
    )
