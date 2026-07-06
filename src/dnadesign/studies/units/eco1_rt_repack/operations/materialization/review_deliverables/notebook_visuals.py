"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook_visuals.py

Shared visual-frame helpers for the Eco1 review notebook.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import base64
import hashlib
import html
from pathlib import Path
from typing import Any


def render_image(row: dict[str, Any], *, mo: Any, media_path: Path) -> Any:
    mime_type = "image/svg+xml" if media_path.suffix.lower() == ".svg" else "image/png"
    encoded = base64.b64encode(media_path.read_bytes()).decode("ascii")
    alt_text = html.escape(str(row["alt_text"]), quote=True)
    caption = html.escape(_format_deliverable_label(row), quote=True)
    render_mode = html.escape(str(row.get("render_mode") or "standard_visual"), quote=True)
    container_id = f"zoomable-visual-{hashlib.sha256(str(media_path).encode()).hexdigest()[:12]}"
    image_id = f"{container_id}-image"
    container_max_height = _container_height(row)
    container_min_height = _container_min_height(row)
    image_style = "display:block; width:auto; max-width:none; max-height:none; height:auto; transform-origin:top left;"
    zoom_controls = render_visual_zoom_controls(container_id)
    zoom_script = visual_zoom_script(container_id=container_id, image_id=image_id)
    frame_html = zoom_frame_html(
        body_html=f"""
          {zoom_controls}
          <div id="{container_id}" data-render-mode="{render_mode}"
               style="overflow:auto; width:100%; height:calc(100vh - 2.6rem);
                      border:1px solid #d8dee4; border-radius:6px; background:#ffffff; padding:0.25rem;
                      box-sizing:border-box;">
            <img id="{image_id}" src="data:{mime_type};base64,{encoded}" alt="{alt_text}"
                 style="{image_style}" />
          </div>
          {zoom_script}
        """
    )
    caption_html = mo.Html(
        f"""
        <figcaption style="font-size:0.92rem; color:#57606a; margin-top:0.2rem;">{caption}</figcaption>
        """
    )
    return mo.vstack(
        [
            render_zoom_frame(
                mo=mo,
                frame_html=frame_html,
                title=f"Zoomable visual: {caption}",
                height_css=container_max_height,
                min_height_css=container_min_height,
            ),
            caption_html,
        ],
        gap=0.15,
    )


def render_zoom_frame(*, mo: Any, frame_html: str, title: str, height_css: str, min_height_css: str = "520px") -> Any:
    safe_srcdoc = html.escape(frame_html, quote=True)
    safe_title = html.escape(title, quote=True)
    safe_height = html.escape(height_css, quote=True)
    safe_min_height = html.escape(min_height_css, quote=True)
    return mo.Html(
        f"""
        <iframe title="{safe_title}" srcdoc="{safe_srcdoc}"
                style="display:block; width:100%; height:{safe_height}; min-height:{safe_min_height};
                       border:0; margin:0; padding:0; background:#ffffff;"
                loading="lazy"></iframe>
        """
    )


def zoom_frame_html(*, body_html: str) -> str:
    return f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <style>
          html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: #ffffff;
            color: #24292f;
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }}
          * {{ box-sizing: border-box; }}
        </style>
      </head>
      <body>
        {body_html}
      </body>
    </html>
    """


def render_visual_zoom_controls(container_id: str) -> str:
    safe_container_id = html.escape(container_id, quote=True)
    return f"""
          <div style="display:flex; gap:0.35rem; align-items:center; justify-content:flex-end;
                      margin:0 0 0.25rem 0;">
            <button type="button" data-zoom-target="{safe_container_id}" data-zoom-action="out"
                    aria-label="Zoom out"
                    style="border:1px solid #d8dee4; background:#ffffff; border-radius:4px;
                           padding:0.12rem 0.45rem; line-height:1.25; cursor:pointer;">-</button>
            <button type="button" data-zoom-target="{safe_container_id}" data-zoom-action="fit"
                    aria-label="Fit image to panel"
                    style="border:1px solid #d8dee4; background:#ffffff; border-radius:4px;
                           padding:0.12rem 0.45rem; line-height:1.25; cursor:pointer;">Fit</button>
            <button type="button" data-zoom-target="{safe_container_id}" data-zoom-action="in"
                    aria-label="Zoom in"
                    style="border:1px solid #d8dee4; background:#ffffff; border-radius:4px;
                           padding:0.12rem 0.45rem; line-height:1.25; cursor:pointer;">+</button>
          </div>
    """


def visual_zoom_script(*, container_id: str, image_id: str) -> str:
    safe_container_id = html.escape(container_id, quote=True)
    safe_image_id = html.escape(image_id, quote=True)
    return f"""
          <script>
          (function() {{
            const container = document.getElementById("{safe_container_id}");
            const image = document.getElementById("{safe_image_id}");
            if (!container || !image || container.dataset.zoomReady === "true") {{
              return;
            }}
            container.dataset.zoomReady = "true";
            const minScale = 0.2;
            const maxScale = 24.0;
            let scale = 1.0;
            let baseWidth = 0;
            let baseHeight = 0;
            let fittedToPanel = true;
            const numeric = function(value) {{
              const parsed = parseFloat(value || "0");
              return Number.isFinite(parsed) ? parsed : 0;
            }};
            const intrinsicSize = function() {{
              let width = Number(image.naturalWidth || 0);
              let height = Number(image.naturalHeight || 0);
              if ((!width || !height) && image.viewBox && image.viewBox.baseVal) {{
                width = Number(image.viewBox.baseVal.width || width);
                height = Number(image.viewBox.baseVal.height || height);
              }}
              if ((!width || !height) && image.getAttribute("viewBox")) {{
                const values = image.getAttribute("viewBox").trim().split(/\\s+/).map(Number);
                if (values.length === 4 && values.every(Number.isFinite)) {{
                  width = width || values[2];
                  height = height || values[3];
                }}
              }}
              const rect = image.getBoundingClientRect();
              width = Math.max(320, width || rect.width || 0);
              height = Math.max(240, height || rect.height || width * 0.62);
              return {{ width, height }};
            }};
            const availableSize = function() {{
              const computedStyle = window.getComputedStyle(container);
              const horizontalPadding = numeric(computedStyle.paddingLeft) + numeric(computedStyle.paddingRight);
              const verticalPadding = numeric(computedStyle.paddingTop) + numeric(computedStyle.paddingBottom);
              return {{
                width: Math.max(280, container.clientWidth - horizontalPadding),
                height: Math.max(220, container.clientHeight - verticalPadding),
              }};
            }};
            const setBaseSize = function() {{
              const size = intrinsicSize();
              baseWidth = size.width;
              baseHeight = size.height;
              image.style.width = baseWidth + "px";
              image.style.height = "auto";
            }};
            const applyScale = function(nextScale, keepFitted) {{
              if (!baseWidth || !baseHeight) {{
                setBaseSize();
              }}
              if (!keepFitted) {{
                fittedToPanel = false;
              }}
              scale = Math.max(minScale, Math.min(maxScale, nextScale));
              image.dataset.zoomScale = String(scale);
              image.style.width = (baseWidth * scale) + "px";
            }};
            const fitImage = function() {{
              setBaseSize();
              const available = availableSize();
              const fitScale = Math.min(available.width / baseWidth, available.height / baseHeight, 1.0);
              fittedToPanel = true;
              applyScale(fitScale, true);
              image.dataset.zoomScale = String(scale);
              container.scrollLeft = 0;
              container.scrollTop = 0;
            }};
            const scheduleFit = function() {{
              window.requestAnimationFrame(fitImage);
            }};
            if (image.complete || image.tagName.toLowerCase() === "svg") {{
              scheduleFit();
            }} else {{
              image.addEventListener("load", scheduleFit, {{ once: true }});
              scheduleFit();
            }}
            container.addEventListener("wheel", function(event) {{
              if (!event.ctrlKey && !event.metaKey) {{
                return;
              }}
              event.preventDefault();
              const factor = event.deltaY < 0 ? 1.14 : 0.88;
              applyScale(scale * factor, false);
            }}, {{ passive: false }});
            document.querySelectorAll('[data-zoom-target="{safe_container_id}"]').forEach(function(button) {{
              button.addEventListener("click", function() {{
                const action = button.getAttribute("data-zoom-action");
                if (action === "fit") {{
                  fitImage();
                }} else if (action === "in") {{
                  applyScale(scale * 1.25, false);
                }} else if (action === "out") {{
                  applyScale(scale * 0.8, false);
                }}
              }});
            }});
            window.addEventListener("resize", function() {{
              if (fittedToPanel) {{
                fitImage();
              }}
            }}, {{ passive: true }});
          }})();
          </script>
    """


def is_wide_visual(row: dict[str, Any]) -> bool:
    """Return whether a visual should preserve horizontal detail instead of squeezing to fit."""

    return str(row.get("render_mode") or "") in {"wide_visual", "compact_wide_visual"}


def _container_height(row: dict[str, Any]) -> str:
    render_mode = str(row.get("render_mode") or "")
    if render_mode == "compact_wide_visual":
        return "58vh"
    return "88vh" if is_wide_visual(row) else "78vh"


def _container_min_height(row: dict[str, Any]) -> str:
    render_mode = str(row.get("render_mode") or "")
    if render_mode == "compact_wide_visual":
        return "420px"
    return "560px" if is_wide_visual(row) else "500px"


def _format_deliverable_label(row: dict[str, Any]) -> str:
    row_title = str(row.get("title") or "")
    if row_title:
        return row_title
    return str(row.get("deliverable_id") or "").replace("_", " ").title()
