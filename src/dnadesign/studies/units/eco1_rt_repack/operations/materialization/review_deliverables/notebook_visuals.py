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
    is_wide = is_wide_visual(row)
    render_mode = html.escape(str(row.get("render_mode") or "standard_visual"), quote=True)
    container_id = f"zoomable-visual-{hashlib.sha256(str(media_path).encode()).hexdigest()[:12]}"
    image_id = f"{container_id}-image"
    container_max_height = "86vh" if is_wide else "82vh"
    image_style = "display:block; width:100%; max-width:none; max-height:none; height:auto; transform-origin:top left;"
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
            ),
            caption_html,
        ],
        gap=0.15,
    )


def render_zoom_frame(*, mo: Any, frame_html: str, title: str, height_css: str) -> Any:
    safe_srcdoc = html.escape(frame_html, quote=True)
    safe_title = html.escape(title, quote=True)
    safe_height = html.escape(height_css, quote=True)
    return mo.Html(
        f"""
        <iframe title="{safe_title}" srcdoc="{safe_srcdoc}"
                style="display:block; width:100%; height:{safe_height}; min-height:520px;
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
            const setBaseWidth = function() {{
              const rect = image.getBoundingClientRect();
              const computedStyle = window.getComputedStyle(container);
              const horizontalPadding = parseFloat(computedStyle.paddingLeft || "0") +
                parseFloat(computedStyle.paddingRight || "0");
              baseWidth = Math.max(320, container.clientWidth - horizontalPadding, rect.width || 0);
              image.style.width = baseWidth + "px";
            }};
            const applyScale = function(nextScale) {{
              if (!baseWidth) {{
                setBaseWidth();
              }}
              scale = Math.max(minScale, Math.min(maxScale, nextScale));
              image.dataset.zoomScale = String(scale);
              image.style.width = (baseWidth * scale) + "px";
            }};
            const fitImage = function() {{
              scale = 1.0;
              setBaseWidth();
              image.dataset.zoomScale = String(scale);
              container.scrollLeft = 0;
              container.scrollTop = 0;
            }};
            window.requestAnimationFrame(fitImage);
            container.addEventListener("wheel", function(event) {{
              if (!event.ctrlKey && !event.metaKey) {{
                return;
              }}
              event.preventDefault();
              const factor = event.deltaY < 0 ? 1.14 : 0.88;
              applyScale(scale * factor);
            }}, {{ passive: false }});
            document.querySelectorAll('[data-zoom-target="{safe_container_id}"]').forEach(function(button) {{
              button.addEventListener("click", function() {{
                const action = button.getAttribute("data-zoom-action");
                if (action === "fit") {{
                  fitImage();
                }} else if (action === "in") {{
                  applyScale(scale * 1.25);
                }} else if (action === "out") {{
                  applyScale(scale * 0.8);
                }}
              }});
            }});
            window.addEventListener("resize", function() {{
              if (scale === 1.0) {{
                fitImage();
              }}
            }}, {{ passive: true }});
          }})();
          </script>
    """


def is_wide_visual(row: dict[str, Any]) -> bool:
    """Return whether a visual should preserve horizontal detail instead of squeezing to fit."""

    return str(row.get("render_mode") or "") == "wide_visual"


def _format_deliverable_label(row: dict[str, Any]) -> str:
    row_title = str(row.get("title") or "")
    if row_title:
        return row_title
    return str(row.get("deliverable_id") or "").replace("_", " ").title()
