"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/zoomable_visual.py

Zoomable image frames for generated OPAL marimo notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import base64
import hashlib
import html
from typing import Any


def render_notebook_zoomable_image(
    *,
    mo: Any,
    image_bytes: bytes,
    mime_type: str,
    alt_text: str,
    caption: str,
    artifact_key: str,
    render_mode: str = "standard_visual",
) -> Any:
    """Render image bytes in an auto-fit frame with manual zoom controls."""

    encoded = base64.b64encode(image_bytes).decode("ascii")
    safe_alt_text = html.escape(alt_text, quote=True)
    safe_caption = html.escape(caption, quote=True)
    safe_render_mode = html.escape(render_mode, quote=True)
    safe_mime_type = html.escape(mime_type, quote=True)
    container_id = _container_id(artifact_key=artifact_key, image_bytes=image_bytes)
    image_id = f"{container_id}-image"
    image_style = "display:block; width:auto; max-width:none; max-height:none; height:auto; transform-origin:top left;"
    frame_html = zoom_frame_html(
        body_html=f"""
          {render_visual_zoom_controls(container_id)}
          <div id="{container_id}" data-render-mode="{safe_render_mode}"
               style="overflow:auto; width:100%; max-width:100%; height:calc(100vh - 2.6rem);
                      border:1px solid #d8dee4; border-radius:6px; background:#ffffff; padding:0.25rem;
                      box-sizing:border-box;">
            <img id="{image_id}" src="data:{safe_mime_type};base64,{encoded}" alt="{safe_alt_text}"
                 style="{image_style}" />
          </div>
          {visual_zoom_script(container_id=container_id, image_id=image_id)}
        """
    )
    caption_html = mo.Html(
        f"""
        <figcaption style="font-size:0.92rem; color:#57606a; margin-top:0.2rem; max-width:100%;
                          overflow-wrap:anywhere;">{safe_caption}</figcaption>
        """
    )
    return mo.vstack(
        [
            render_zoom_frame(
                mo=mo,
                frame_html=frame_html,
                title=f"Zoomable visual: {caption}",
                height_css=_container_height(render_mode),
                min_height_css=_container_min_height(render_mode),
            ),
            caption_html,
        ],
        gap=0.15,
    )


def render_zoom_frame(*, mo: Any, frame_html: str, title: str, height_css: str, min_height_css: str) -> Any:
    """Render an isolated visual frame so plot controls do not affect notebook layout."""

    safe_srcdoc = html.escape(frame_html, quote=True)
    safe_title = html.escape(title, quote=True)
    safe_height = html.escape(height_css, quote=True)
    safe_min_height = html.escape(min_height_css, quote=True)
    return mo.Html(
        f"""
        <iframe title="{safe_title}" srcdoc="{safe_srcdoc}"
                style="display:block; width:100%; max-width:100%; min-width:0; box-sizing:border-box;
                       height:{safe_height}; min-height:{safe_min_height};
                       border:0; margin:0; padding:0; background:#ffffff;"
                loading="lazy"></iframe>
        """
    )


def zoom_frame_html(*, body_html: str) -> str:
    """Return a minimal HTML document for a zoomable notebook visual frame."""

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
            max-width: 100%;
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
    """Return small frame-local zoom controls."""

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
    """Return frame-local JavaScript for auto-fit and manual zoom interactions."""

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
              const rect = image.getBoundingClientRect();
              const width = Math.max(320, Number(image.naturalWidth || 0) || rect.width || 0);
              const height = Math.max(240, Number(image.naturalHeight || 0) || rect.height || width * 0.62);
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
              image.style.width = baseWidth * scale + "px";
            }};
            const fitImage = function() {{
              setBaseSize();
              const available = availableSize();
              const fitScale = Math.min(available.width / baseWidth, available.height / baseHeight, 1.0);
              fittedToPanel = true;
              applyScale(fitScale, true);
              container.scrollLeft = 0;
              container.scrollTop = 0;
            }};
            const scheduleFit = function() {{
              window.requestAnimationFrame(fitImage);
            }};
            if (image.complete) {{
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


def _container_id(*, artifact_key: str, image_bytes: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(artifact_key.encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(image_bytes)
    return f"zoomable-visual-{digest.hexdigest()[:12]}"


def _container_height(render_mode: str) -> str:
    if render_mode == "compact_wide_visual":
        return "58vh"
    if render_mode == "wide_visual":
        return "88vh"
    return "78vh"


def _container_min_height(render_mode: str) -> str:
    if render_mode == "compact_wide_visual":
        return "420px"
    if render_mode == "wide_visual":
        return "560px"
    return "500px"


__all__ = [
    "render_notebook_zoomable_image",
    "render_visual_zoom_controls",
    "render_zoom_frame",
    "visual_zoom_script",
    "zoom_frame_html",
]
