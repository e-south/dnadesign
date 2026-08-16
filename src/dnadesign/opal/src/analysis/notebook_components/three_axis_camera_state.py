"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/three_axis_camera_state.py

Preserves a three-axis Plotly camera across reactive notebook renders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from typing import Any


def render_three_axis_camera_state(*, mo: Any, revision: str) -> Any:
    """Bind one browser-local camera state to replacement Plotly components."""

    revision_json = json.dumps(revision)
    return mo.iframe(
        f"""
<script>
(() => {{
  const revision = {revision_json};
  const host = parent;
  const cameras = host.__opalThreeAxisCameras ??= Object.create(null);

  const plots = () => {{
    const found = [];
    const visit = (root) => {{
      for (const element of root.querySelectorAll("*")) {{
        if (element.classList?.contains("js-plotly-plot")) found.push(element);
        if (element.shadowRoot) visit(element.shadowRoot);
      }}
    }};
    visit(host.document);
    return found;
  }};

  const cameraMatches = (current, saved) =>
    JSON.stringify(current ?? null) === JSON.stringify(saved ?? null);

  const bind = (plot) => {{
    if (plot.layout?.scene?.uirevision !== revision) return;
    const saved = cameras[revision];
    if (saved && !cameraMatches(plot.layout.scene.camera, saved) && host.Plotly?.relayout) {{
      host.Plotly.relayout(plot, {{"scene.camera": saved}});
    }}
    if (plot.dataset.opalCameraRevision === revision) return;
    plot.on?.("plotly_relayout", (change) => {{
      if (Object.keys(change).some((key) => key.startsWith("scene.camera"))) {{
        cameras[revision] = JSON.parse(JSON.stringify(plot.layout.scene.camera));
      }}
    }});
    plot.dataset.opalCameraRevision = revision;
  }};

  let attempts = 40;
  const attach = () => {{
    for (const plot of plots()) bind(plot);
    attempts -= 1;
    if (attempts > 0) host.setTimeout(attach, 25);
  }};
  attach();
}})();
</script>
""",
        width="0",
        height="0",
    )


__all__ = ["render_three_axis_camera_state"]
