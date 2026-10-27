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
  const observerRegistry = host.__opalThreeAxisCameraObservers ??= Object.create(null);
  for (const observer of observerRegistry[revision] ?? []) observer.disconnect();
  const observers = [];
  const observedRoots = new WeakSet();
  observerRegistry[revision] = observers;

  const cameraMatches = (current, saved) =>
    JSON.stringify(current ?? null) === JSON.stringify(saved ?? null);

  const reveal = (plot) => {{
    plot.style.visibility = "";
  }};

  const bind = (plot) => {{
    if (plot.layout?.scene?.uirevision !== revision) return;
    const saved = cameras[revision];
    const activeCamera = plot._fullLayout?.scene?.camera;
    if (saved && activeCamera && !cameraMatches(activeCamera, saved) && host.Plotly?.relayout) {{
      plot.style.visibility = "hidden";
      try {{
        Promise.resolve(host.Plotly.relayout(plot, {{"scene.camera": saved}})).then(
          () => reveal(plot),
          () => reveal(plot),
        );
      }} catch {{
        reveal(plot);
      }}
    }} else {{
      reveal(plot);
    }}
    if (plot.dataset.opalCameraRevision === revision) return;
    plot.on?.("plotly_beforeplot", () => {{
      const retained = cameras[revision];
      if (retained && plot.layout?.scene && !cameraMatches(plot.layout.scene.camera, retained)) {{
        plot.layout.scene.camera = JSON.parse(JSON.stringify(retained));
      }}
    }});
    plot.on?.("plotly_relayout", (change) => {{
      if (Object.keys(change).some((key) => key.startsWith("scene.camera"))) {{
        cameras[revision] = JSON.parse(JSON.stringify(plot.layout.scene.camera));
      }}
    }});
    plot.dataset.opalCameraRevision = revision;
  }};

  let scanning = false;
  const observe = (root) => {{
    if (observedRoots.has(root)) return;
    const observer = new host.MutationObserver(() => scan());
    observer.observe(root, {{childList: true, subtree: true}});
    observers.push(observer);
    observedRoots.add(root);
  }};
  const scan = () => {{
    if (scanning) return;
    scanning = true;
    try {{
      observe(host.document);
      const visit = (root) => {{
        for (const element of root.querySelectorAll("*")) {{
          if (element.classList?.contains("js-plotly-plot")) bind(element);
          if (element.shadowRoot) {{
            observe(element.shadowRoot);
            visit(element.shadowRoot);
          }}
        }}
      }};
      visit(host.document);
    }} finally {{
      scanning = false;
    }}
  }};

  let attempts = 40;
  const attach = () => {{
    scan();
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
