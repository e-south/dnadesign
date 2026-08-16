"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/notebooks/test_three_axis_camera_state.py

Exercises the generated three-axis camera binding against reactive plot failures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
import subprocess


def test_camera_restore_failure_reveals_plot_and_allows_later_binding() -> None:
    from dnadesign.opal.src.analysis.notebook_components.three_axis_camera_state import (
        render_three_axis_camera_state,
    )

    class _Mo:
        @staticmethod
        def iframe(text: str, *, width: str, height: str) -> dict[str, str]:
            return {"text": text, "width": width, "height": height}

    rendered = render_three_axis_camera_state(mo=_Mo(), revision="three_axis_scatter_v1:camera")
    match = re.search(r"<script>\s*(.*?)\s*</script>", rendered["text"], flags=re.DOTALL)
    assert match is not None

    completed = subprocess.run(
        ("node", "-e", _NODE_HARNESS),
        input=match.group(1),
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result == {
        "failed_plot_hidden": False,
        "later_plot_bound": True,
        "relayout_calls": 1,
        "script_error": None,
    }


_NODE_HARNESS = r"""
const fs = require("fs");
const source = fs.readFileSync(0, "utf8");
const revision = "three_axis_scatter_v1:camera";
const observers = [];
const elements = [];
const timers = [];
let relayoutCalls = 0;

class MutationObserver {
  constructor(callback) {
    this.callback = callback;
    observers.push(this);
  }
  observe() {}
  disconnect() {}
}

const document = {
  querySelectorAll() {
    return elements;
  },
};
const host = {
  document,
  MutationObserver,
  Plotly: {
    relayout() {
      relayoutCalls += 1;
      throw new Error("forced synchronous relayout failure");
    },
  },
  setTimeout(callback) {
    timers.push(callback);
  },
};
host.__opalThreeAxisCameras = {
  [revision]: {
    up: {x: 0, y: 0, z: 1},
    center: {x: 0.1, y: 0.2, z: 0},
    eye: {x: 2.2, y: -0.4, z: 0.8},
    projection: {type: "perspective"},
  },
};
global.parent = host;

const makePlot = () => ({
  classList: {contains: (value) => value === "js-plotly-plot"},
  dataset: {},
  layout: {
    scene: {
      uirevision: revision,
      camera: {eye: {x: 1.55, y: 1.55, z: 1.2}},
    },
  },
  _fullLayout: {
    scene: {camera: {eye: {x: 1.55, y: 1.55, z: 1.2}}},
  },
  on() {},
  style: {},
});

const failedPlot = makePlot();
elements.push(failedPlot);
let scriptError = null;
try {
  eval(source);
} catch (error) {
  scriptError = error.message;
}

host.Plotly.relayout = () => Promise.resolve();
const laterPlot = makePlot();
elements.push(laterPlot);
for (const observer of [...observers]) observer.callback();

setImmediate(() => {
  process.stdout.write(JSON.stringify({
    failed_plot_hidden: failedPlot.style.visibility === "hidden",
    later_plot_bound: laterPlot.dataset.opalCameraRevision === revision,
    relayout_calls: relayoutCalls,
    script_error: scriptError,
  }));
});
"""
