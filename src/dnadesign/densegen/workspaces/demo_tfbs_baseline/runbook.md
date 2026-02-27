## demo_tfbs_baseline Runbook

**Workspace Path**
- src/dnadesign/densegen/workspaces/demo_tfbs_baseline/

**Regulators**
- [none required]

**Purpose**
- Run the smallest DenseGen baseline from inside one workspace without root-level path exports.

**Runbook command**

Run this command from the workspace root:

    ./runbook.sh

### Step-by-Step Commands

    set -euo pipefail
    CONFIG="$PWD/config.yaml"

    uv run dense validate-config --probe-solver -c "$CONFIG"
    uv run dense run --fresh --no-plot -c "$CONFIG"
    uv run dense inspect run --events --library -c "$CONFIG"
    uv run dense plot -c "$CONFIG"
    uv run dense notebook generate -c "$CONFIG"
    uv run marimo check "$PWD/outputs/notebooks/densegen_run_overview.py"

### Optional notebook open

    uv run dense notebook run -c "$CONFIG"

### Optional workspace reset

    uv run dense campaign-reset -c "$CONFIG"
