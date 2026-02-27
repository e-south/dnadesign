## study_constitutive_sigma_panel Runbook

**Workspace Path**
- src/dnadesign/densegen/workspaces/study_constitutive_sigma_panel/

**Regulators**
- [lacI, araC] exclusion motifs for background filtering

**Purpose**
- Run the constitutive sigma70 panel with fixed-element expansion and plot/notebook outputs from one workspace.

**Runbook command**

Run this command from the workspace root:

    ./runbook.sh

### Step-by-Step Commands

    set -euo pipefail
    CONFIG="$PWD/config.yaml"
    USR_REGISTRY="$PWD/outputs/usr_datasets/registry.yaml"
    ROOT_REGISTRY="$(git rev-parse --show-toplevel)/src/dnadesign/usr/datasets/registry.yaml"

    if [ ! -f "$USR_REGISTRY" ]; then
      mkdir -p "$(dirname "$USR_REGISTRY")"
      cp "$ROOT_REGISTRY" "$USR_REGISTRY"
    fi

    pixi run fimo --version
    pixi run dense validate-config --probe-solver -c "$CONFIG"
    pixi run dense run --fresh --no-plot -c "$CONFIG"
    pixi run dense inspect run --events --library -c "$CONFIG"
    pixi run dense plot -c "$CONFIG"
    pixi run dense notebook generate -c "$CONFIG"
    uv run marimo check "$PWD/outputs/notebooks/densegen_run_overview.py"

### Optional notebook open

    pixi run dense notebook run -c "$CONFIG"

### Optional artifact refresh from Cruncher

    uv run cruncher catalog export-densegen --set 1 --densegen-workspace "$PWD" -c "$(git rev-parse --show-toplevel)/src/dnadesign/cruncher/workspaces/pairwise_laci_arac/configs/config.yaml"

### Optional workspace reset

    pixi run dense campaign-reset -c "$CONFIG"
