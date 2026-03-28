## demo_yiu_circularized YIU Runbook

**Workspace Path**
- src/dnadesign/cruncher/workspaces/demo_yiu_circularized/

**Purpose**
- Canonical checked-in YIU demo for the split-payload circularized flow.
- Covers validate, explicit materialization, trace-alias materialization, and solve from one repo workspace.

**Run This Single Command**

    uv run cruncher workspaces run --workspace demo_yiu_circularized --runbook configs/runbook.yaml

### Step-by-Step Commands

    set -euo pipefail
    cd src/dnadesign/cruncher/workspaces/demo_yiu_circularized
    cruncher() { uv run cruncher "$@"; }

    # Standard machine-runbook sequence (matches configs/runbook.yaml).
    cruncher yiu validate --spec configs/yiu/example_canonical_circularized.yiu.yaml
    cruncher yiu design --spec configs/yiu/example_canonical_circularized.yiu.yaml --force-overwrite
    cruncher yiu trace --spec configs/yiu/example_canonical_circularized.yiu.yaml --force-overwrite
    cruncher yiu solve --spec configs/yiu/example_canonical_circularized.yiu.solve.yaml --force-overwrite

### Optional follow-up commands

    DESIGN_ID="$(ls -1 outputs/yiu/explicit/example_canonical_circularized | tail -n 1)"
    SOLVE_ID="$(ls -1 outputs/yiu/solve/example_canonical_circularized | tail -n 1)"
    uv run cruncher yiu show --run "outputs/yiu/explicit/example_canonical_circularized/$DESIGN_ID"
    uv run cruncher yiu show --run "outputs/yiu/solve/example_canonical_circularized/$SOLVE_ID"
    uv run cruncher visuals validate --job "outputs/yiu/explicit/example_canonical_circularized/$DESIGN_ID/published/baserender_jobs/circularized_payload_candidate.job.yaml"
    uv run cruncher visuals run --job "outputs/yiu/explicit/example_canonical_circularized/$DESIGN_ID/published/baserender_jobs/circularized_payload_candidate.job.yaml"
