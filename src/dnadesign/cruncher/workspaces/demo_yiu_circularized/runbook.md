## demo_yiu_circularized YIU Runbook

**Workspace Path**
- src/dnadesign/cruncher/workspaces/demo_yiu_circularized/

**Purpose**
- Checked-in YIU demo for the circularized payload workflow.
- Covers validate, trace, solve, show, and render from one repo workspace without the legacy design alias.

**Run This Single Command**

    uv run cruncher workspaces run --workspace demo_yiu_circularized --runbook configs/runbook.yaml

### Step-by-Step Commands

    set -euo pipefail
    cd src/dnadesign/cruncher/workspaces/demo_yiu_circularized
    cruncher() { uv run cruncher "$@"; }

    # Standard machine-runbook sequence (matches configs/runbook.yaml).
    cruncher yiu validate --spec configs/yiu/example_reference_circularized.yiu.yaml
    cruncher yiu trace --spec configs/yiu/example_reference_circularized.yiu.yaml --force-overwrite --emit-renders
    cruncher yiu solve --spec configs/yiu/example_reference_circularized.yiu.solve.yaml --force-overwrite --emit-renders

### Optional follow-up commands

    WORKFLOW_NAME="example_reference_circularized"
    TRACE_ID="$(ls -1 "outputs/yiu/explicit/$WORKFLOW_NAME" | tail -n 1)"
    SOLVE_ID="$(ls -1 "outputs/yiu/solve/$WORKFLOW_NAME" | tail -n 1)"
    uv run cruncher yiu show --run "outputs/yiu/explicit/$WORKFLOW_NAME/$TRACE_ID"
    uv run cruncher yiu show --run "outputs/yiu/solve/$WORKFLOW_NAME/$SOLVE_ID"
    uv run cruncher yiu render --run "outputs/yiu/explicit/$WORKFLOW_NAME/$TRACE_ID"
    uv run cruncher yiu render --run "outputs/yiu/solve/$WORKFLOW_NAME/$SOLVE_ID"
