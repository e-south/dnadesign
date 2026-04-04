## demo_yiu_payload YIU Runbook

**Workspace Path**
- src/dnadesign/cruncher/workspaces/demo_yiu_payload/

**Purpose**
- Checked-in YIU demo for the v4 payload optimization and rendering workflow.
- Covers one minimal no-PWM spec in a user-sequence-only YIU workspace.
- Uses the validate -> render -> show loop with outputs rooted under `outputs/`.
- Sample-hit YIU demos now live beside their source Sample outputs in the monotypic workspaces.
- The workspace also includes a generic PWM context sidecar under `motifs/example_pwm_context.yaml` for extra experimentation.

**Run This Single Command**

    uv run cruncher workspaces run --workspace demo_yiu_payload --runbook configs/runbook.yaml

### Step-by-Step Commands

    set -euo pipefail
    cd src/dnadesign/cruncher/workspaces/demo_yiu_payload
    cruncher() { uv run cruncher "$@"; }

    # Standard machine-runbook sequence (matches configs/runbook.yaml).
    cruncher yiu validate --spec configs/yiu/example_payload.yiu.yaml
    cruncher yiu render --spec configs/yiu/example_payload.yiu.yaml --force-overwrite --emit-renders
    cruncher yiu show --bundle outputs/example_payload

### Optional follow-up commands

    uv run cruncher yiu render --spec configs/yiu/example_payload.yiu.yaml --force-overwrite
    uv run cruncher yiu show --bundle outputs/example_payload
    cp configs/yiu/example_payload.advanced_pwm.example.yaml configs/yiu/example_payload_pwm.yiu.yaml
    uv run cruncher yiu render --spec configs/yiu/example_payload_pwm.yiu.yaml --force-overwrite --emit-renders
    uv run cruncher yiu show --bundle outputs/example_payload_pwm
