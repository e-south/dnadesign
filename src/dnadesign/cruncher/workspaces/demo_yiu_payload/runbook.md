## demo_yiu_payload YIU Runbook

**Workspace Path**
- src/dnadesign/cruncher/workspaces/demo_yiu_payload/

**Purpose**
- Checked-in YIU demo for the v4 payload optimization and rendering workflow.
- Covers one minimal no-PWM spec and one sample-hit spec from one repo workspace.
- Uses the validate -> render -> show loop with workspace-local bundles.
- The workspace also includes a local PWM context sidecar under `motifs/example_pwm_context.yaml` for advanced experimentation.

**Run This Single Command**

    uv run cruncher workspaces run --workspace demo_yiu_payload --runbook configs/runbook.yaml

### Step-by-Step Commands

    set -euo pipefail
    cd src/dnadesign/cruncher/workspaces/demo_yiu_payload
    cruncher() { uv run cruncher "$@"; }

    # Standard machine-runbook sequence (matches configs/runbook.yaml).
    cruncher yiu validate --spec configs/yiu/example_payload.yiu.yaml
    cruncher yiu render --spec configs/yiu/example_payload.yiu.yaml --force-overwrite --emit-renders
    cruncher yiu show --bundle bundles/example_payload
    cruncher yiu validate --spec configs/yiu/tetr_monotypic_hit.yiu.yaml
    cruncher yiu render --spec configs/yiu/tetr_monotypic_hit.yiu.yaml --force-overwrite --emit-renders
    cruncher yiu show --bundle bundles/tetr_monotypic_hit

### Optional follow-up commands

    uv run cruncher yiu render --spec configs/yiu/example_payload.yiu.yaml --force-overwrite
    uv run cruncher yiu show --bundle bundles/example_payload
    uv run cruncher yiu render --spec configs/yiu/tetr_monotypic_hit.yiu.yaml --force-overwrite
    uv run cruncher yiu show --bundle bundles/tetr_monotypic_hit
