## demo_snapback Snapback Runbook

**Workspace Path**
- src/dnadesign/cruncher/workspaces/demo_snapback/

**Purpose**
- Checked-in snapback demo for one explicit Bpu10I design and one broader v3 catalog-scan solve workflow.
- Uses stable workspace output roots under `outputs/design` and `outputs/solve`.
- The explicit lane uses the local `Nt.Bpu10I` overlay; the solve lane searches built-in `neb_nicking_v1` plus `thermo_nicking_v1`.
- Keeps materialized solve hits inside `outputs/solve/analysis/materialized_hits/` instead of nested run-id bundles.

**Run This Single Command**

    uv run cruncher workspaces run --workspace demo_snapback --runbook configs/runbook.yaml

### Step-by-Step Commands

    set -euo pipefail
    cd src/dnadesign/cruncher/workspaces/demo_snapback
    cruncher() { uv run cruncher "$@"; }

    # Standard machine-runbook sequence (matches configs/runbook.yaml).
    cruncher snapback validate --spec configs/snapback/demo_teto_bpu10i_cap.snapback.yaml
    cruncher snapback design --spec configs/snapback/demo_teto_bpu10i_cap.snapback.yaml --force-overwrite
    cruncher snapback show --run outputs/design
    cruncher snapback solve --spec configs/snapback/demo_teto_catalog_scan.snapback.solve.yaml --force-overwrite
    cruncher snapback show --run outputs/solve

### Optional follow-up commands

    uv run baserender job run outputs/design/baserender_jobs/snapback_triptych.job.yaml
    uv run baserender job run outputs/solve/analysis/materialized_hits/hit_01/baserender_jobs/snapback_triptych.job.yaml
    uv run cruncher snapback show --run outputs/design --json
    uv run cruncher snapback show --run outputs/solve --json
