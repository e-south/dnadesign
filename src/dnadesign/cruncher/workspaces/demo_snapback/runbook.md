## demo_snapback Snapback Runbook

**Workspace Path**
- src/dnadesign/cruncher/workspaces/demo_snapback/

**Purpose**
- Checked-in snapback demo for one explicit Bpu10I design, one released-product precursor demo, and one broader v3 catalog-scan solve workflow.
- Uses stable workspace output roots under `outputs/design`, `outputs/released_design`, and `outputs/solve`.
- The explicit lane uses the local `Nt.Bpu10I` overlay; the solve lane searches built-in `neb_nicking_v1` plus `thermo_nicking_v1`.
- The released-product lane uses local `Nx.Exact7` and `Re.Exact` overlays so the retained post-release geometry is deterministic and self-contained.
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
    cruncher snapback released-design --spec configs/snapback/demo_released_origin_033.released.snapback.yaml --force-overwrite
    cruncher snapback released-show --run outputs/released_design
    cruncher snapback released-target-search --workspace-root . --nick-additional-path inputs/nickases/local.nickases.yaml --release-additional-path inputs/release_enzymes/local.release.yaml --nick-boundary 0 --paired-bp 3 --cap-nt 3 --json
    cruncher snapback solve --spec configs/snapback/demo_teto_catalog_scan.snapback.solve.yaml --force-overwrite
    cruncher snapback show --run outputs/solve

### Optional follow-up commands

    uv run baserender job run outputs/design/baserender_jobs/snapback_triptych.job.yaml
    uv run baserender job run outputs/solve/analysis/materialized_hits/hit_01/baserender_jobs/snapback_triptych.job.yaml
    uv run cruncher snapback show --run outputs/design --json
    uv run cruncher snapback released-show --run outputs/released_design --json
    uv run cruncher snapback show --run outputs/solve --json
