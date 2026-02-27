## portfolios Runbook

**Workspace Path**
- src/dnadesign/cruncher/workspaces/portfolios/

**Regulators**
- N/A (portfolio aggregation workspace).

**Purpose**
- Prepare source workspaces as needed, then aggregate elite and study outputs into one portfolio-level handoff package.

### Portfolio Specs
- configs/master_all_workspaces.portfolio.yaml

**Run This Single Command**

Run this single command to do everything below:

    uv run cruncher workspaces run --runbook configs/runbook.yaml

### Step-by-Step Commands

    set -euo pipefail
    cd src/dnadesign/cruncher/workspaces/portfolios
    cruncher() { uv run cruncher "$@"; }

    # Canonical machine-runbook sequence (matches configs/runbook.yaml).
    cruncher portfolio run --spec configs/master_all_workspaces.portfolio.yaml --prepare-ready skip --force-overwrite

### Optional follow-up commands

    # Inspect an individual portfolio run id produced by the command above.
    cruncher portfolio show --run outputs/portfolios/master_all_workspaces/<portfolio_id>

    # Force full source-workspace re-prepare before aggregation.
    cruncher portfolio run --spec configs/master_all_workspaces.portfolio.yaml --prepare-ready rerun --force-overwrite

    # Verify handoff tables.
    ls -1 outputs/portfolios/master_all_workspaces/<portfolio_id>/tables/table__handoff_windows_long.*
    ls -1 outputs/portfolios/master_all_workspaces/<portfolio_id>/tables/table__handoff_elites_summary.*
    ls -1 outputs/portfolios/master_all_workspaces/<portfolio_id>/tables/table__handoff_sequence_length.*
