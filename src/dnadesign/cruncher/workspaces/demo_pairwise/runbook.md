## demo_pairwise Runbook

**Workspace Path**
- src/dnadesign/cruncher/workspaces/demo_pairwise/

**Regulators**
- [lexA, cpxR]

**Purpose**
- Build one merged motif source for optimization by combining curated RegulonDB site sets with local DAP-seq site sets where available.

**Run This Single Command**

Run this single command to do everything below:

    uv run cruncher workspaces run --runbook configs/runbook.yaml

This full runbook includes two study sweeps after the main sample/analyze/export flow and can take substantially longer than the base demo run.

Quick smoke path (main lifecycle only):

    uv run cruncher workspaces run --runbook configs/runbook.yaml --step reset_workspace --step config_summary --step fetch_sites_demo_local_meme --step fetch_sites_regulondb --step discover_motifs --step lock_targets --step parse_run --step sample_run --step analyze_summary --step export_sequences_latest --step render_logos

### Step-by-Step Commands

    set -euo pipefail
    cd src/dnadesign/cruncher/workspaces/demo_pairwise
    CONFIG="$PWD/configs/config.yaml"
    cruncher() { uv run cruncher "$@"; }

    # Canonical machine-runbook sequence (matches configs/runbook.yaml).
    # Canonical transient cleanup command for workspace hygiene.
    cruncher workspaces reset --root . --confirm
    # Optional config sanity check.
    cruncher config summary -c "$CONFIG"
    cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
    cruncher fetch sites --source regulondb --tf lexA --tf cpxR --update -c "$CONFIG"
    # Discover motifs from merged site evidence. This step merges all fetched site sets across sources and runs MEME OOPS into the workspace-specific source id.
    cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id demo_merged_meme_oops -c "$CONFIG"
    # Freeze resolved motif/site provenance for this set.
    # If you change catalog.source_preference or discovery --source-id, re-run cruncher lock -c "$CONFIG" before parse.
    cruncher lock -c "$CONFIG"
    cruncher parse --force-overwrite -c "$CONFIG"
    cruncher sample --force-overwrite -c "$CONFIG"
    cruncher analyze --summary -c "$CONFIG"
    cruncher export sequences --latest -c "$CONFIG"
    cruncher catalog logos --source demo_merged_meme_oops --set 1 -c "$CONFIG"
    cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite
    cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite

### Optional output checks

    find outputs/plots -maxdepth 1 -type f | sort
    ls -1 outputs/analysis/reports outputs/export
