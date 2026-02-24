## pairwise_cpxr_soxr Runbook

**Workspace Path**
- src/dnadesign/cruncher/workspaces/pairwise_cpxr_soxr/

**Regulators**
- [cpxR, soxR]

**Purpose**
- Run the full Cruncher flow for this regulator set: fetch sites, discover motifs, lock inputs, sample sequences, analyze outputs, and export handoff artifacts.

**Run This Single Command**

Run this single command to do everything below:

    uv run cruncher workspaces run --runbook configs/runbook.yaml

### Step-by-Step Commands

    set -euo pipefail
    cd src/dnadesign/cruncher/workspaces/pairwise_cpxr_soxr
    CONFIG="$PWD/configs/config.yaml"
    cruncher() { uv run cruncher "$@"; }

    # Canonical machine-runbook sequence (matches configs/runbook.yaml).
    # Canonical transient cleanup command for workspace hygiene.
    cruncher workspaces reset --root . --confirm
    # Optional config sanity check.
    cruncher config summary -c "$CONFIG"
    cruncher fetch sites --source demo_local_meme --tf cpxR --tf soxR --update -c "$CONFIG"
    cruncher fetch sites --source regulondb --tf cpxR --tf soxR --update -c "$CONFIG"
    # Discover motifs from merged site evidence. This step merges all fetched site sets across sources and runs MEME OOPS into the workspace-specific source id.
    cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id pairwise_cpxr_soxr_merged_meme_oops -c "$CONFIG"
    # Freeze resolved motif/site provenance for this set.
    # If you change catalog.source_preference or discovery --source-id, re-run cruncher lock -c "$CONFIG" before parse.
    cruncher lock -c "$CONFIG"
    cruncher parse --force-overwrite -c "$CONFIG"
    cruncher sample --force-overwrite -c "$CONFIG"
    cruncher analyze --summary -c "$CONFIG"
    cruncher export sequences --latest -c "$CONFIG"
    cruncher catalog logos --source pairwise_cpxr_soxr_merged_meme_oops --set 1 -c "$CONFIG"
    cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite
    cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite

### Optional output checks

    find outputs/plots -maxdepth 1 -type f | sort
    ls -1 outputs/analysis/reports outputs/export
