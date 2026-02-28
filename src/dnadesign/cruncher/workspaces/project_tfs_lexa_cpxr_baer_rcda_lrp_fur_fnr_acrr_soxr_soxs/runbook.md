## project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs Runbook

**Workspace Path**
- src/dnadesign/cruncher/workspaces/project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs/

**Regulators**
- [lexA, cpxR, baeR, rcdA, lrp, fur, fnr, acrR, soxR, soxS]

**Purpose**
- Build one merged motif source for project-scale optimization by combining curated RegulonDB site sets with all available local DAP-seq site sets and BaeR ChIP-exo evidence.

**Run This Single Command**

Run this single command to do everything below:

    uv run cruncher workspaces run --runbook configs/runbook.yaml

### Step-by-Step Commands

    set -euo pipefail
    cd src/dnadesign/cruncher/workspaces/project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs
    CONFIG="$PWD/configs/config.yaml"
    cruncher() { uv run cruncher "$@"; }

    # Standard machine-runbook sequence (matches configs/runbook.yaml).
    # Standard transient cleanup command for workspace hygiene.
    cruncher workspaces reset --root . --confirm
    # Optional config sanity check.
    cruncher config summary -c "$CONFIG"
    cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --tf rcdA --tf lrp --tf acrR --tf soxR --update -c "$CONFIG"
    cruncher fetch sites --source regulondb --tf lexA --tf cpxR --tf baeR --tf rcdA --tf lrp --tf fur --tf fnr --tf acrR --tf soxR --tf soxS --update -c "$CONFIG"
    cruncher fetch sites --source baer_chip_exo --tf baeR --update -c "$CONFIG"
    # Discover motifs from merged site evidence. This step merges all fetched site sets across sources and runs MEME OOPS into the workspace-specific source id.
    cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id project_merged_meme_oops_all_tfs -c "$CONFIG"
    # Freeze resolved motif/site provenance for this set.
    # If you change catalog.source_preference or discovery --source-id, re-run cruncher lock -c "$CONFIG" before parse.
    cruncher lock -c "$CONFIG"
    cruncher parse --force-overwrite -c "$CONFIG"
    cruncher sample --force-overwrite -c "$CONFIG"
    cruncher analyze --summary -c "$CONFIG"
    cruncher export sequences --latest -c "$CONFIG"
    cruncher catalog logos --source project_merged_meme_oops_all_tfs --set 1 -c "$CONFIG"
    cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite
    cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite

### Optional output checks

    find outputs/plots -maxdepth 1 -type f | sort
    ls -1 outputs/analysis/reports outputs/export
