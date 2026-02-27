## Project Workspace (all TFs)

**Last updated by:** cruncher-maintainers on 2026-02-23


### Contents
- [Intent](#intent)
- [Source flow](#source-flow)
- [End-to-end commands](#end-to-end-commands)
- [Verify outputs](#verify-outputs)
- [Provenance check for lexA/cpxR consistency](#provenance-check-for-lexacpxr-consistency)
- [Current workspace defaults (operational)](#current-workspace-defaults-operational)

This is an operational runbook for the all-TFs workspace. It is intentionally command-first, with only the context needed to run and audit the pipeline.

Workspace path:
- `src/dnadesign/cruncher/workspaces/project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs/`

Run this single command to do everything in this demo:

```bash
# Execute the Cruncher machine runbook for this workspace.
uv run cruncher workspaces run --runbook configs/runbook.yaml
```

Or run the same flow step by step with context below.

Joint regulator set:
- `lexA`, `cpxR`, `baeR`, `rcdA`, `lrp`, `fur`, `fnr`, `acrR`, `soxR`, `soxS`

### Intent

Run one strict end-to-end lifecycle:
1. fetch site sets from configured sources,
2. discover merged MEME OOPS motifs into a pinned source,
3. lock/parse/sample/analyze against that discovered source,
4. generate exports and discovered-source logos for audit artifacts,
5. run standard study sweeps for portfolio aggregation.

### Source flow

- `demo_local_meme`: local MEME site sets for `lexA`, `cpxR`, `rcdA`, `lrp`, `acrR`, `soxR`.
- `regulondb`: curated TFBS site sets for all project TFs.
- `baer_chip_exo`: BaeR ChIP-exo site set.
- discovery output source: `project_merged_meme_oops_all_tfs`.

Input parity note:
- `inputs/local_motifs/lexA.txt` and `inputs/local_motifs/cpxR.txt` are shared with `demo_pairwise` and `demo_multitf`.
- If those files drift, `lexA/cpxR` logos will drift even with the same fetch/discover command sequence.

### End-to-end commands

```bash
# Fail fast on command errors, unset vars, and pipe failures.
set -euo pipefail
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/cruncher/workspaces/project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs
# Pin config path for repeated CLI calls.
CONFIG="$PWD/configs/config.yaml"
cruncher() { uv run cruncher "$@"; }

# Optional reset of generated artifacts/state for a clean rerun.
cruncher workspaces reset --root . --confirm

# 0) Quick config sanity check.
cruncher config summary -c "$CONFIG"

# 1) Fetch site sets from all configured sources.
cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --tf rcdA --tf lrp --tf acrR --tf soxR --update -c "$CONFIG"
# Fetch TF binding sites from the configured source.
cruncher fetch sites --source regulondb      --tf lexA --tf cpxR --tf baeR --tf rcdA --tf lrp --tf fur --tf fnr --tf acrR --tf soxR --tf soxS --update -c "$CONFIG"
# Fetch TF binding sites from the configured source.
cruncher fetch sites --source baer_chip_exo  --tf baeR --update -c "$CONFIG"

# 2) Discover merged motifs for the project source (MEME OOPS).
cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id project_merged_meme_oops_all_tfs -c "$CONFIG"

# 3) Freeze motif refs and hashes, then stage parsed inputs.
# If you change `catalog.source_preference` or discovery `--source-id`, re-run `cruncher lock -c "$CONFIG"` before parse.
cruncher lock -c "$CONFIG"
# Parse inputs into normalized Cruncher artifacts.
cruncher parse --force-overwrite -c "$CONFIG"

# 4) Optimize and analyze.
cruncher sample --force-overwrite -c "$CONFIG"
# Compute analysis summaries for generated sequence sets.
cruncher analyze --summary -c "$CONFIG"

# 5) Export sequence tables.
cruncher export sequences --latest -c "$CONFIG"

# 6) Render discovered-source logos.
cruncher catalog logos --source project_merged_meme_oops_all_tfs --set 1 -c "$CONFIG"

# 7) Run standard workspace studies used by portfolio aggregation.
#    - length_vs_score: sequence_length sweep with step-2 spacing + base config anchor
#    - diversity_vs_score: diversity sweep from 0.00 -> 1.00
cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite
# Run the configured Cruncher parameter-sweep study.
cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite
```

### Verify outputs

```bash
# Core analysis plots.
find outputs/plots -maxdepth 1 -type f | sort

# Logo outputs and manifest.
find outputs/plots -maxdepth 1 -type f -name '*_logo.png' | sort

# Run report + export manifest.
ls -1 outputs/analysis/reports outputs/export

# Study outputs (length sweep + diversity sweep).
find outputs/studies -maxdepth 3 -type f -path '*/tables/*' | sort
find outputs/plots -maxdepth 1 -type f -name 'study__*__plot__*.pdf' | sort
```

### Provenance check for lexA/cpxR consistency

If you want project `lexA/cpxR` logos to match the pairwise demo exactly, the local motif inputs for those TFs must match before running discovery.

```bash
# Compare local motif file checksums with pairwise workspace.
shasum inputs/local_motifs/lexA.txt ../demo_pairwise/inputs/local_motifs/lexA.txt
# Compare checksums to confirm motif inputs are byte-identical.
shasum inputs/local_motifs/cpxR.txt ../demo_pairwise/inputs/local_motifs/cpxR.txt
```

### Current workspace defaults (operational)

- `sample.sequence_length: 15`
- `sample.budget.tune: 50000`
- `sample.budget.draws: 300000`
- `optimizer.chains: 8`
- `analysis.pairwise: all_pairs_grid` (render all TF-pair score-space panels in one figure)
