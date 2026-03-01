## Pairwise Demo (end-to-end)

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


**Last updated by:** cruncher-maintainers on 2026-02-23

### Contents
- [Overview](#overview)
- [What This Demo Teaches](#what-this-demo-teaches)
- [Environment and CLI setup](#environment-and-cli-setup)
- [Source map (where data comes from)](#source-map-where-data-comes-from)
- [Config gist (annotated)](#config-gist-annotated)
- [Step-by-step flow](#step-by-step-flow)
- [Workspace studies](#workspace-studies)
- [Inspect outputs](#inspect-outputs)
- [Related docs](#related-docs)

### Overview

This demo is the default two-regulator Cruncher flow for `lexA` and `cpxR`.

You run one strict lifecycle in order:
1. fetch site data from explicit sources,
2. discover MEME OOPS motifs,
3. lock exact motif artifacts,
4. parse locked motifs,
5. sample optimized sequences,
6. analyze and export outputs.

### What This Demo Teaches

- How source provenance is made explicit before optimization.
- Why `lock` is a hard reproducibility boundary.
- How discovered motifs (`demo_merged_meme_oops`) become the only allowed motifs for sample/analyze.
- How to read core analysis plots, including `optimizer_vs_fimo`.

### Environment and CLI setup

Run from the workspace and define a shell helper:

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/cruncher/workspaces/demo_pairwise
# Pin config path for repeated CLI calls.
CONFIG="$PWD/configs/config.yaml"
cruncher() { uv run cruncher "$@"; }
```

Run this single command to do everything in this demo:

```bash
# Execute the Cruncher machine runbook for this workspace.
uv run cruncher workspaces run --runbook configs/runbook.yaml
```

Stop here if you only need the happy path.

This full runbook also executes the two workspace study sweeps at the end, so runtime is longer than the base sample/analyze flow.

Quick smoke path (base lifecycle only, no study sweeps):

```bash
# Execute the Cruncher machine runbook for this workspace.
uv run cruncher workspaces run --runbook configs/runbook.yaml --step reset_workspace --step config_summary --step fetch_sites_demo_local_meme --step fetch_sites_regulondb --step discover_motifs --step lock_targets --step parse_run --step sample_run --step analyze_summary --step export_sequences_latest --step render_logos
```

Or run the same flow step by step with context below. Each step corresponds to the same runbook flow, but split for inspection and debugging.

Check MEME Suite tools used by discovery and optional FIMO compare:

```bash
# Validate tool availability as seen by Cruncher.
cruncher discover check -c "$CONFIG"

# Optional explicit path setup if tools are not found on PATH.
export MEME_BIN="$(cd ../../../../../.pixi/envs/default/bin && pwd)"
# Export environment variables consumed by later commands.
export PATH="$MEME_BIN:$PATH"
# Confirm MEME/FIMO executables are available on PATH.
command -v meme streme fimo
```

Notes:
- This workspace config already sets `discover.tool_path` to the repo pixi env.
- `fimo` is required for `analysis.fimo_compare.enabled: true` in this demo.

### Source map (where data comes from)

This demo intentionally merges two site sources per TF before discovery:

- `demo_local_meme`:
  local MEME files bundled with the demo; represents DAP-seq-derived motif/site data.
- `regulondb`:
  curated RegulonDB TF-binding-site records fetched through the RegulonDB GraphQL API.

Why both:
- local DAP-seq gives broad/consistent discovery input,
- RegulonDB contributes curated in vivo context,
- merged site pools produce the discovery source used for lock/sample/analyze.

Consistency contract:
- `inputs/local_motifs/lexA.txt` and `inputs/local_motifs/cpxR.txt` are intentionally shared with
  `demo_multitf` and `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs`.
- If these files diverge, discovered MEME-OOPS motifs and logos for `lexA/cpxR` will diverge too.

Optional source introspection:

```bash
# List configured source definitions for this workspace.
cruncher sources list -c "$CONFIG"
# Inspect cached source summary for this source id.
cruncher sources summary --source demo_local_meme --scope cache -c "$CONFIG"
# Inspect cached source summary for this source id.
cruncher sources summary --source regulondb --scope cache -c "$CONFIG"
```

### Config gist (annotated)

This excerpt captures the key intent from `workspaces/demo_pairwise/configs/config.yaml`:

```yaml
cruncher:
  schema_version: 3                            # strict config schema contract
  workspace:
    out_dir: outputs                           # write run artifacts under workspace outputs/
    regulator_sets:
      - [lexA, cpxR]                           # optimize this regulator set jointly

  catalog:
    root: .cruncher/demo_pairwise              # workspace-local cache, lock, and parsed artifacts
    source_preference:
      - demo_merged_meme_oops                  # lock resolves motifs only from this discovered source
    pwm_source: matrix                         # sample from motif matrices, not raw site windows
    combine_sites: true                        # merge cached sites per TF before motif discovery

  discover:
    enabled: true                              # enable discovery stage in the runbook flow
    tool: meme                                 # use MEME for motif discovery
    meme_mod: oops                             # one occurrence per sequence during motif discovery
    source_id: demo_merged_meme_oops           # write discovered motifs under this source id

  sample:
    sequence_length: 18                        # design fixed-length 18 bp sequences
    budget:
      tune: 50000                              # warmup sweeps
      draws: 300000                            # optimization sweeps after warmup
    objective:
      combine: min                             # optimize the weakest TF score in the set
      score_scale: normalized-llr              # use normalized LLR scaling for comparability
      bidirectional: true                      # evaluate best motif hit across both strands
    optimizer:
      kind: gibbs_anneal                       # run hybrid Gibbs annealing optimization
      chains: 8                                # use 8 independent optimizer chains
    elites:
      k: 8                                     # persist the top 8 elite sequences
      select:
        diversity: 0.05                        # apply low-but-nonzero MMR diversity pressure
        pool_size: 32000                       # evaluate this many candidates before MMR selection

  analysis:
    pairwise: [lexA, cpxR]                     # render a two-axis score-space for this TF pair
    trajectory_sweep_mode: best_so_far         # show monotonic best-so-far chain trajectories
    fimo_compare:
      enabled: true                            # emit optimizer_vs_fimo plot (requires fimo)
```

### Step-by-step flow

#### 0) Reset generated workspace state

```bash
# Reset workspace outputs before a fresh Cruncher run.
uv run cruncher workspaces reset --root . --confirm
```

#### 1) Check resolved config

```bash
# Print resolved Cruncher config to confirm active knobs.
uv run cruncher config summary -c "$CONFIG"
```

#### 2) Fetch source sites

Intent:
- cache all source inputs you want discovery to see,
- avoid accidental hidden source mixing.

```bash
# Fetch TF binding sites from the configured source.
uv run cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
# Fetch TF binding sites from the configured source.
uv run cruncher fetch sites --source regulondb      --tf lexA --tf cpxR --update -c "$CONFIG"
```

#### 3) Discover motifs (MEME OOPS)

Intent:
- build one discovered motif source from merged site sets,
- pin that source in downstream lock/sample/analyze.

```bash
# Run motif discovery over fetched site evidence.
uv run cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id demo_merged_meme_oops -c "$CONFIG"
```

#### 4) Lock and parse

Intent:
- `lock`: freeze exact motif IDs + hashes.
- `parse`: validate and stage locked motifs for sampling.
- If you change `catalog.source_preference` or discovery `--source-id`, re-run `cruncher lock -c "$CONFIG"` before parse.

```bash
# Freeze motif/source provenance for deterministic downstream steps.
uv run cruncher lock -c "$CONFIG"
# Parse inputs into normalized Cruncher artifacts.
uv run cruncher parse --force-overwrite -c "$CONFIG"
```

#### 5) Sample and analyze

Intent:
- `sample`: run tuned 8-chain Gibbs annealing.
- `analyze`: generate reports/tables/plots from artifacts only.

```bash
# Generate candidate sequences from parsed motif artifacts.
uv run cruncher sample --force-overwrite -c "$CONFIG"
# Compute analysis summaries for generated sequence sets.
uv run cruncher analyze --summary -c "$CONFIG"

# Optional: enable trajectory video in analysis config to emit plots/chain_trajectory_video.mp4.
# Default best-of chain timeline is monotonic (timeline_mode: best_so_far).
```

#### 6) Export sequence-facing tables

```bash
# Export latest elite sequences for downstream use.
uv run cruncher export sequences --latest -c "$CONFIG"
```

#### 7) Generate motif logos

```bash
# Render motif logos for visual QC of discovered motifs.
uv run cruncher catalog logos --source demo_merged_meme_oops --set 1 -c "$CONFIG"
```

### Workspace studies

These workspace-local studies are the standard sweep layer used by portfolio aggregation:

- `length_vs_score`: sweeps `sample.sequence_length` with step-2 spacing and always includes the base `sample.sequence_length` from `config.yaml` as an anchor trial.
- `diversity_vs_score`: sweeps diversity from `0.00` to `1.00` at fixed workspace sequence length and writes diversity tradeoff tables/plots.

Run them:

```bash
# Run the configured Cruncher parameter-sweep study.
uv run cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite
# Run the configured Cruncher parameter-sweep study.
uv run cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite
```

Inspect study outputs:

```bash
# Inspect study outputs and resolved sweep metadata.
uv run cruncher study show --run outputs/studies/length_vs_score/<study_id>
# Inspect study outputs and resolved sweep metadata.
uv run cruncher study show --run outputs/studies/diversity_vs_score/<study_id>
```

### Inspect outputs

Primary files after a successful run:

- `outputs/analysis/reports/summary.json`
- `outputs/analysis/reports/report.md`
- `outputs/plots/elite_score_space_context.pdf`
- `outputs/plots/chain_trajectory_sweep.pdf`
- `outputs/plots/elites_nn_distance.pdf`
- `outputs/plots/elites_showcase.pdf`
- `outputs/plots/optimizer_vs_fimo.pdf`
- `outputs/export/table__elites.csv`
- `outputs/export/export_manifest.json`
- `outputs/plots/` (`*_logo.png`)
- `outputs/studies/length_vs_score/<study_id>/tables/table__*.parquet`
- `outputs/studies/diversity_vs_score/<study_id>/tables/table__*.parquet`
- `outputs/plots/study__length_vs_score__<study_id>__plot__sequence_length_tradeoff.pdf`
- `outputs/plots/study__diversity_vs_score__<study_id>__plot__mmr_diversity_tradeoff.pdf`

### Related docs

- [Multi-TF demo](demo_multitf.md)
- [Project workspace](project_all_tfs.md)
- [MEME Suite setup](../guides/meme_suite.md)
- [Ingestion + sources](../guides/ingestion.md)
- [Study: length vs score](../guides/study_length_vs_score.md)
- [Study: diversity vs score](../guides/study_diversity_vs_score.md)
- [Config reference](../reference/config.md)
- [CLI reference](../reference/cli.md)
