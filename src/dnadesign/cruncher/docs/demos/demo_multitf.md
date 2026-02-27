## Multi-TF Demo (three TF, end-to-end)

**Last updated by:** cruncher-maintainers on 2026-02-23

### Contents
- [Overview](#overview)
- [What This Demo Teaches](#what-this-demo-teaches)
- [Environment and CLI setup](#environment-and-cli-setup)
- [How this differs from pairwise](#how-this-differs-from-pairwise)
- [Source map (where data comes from)](#source-map-where-data-comes-from)
- [Config gist (annotated)](#config-gist-annotated)
- [Step-by-step flow](#step-by-step-flow)
- [Workspace studies](#workspace-studies)
- [Inspect outputs](#inspect-outputs)
- [Related docs](#related-docs)

### Overview

This demo extends the pairwise flow from two regulators to three:
`lexA`, `cpxR`, and `baeR`.

Lifecycle is unchanged (fetch -> discover -> lock -> parse -> sample -> analyze),
but this workspace adds a third-TF site source (`baer_chip_exo`) and renders
score-space as an all-pairs grid (`analysis.pairwise: all_pairs_grid`).

### What This Demo Teaches

- How to add a third TF without changing optimizer contracts.
- How mixed source provenance is staged before discovery.
- How multi-TF analysis plots differ semantically from strict two-axis pairwise plots.

### Environment and CLI setup

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/cruncher/workspaces/demo_multitf
# Pin config path for repeated CLI calls.
CONFIG="$PWD/configs/config.yaml"
cruncher() { uv run cruncher "$@"; }
```

Run this single command to do everything in this demo:

```bash
# Execute the Cruncher machine runbook for this workspace.
uv run cruncher workspaces run --runbook configs/runbook.yaml
```

Or run the same flow step by step with context below.
Each step corresponds to the same runbook flow, but split for inspection and debugging.

MEME/FIMO tooling checks:

```bash
# Validate MEME tool availability through Cruncher.
cruncher discover check -c "$CONFIG"

# Optional explicit PATH wiring if needed.
export MEME_BIN="$(cd ../../../../../.pixi/envs/default/bin && pwd)"
# Export environment variables consumed by later commands.
export PATH="$MEME_BIN:$PATH"
# Confirm MEME/FIMO executables are available on PATH.
command -v meme streme fimo
```

Notes:
- This workspace config sets `discover.tool_path` to the pixi MEME bin.
- `analysis.fimo_compare` is disabled here by default; enable it if you want `optimizer_vs_fimo` for this demo.

### How this differs from pairwise

- **Regulators:** three TFs (`lexA`, `cpxR`, `baeR`) instead of two.
- **Input sources:** keeps `demo_local_meme` + `regulondb` for `lexA/cpxR`, adds `baer_chip_exo` + `regulondb` for `baeR`.
- **Optimization surface:** same tuned sampler/optimizer settings as pairwise (8 chains, same budgets, same move/cooling settings).
- **Analysis projection:** `pairwise: all_pairs_grid` renders every TF pair panel in one figure.
- **Plot outputs:** same core plot files are produced; score-space is now a multi-panel TF-pair grid.

### Source map (where data comes from)

- `demo_local_meme`:
  local MEME files representing DAP-seq-derived data for `lexA`/`cpxR`.
- `regulondb`:
  curated RegulonDB site sets for all three TFs.
- `baer_chip_exo`:
  local FASTA source of BaeR ChIP-exo sites used to strengthen `baeR` discovery input.

Consistency contract:
- `lexA/cpxR` local MEME files in this workspace are byte-identical to `demo_pairwise` and the project workspace.
- This keeps `lexA/cpxR` discovery inputs aligned; differences in resulting logos then reflect run content, not source drift.

Optional source introspection:

```bash
# List configured source definitions for this workspace.
cruncher sources list -c "$CONFIG"
# Inspect cached source summary for this source id.
cruncher sources summary --source demo_local_meme --scope cache -c "$CONFIG"
# Inspect cached source summary for this source id.
cruncher sources summary --source regulondb --scope cache -c "$CONFIG"
# Inspect cached source summary for this source id.
cruncher sources summary --source baer_chip_exo --scope cache -c "$CONFIG"
```

### Config gist (annotated)

This excerpt captures the key intent from `workspaces/demo_multitf/configs/config.yaml`:

```yaml
cruncher:
  workspace:
    out_dir: outputs                          # Sets `out_dir` for this example configuration.
    regulator_sets:
      - [lexA, cpxR, baeR]          # optimize all three TFs together

  catalog:
    root: .cruncher/demo_multitf              # Root directory used by this run/workspace.
    source_preference: [demo_merged_meme_oops_multitf]  # lock must resolve only discovered motifs
    pwm_source: matrix                        # Sets `pwm_source` for this example configuration.
    combine_sites: true                       # Boolean switch for `combine_sites`.

  discover:
    tool: meme                                # Sets `tool` for this example configuration.
    meme_mod: oops                            # Sets `meme_mod` for this example configuration.
    source_id: demo_merged_meme_oops_multitf  # Sets `source_id` for this example configuration.

  sample:
    sequence_length: 16             # same fixed length as pairwise demo
    budget:
      tune: 25000                             # Sets `tune` for this example configuration.
      draws: 150000                           # Sets `draws` for this example configuration.
    optimizer:
      kind: gibbs_anneal                      # Sets `kind` for this example configuration.
      chains: 8                     # same tuned optimizer defaults as pairwise
    objective:
      combine: min                            # Sets `combine` for this example configuration.
      score_scale: normalized-llr             # Sets `score_scale` for this example configuration.
      bidirectional: true                     # Boolean switch for `bidirectional`.

  analysis:
    pairwise: all_pairs_grid        # render every TF pair panel in one score-space figure
    trajectory_sweep_mode: best_so_far        # Sets `trajectory_sweep_mode` for this example configuration.
    fimo_compare:
      enabled: false                # enable true if you want optimizer_vs_fimo here
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

#### 2) Fetch site sets

Intent:
- make every source contribution explicit,
- include both curated and assay-specific data before discovery.

```bash
# Fetch TF binding sites from the configured source.
uv run cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
# Fetch TF binding sites from the configured source.
uv run cruncher fetch sites --source regulondb      --tf lexA --tf cpxR --tf baeR --update -c "$CONFIG"
# Fetch TF binding sites from the configured source.
uv run cruncher fetch sites --source baer_chip_exo --tf baeR --update -c "$CONFIG"
```

#### 3) Discover motifs from merged pools

```bash
# Run motif discovery over fetched site evidence.
uv run cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id demo_merged_meme_oops_multitf -c "$CONFIG"
```

#### 4) Lock and parse

If you change `catalog.source_preference` or discovery `--source-id`, re-run `cruncher lock -c "$CONFIG"` before parse.

```bash
# Freeze motif/source provenance for deterministic downstream steps.
uv run cruncher lock -c "$CONFIG"
# Parse inputs into normalized Cruncher artifacts.
uv run cruncher parse --force-overwrite -c "$CONFIG"
```

#### 5) Sample and analyze

```bash
# Generate candidate sequences from parsed motif artifacts.
uv run cruncher sample --force-overwrite -c "$CONFIG"
# Compute analysis summaries for generated sequence sets.
uv run cruncher analyze --summary -c "$CONFIG"
```

#### 6) Export sequence-facing tables

```bash
# Export latest elite sequences for downstream use.
uv run cruncher export sequences --latest -c "$CONFIG"
```

#### 7) Generate logos (optional)

```bash
# Render motif logos for visual QC of discovered motifs.
uv run cruncher catalog logos --source demo_merged_meme_oops_multitf --set 1 -c "$CONFIG"
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
- `outputs/export/table__elites.csv`
- `outputs/export/export_manifest.json`
- `outputs/plots/` (`*_logo.png`)
- `outputs/studies/length_vs_score/<study_id>/tables/table__*.parquet`
- `outputs/studies/diversity_vs_score/<study_id>/tables/table__*.parquet`
- `outputs/plots/study__length_vs_score__<study_id>__plot__sequence_length_tradeoff.pdf`
- `outputs/plots/study__diversity_vs_score__<study_id>__plot__mmr_diversity_tradeoff.pdf`

### Related docs

- [Pairwise demo](demo_pairwise.md)
- [Project workspace](project_all_tfs.md)
- [MEME Suite setup](../guides/meme_suite.md)
- [Ingestion + sources](../guides/ingestion.md)
- [Study: length vs score](../guides/study_length_vs_score.md)
- [Study: diversity vs score](../guides/study_diversity_vs_score.md)
- [Config reference](../reference/config.md)
- [CLI reference](../reference/cli.md)
