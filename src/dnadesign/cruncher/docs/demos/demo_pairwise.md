## Pairwise Demo (end-to-end)

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

This is the canonical two-regulator Cruncher walkthrough for `lexA` + `cpxR`.

You will run the full lifecycle in order:
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
cd src/dnadesign/cruncher/workspaces/demo_pairwise
CONFIG="$PWD/configs/config.yaml"
cruncher() { uv run cruncher "$@"; }
```

Run this single command to do everything in this demo:

```bash
uv run cruncher workspaces run --runbook configs/runbook.yaml
```

This full runbook also executes two study sweeps at the end, so runtime is longer than the base sample/analyze flow.

Quick smoke path (base lifecycle only):

```bash
uv run cruncher workspaces run --runbook configs/runbook.yaml --step reset_workspace --step config_summary --step fetch_sites_demo_local_meme --step fetch_sites_regulondb --step discover_motifs --step lock_targets --step parse_run --step sample_run --step analyze_summary --step export_sequences_latest --step render_logos
```

Or run the same flow step by step with context below. Each step corresponds to the same runbook flow, but split for inspection and debugging.

Check MEME Suite tools used by discovery and optional FIMO compare:

```bash
# Validate tool availability as seen by Cruncher.
cruncher discover check -c "$CONFIG"

# Optional explicit path setup if tools are not found on PATH.
export MEME_BIN="$(cd ../../../../../.pixi/envs/default/bin && pwd)"
export PATH="$MEME_BIN:$PATH"
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
cruncher sources list -c "$CONFIG"
cruncher sources summary --source demo_local_meme --scope cache -c "$CONFIG"
cruncher sources summary --source regulondb --scope cache -c "$CONFIG"
```

### Config gist (annotated)

This excerpt captures the key intent from `workspaces/demo_pairwise/configs/config.yaml`:

```yaml
cruncher:
  workspace:
    out_dir: outputs                # all run artifacts are written here
    regulator_sets:
      - [lexA, cpxR]                # optimize both TFs jointly in one set

  catalog:
    root: .cruncher/demo_pairwise   # workspace-local cache/state root
    source_preference: [demo_merged_meme_oops]  # lock must resolve motifs from this discovered source
    pwm_source: matrix              # sample from motif matrices, not raw site windows
    combine_sites: true             # merge all cached sites per TF before discovery

  discover:
    enabled: true
    tool: meme
    meme_mod: oops                  # one occurrence per input sequence during motif discovery
    source_id: demo_merged_meme_oops

  sample:
    sequence_length: 16             # fixed-length optimization contract
    budget:
      tune: 25000                   # adaptation / warmup sweeps
      draws: 150000                 # optimization sweeps after tune
    objective:
      combine: min                  # optimize weakest TF first (joint fairness)
      score_scale: normalized-llr
      bidirectional: true
    optimizer:
      kind: gibbs_anneal
      chains: 8                     # tuned default chain count
    elites:
      k: 8
      select:
        diversity: 0.05             # low but non-zero diversity pressure
        pool_size: 32000

  analysis:
    pairwise: [lexA, cpxR]          # fixed two-axis score-space projection
    trajectory_sweep_mode: best_so_far
    fimo_compare:
      enabled: true                 # emit optimizer_vs_fimo plot (requires fimo)
```

### Step-by-step flow

#### 0) Reset generated workspace state

```bash
uv run cruncher workspaces reset --root . --confirm
```

#### 1) Check resolved config

```bash
uv run cruncher config summary -c "$CONFIG"
```

#### 2) Fetch source sites

Intent:
- cache all source inputs you want discovery to see,
- avoid accidental hidden source mixing.

```bash
uv run cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
uv run cruncher fetch sites --source regulondb      --tf lexA --tf cpxR --update -c "$CONFIG"
```

#### 3) Discover motifs (MEME OOPS)

Intent:
- build one discovered motif source from merged site sets,
- pin that source in downstream lock/sample/analyze.

```bash
uv run cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id demo_merged_meme_oops -c "$CONFIG"
```

#### 4) Lock and parse

Intent:
- `lock`: freeze exact motif IDs + hashes.
- `parse`: validate and stage locked motifs for sampling.
- If you change `catalog.source_preference` or discovery `--source-id`, re-run `cruncher lock -c "$CONFIG"` before parse.

```bash
uv run cruncher lock -c "$CONFIG"
uv run cruncher parse --force-overwrite -c "$CONFIG"
```

#### 5) Sample and analyze

Intent:
- `sample`: run tuned 8-chain Gibbs annealing.
- `analyze`: generate reports/tables/plots from artifacts only.

```bash
uv run cruncher sample --force-overwrite -c "$CONFIG"
uv run cruncher analyze --summary -c "$CONFIG"
```

#### 6) Export sequence-facing tables

```bash
uv run cruncher export sequences --latest -c "$CONFIG"
```

#### 7) Generate motif logos

```bash
uv run cruncher catalog logos --source demo_merged_meme_oops --set 1 -c "$CONFIG"
```

### Workspace studies

These workspace-local studies are the canonical sweep layer used by portfolio aggregation:

- `length_vs_score`: sweeps `sample.sequence_length` with step-2 spacing and always includes the base `sample.sequence_length` from `config.yaml` as an anchor trial.
- `diversity_vs_score`: sweeps diversity from `0.00` to `1.00` at fixed workspace sequence length and writes diversity tradeoff tables/plots.

Run them:

```bash
uv run cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite
uv run cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite
```

Inspect study outputs:

```bash
uv run cruncher study show --run outputs/studies/length_vs_score/<study_id>
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
