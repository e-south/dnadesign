# Cruncher config

## Contents
- [Overview](#overview)
- [Root](#root)
- [Workspace](#workspace)
- [io](#io)
- [catalog](#catalog)
- [discover](#discover)
- [ingest](#ingest)
- [sample](#sample)
- [analysis](#analysis)
- [campaigns](#campaigns)
- [Inspect resolved config](#inspect-resolved-config)
- [Related docs](#related-docs)

## Overview

Cruncher uses a single root key and strict validation. Unknown keys and missing required keys are errors.

- Root key: `cruncher`
- Schema version: `schema_version: 3`
- Fixed-length invariant: `sample.sequence_length >= max_pwm_width`

Use this doc as a *schema map*. If you only change a few knobs, start with:
`workspace.regulator_sets`, `catalog.pwm_source`, `sample.sequence_length`,
`sample.budget.*`, and `sample.elites.*`.

## Root

```yaml
cruncher:
  schema_version: 3
  workspace: { ... }
  io: { ... }
  catalog: { ... }
  discover: { ... }
  ingest: { ... }
  sample: { ... }
  analysis: { ... }
  campaigns: []
  campaign: null
```

Notes:
- `campaigns` are optional helpers for expanding regulator sets.
- `campaign` metadata is optional runtime metadata (for generated or campaign-driven runs).

## Workspace

```yaml
workspace:
  out_dir: outputs/
  regulator_sets:
    - [lexA, cpxR]
  regulator_categories: {}
```

Notes:
- `out_dir` is resolved relative to the config file and must be a relative path.
- `regulator_sets` defines direct run targets (each set becomes a run).
- `regulator_sets` may be empty when you run commands with `--campaign <name>`.
- `regulator_categories` are used by campaigns.

## io

```yaml
io:
  parsers:
    extra_modules: []
```

Notes:
- `extra_modules` is a list of importable Python modules that register parsers via
  `@register("FMT")` decorators in `dnadesign.cruncher.io.parsers.backend`.

## catalog

Controls how the local catalog is queried and how PWMs are prepared.

```yaml
catalog:
  root: .cruncher
  source_preference: [regulondb]
  allow_ambiguous: false
  pwm_source: matrix              # matrix | sites
  site_kinds: null                # optional filter: [curated], [ht_tfbinding], [ht_peak]
  combine_sites: false
  pseudocounts: 0.5               # smoothing for site-derived PWMs
  dataset_preference: []          # HT dataset ranking
  dataset_map: {}                 # TF -> dataset ID
  site_window_lengths: {}         # TF or dataset:<id> -> length (bp)
  site_window_center: midpoint    # midpoint | summit
  pwm_window_lengths: {}          # TF or dataset:<id> -> PWM window length (bp)
  pwm_window_strategy: max_info   # max_info
  min_sites_for_pwm: 2
  allow_low_sites: false
```

Notes:
- `pwm_source=matrix` uses cached motif matrices (default).
- `pwm_source=sites` builds PWMs from cached binding-site sequences at runtime.
- If site lengths vary, set `site_window_lengths` per TF or dataset.
- To constrain long PWMs, set `pwm_window_lengths` and use `pwm_window_strategy=max_info`.

## discover

Motif discovery/alignment via MEME Suite.

```yaml
discover:
  enabled: false
  tool: auto                     # auto | streme | meme
  tool_path: null                # optional path to meme/streme or a bin dir
  window_sites: false            # pre-window binding sites before discovery
  minw: null                     # minimum motif width (auto from site lengths if unset)
  maxw: null                     # maximum motif width (auto from site lengths if unset)
  nmotifs: 1                     # motifs per TF
  meme_mod: null                 # oops | zoops | anr
  meme_prior: null               # dirichlet | dmix | mega | megap | addone
  min_sequences_for_streme: 50   # auto threshold
  source_id: meme_suite
  replace_existing: true
```

Notes:
- Discovery requires cached binding sites and MEME Suite binaries.
- `tool=auto` chooses STREME above `min_sequences_for_streme`, MEME otherwise.

## ingest

Controls source adapters, genome hydration, and HTTP retries.

```yaml
ingest:
  genome_source: ncbi       # ncbi | fasta | none
  genome_fasta: null        # local FASTA (optional)
  genome_cache: .cruncher/genomes
  genome_assembly: null
  contig_aliases: {}
  ncbi_email: null
  ncbi_tool: cruncher
  ncbi_api_key: null
  ncbi_timeout_seconds: 30
  http:
    retries: 3
    backoff_seconds: 0.5
    max_backoff_seconds: 8.0
    retry_statuses: [429, 500, 502, 503, 504]
    respect_retry_after: true
  regulondb:
    base_url: https://regulondb.ccg.unam.mx/graphql
    verify_ssl: true
    ca_bundle: null
    timeout_seconds: 30
    motif_matrix_source: alignment   # alignment | sites
    alignment_matrix_semantics: probabilities  # probabilities | counts
    min_sites_for_pwm: 2
    pseudocounts: 0.5
    allow_low_sites: false
    curated_sites: true
    ht_sites: false
    ht_dataset_sources: null
    ht_dataset_type: TFBINDING
    ht_binding_mode: tfbinding       # tfbinding | peaks
    uppercase_binding_site_only: true
  local_sources: []
  site_sources: []
```

## sample

Fixed-length sampling with PT-only optimization and MMR elite selection.

```yaml
sample:
  seed: 42
  sequence_length: 30

  budget:
    tune: 1000
    draws: 2000

  objective:
    bidirectional: true
    score_scale: normalized-llr   # normalized-llr | llr | logp | z | consensus-neglop-sum
    combine: min                  # min | sum
    softmin:
      enabled: true
      schedule: linear            # fixed | linear
      beta_start: 0.5
      beta_end: 10.0
    scoring:
      pwm_pseudocounts: 0.10
      log_odds_clip: null

  moves:
    profile: balanced             # enum only
    overrides:
      # Optional per-move/operator knobs (advanced).
      # These tune proposal shapes and relative mix; leave unset to use the profile defaults.
      block_len_range: null       # [min, max] bp (B move)
      multi_k_range: null         # [min, max] positions (M move)
      slide_max_shift: null       # max shift in bp (L move, if enabled by profile)
      swap_len_range: null        # [min, max] bp (W move, if enabled by profile)
      move_probs: null            # map of move -> probability (e.g., {S: 0.4, B: 0.3, M: 0.3, I: 0.0})
      move_schedule: null         # optional schedule policy (when supported)
      target_worst_tf_prob: null  # bias proposals toward the worst TF (when supported)
      target_window_pad: null     # pad around target window for proposals (when supported)
      insertion_consensus_prob: null  # consensus bias for insertions (when supported)
      adaptive_weights:
        enabled: false
        window: 250
        k: 0.5
        min_prob: 0.01
        max_prob: 0.95
        kinds: [S, B, M, I]
        targets: {S: 0.95, B: 0.40, M: 0.35, I: 0.35}
      proposal_adapt:
        enabled: false
        window: 250
        step: 0.10
        min_scale: 0.50
        max_scale: 2.00
        target_low: 0.25
        target_high: 0.75

  pt:
    n_temps: 6
    temp_max: 20.0
    swap_stride: 1
    adapt:
      enabled: true
      target_swap: 0.25
      window: 50
      k: 0.5
      min_scale: 0.25
      max_scale: 4.0
      strict: false
      saturation_windows: 5
      stop_after_tune: true

  elites:
    k: 10
    filter:
      min_per_tf_norm: auto       # auto | float | null
      require_all_tfs: true
      pwm_sum_min: 0.0
    select:
      policy: mmr
      alpha: 0.85
      pool_size: auto
      diversity_metric: tfbs_core_weighted_hamming

  output:
    save_sequences: true
    save_trace: true
    include_tune_in_sequences: false
    live_metrics: true
```

Notes:
- `sequence_length` must be at least the widest PWM (after any `pwm_window_lengths`).
- Canonicalization is automatic when `objective.bidirectional=true`.
- MMR distance is TFBS-core weighted Hamming (tolerant weighting, low-information positions emphasized).
- `moves.overrides.*` contains optional expert controls (operator mix + adaptation). Leave unset unless you are actively tuning proposals.
- Set `pt.adapt.strict=true` to fail the run when PT ladder adaptation saturates at `max_scale` for too many windows.

## analysis

Curated plot + table suite with data-driven skipping only (no plot booleans).

```yaml
analysis:
  enabled: true
  run_selector: latest      # latest | explicit
  runs: []                  # used only if run_selector=explicit
  pairwise: auto            # off | auto | [tf1, tf2]
  plot_format: png          # png | pdf | svg
  plot_dpi: 150
  table_format: parquet     # parquet | csv
  archive: false
  max_points: 5000
```

## campaigns

Campaigns expand regulator categories into explicit regulator sets.

```yaml
campaigns:
  - name: regulators_v1
    categories: [Category1, Category2]
    within_category:
      sizes: [2, 3]
    across_categories:
      sizes: [2, 3]
      max_per_category: 2
    allow_overlap: true
    distinct_across_categories: true
    dedupe_sets: true
    selectors:
      min_info_bits: 8.0
      min_site_count: 10
    tags:
      organism: ecoli
      purpose: multi_tf_sweep
```

Notes:
- Campaigns do not execute runs; they materialize configs via `cruncher campaign generate`.

## Inspect resolved config

```bash
cruncher config summary -c path/to/config.yaml
```

## Related docs

- [Sampling + analysis](../guides/sampling_and_analysis.md)
- [Intent + lifecycle](../guides/intent_and_lifecycle.md)
- [CLI reference](cli.md)
- [Architecture and artifacts](architecture.md)
