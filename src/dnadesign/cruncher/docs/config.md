## cruncher config

This page explains the `config.yaml` and how each block maps to the **cruncher** lifecycle. The YAML root key is `cruncher`, the CLI chooses *what* runs (fetch, lock, sample, analyze), the config defines *how* each stage behaves.

### Contents

1. [Root settings](#root-settings)
2. [Categories & campaigns](#categories--campaigns)
3. [IO](#io)
4. [Motif store](#motif_store)
5. [Ingest](#ingest)
6. [Parse](#parse)
7. [Sample](#sample)
8. [Analysis](#analysis)

---

### Root settings

```yaml
cruncher:
  out_dir: runs/
  regulator_sets:
    - [lexA, cpxR]
  regulator_categories: {}
  campaigns: []
```

Notes:
- `out_dir` is resolved relative to the config file.
- Each regulator set creates its own `parse_...` and `sample_...` run folders.
- `runs/` is the recommended workspace-local runs directory (keeps outputs alongside the config).
- Config parsing is strict: unknown keys are rejected to avoid silent typos.

### Categories & campaigns

Category and campaign blocks let you generate many regulator combinations without hand-writing
pairwise sets. They are **additive** and do not change core sampling behavior.

```yaml
cruncher:
  regulator_categories:
    Category1: [CpxR, BaeR]
    Category2: [LexA, RcdA, Lrp, Fur]
    Category3: [Fnr, Fur, AcrR, SoxR, SoxS, Lrp]

  campaigns:
    - name: regulators_v1
      categories: [Category1, Category2, Category3]
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

  regulator_sets: []  # generated via cruncher campaign generate
```

Notes:
- Campaigns expand into explicit `regulator_sets`; they do not run automatically.
- Run `cruncher campaign generate --campaign <name>` to materialize a derived config.
- `allow_overlap=false` rejects TFs shared across categories.
- `distinct_across_categories=true` prevents a single TF from satisfying multiple categories.
- Selector filters require cached motifs/sites; fetch before generating if you use them.
- `selectors.min_info_bits` requires PWMs to be buildable. For site-based sources
  with variable site lengths, set `motif_store.site_window_lengths` or switch to
  matrix-based sources before enabling that selector.

### io

Optional parser extension hooks.

```yaml
io:
  parsers:
    extra_modules: []
```

Notes:
- `extra_modules` is a list of importable Python modules that register parsers via
  `@register("FMT")` decorators in `dnadesign.cruncher.io.parsers.backend`.
- Use this for custom formats without modifying cruncher core code.
- Import errors fail fast with explicit messages.

### motif_store

Controls how **cruncher** uses the local catalog and how PWMs are chosen.

```yaml
motif_store:
  catalog_root: .cruncher
  source_preference: [regulondb]
  allow_ambiguous: false
  pwm_source: matrix   # matrix | sites
  site_kinds: null     # optional filter: ["curated"], ["ht_tfbinding"], ["ht_peak"]
  combine_sites: false
  dataset_preference: []  # HT dataset ranking
  dataset_map: {}         # TF -> dataset ID
  site_window_lengths: {} # TF or dataset:<id> -> length (bp)
  site_window_center: midpoint # midpoint | summit
  min_sites_for_pwm: 2
  allow_low_sites: false
```

Notes:
- `pwm_source=matrix` uses cached motif matrices (default).
- `pwm_source=sites` builds PWMs from cached binding-site sequences at runtime.
- Local motif sources provide matrices by default. Set `ingest.local_sources[].extract_sites=true`
  to opt into MEME BLOCKS site extraction (training-set occurrences) so they can participate
  when `pwm_source=sites`.
- If site lengths vary, set `site_window_lengths` per TF or dataset.
- Window lengths must not exceed the shortest cached site length for a TF; use the
  min length from `cruncher targets stats` if unsure.
- `combine_sites=false` avoids mixing curated and HT sites unless you opt in.
- When `combine_sites=true`, lockfiles hash all matching site sets for the TF (respecting `site_kinds`); adding/removing site sets requires re-locking.
- `site_window_center=summit` requires per-site summit metadata; use `midpoint` unless your source provides summits.

### ingest

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
    allow_low_sites: false
    curated_sites: true
    ht_sites: false
    ht_dataset_sources: null
    ht_dataset_type: TFBINDING
    ht_binding_mode: tfbinding       # tfbinding | peaks
    uppercase_binding_site_only: true
  local_sources:
    - source_id: omalley_ecoli_meme
      description: O'Malley et al. E. coli MEME motifs (Supplementary Data 2)
      root: /path/to/dnadesign-data/primary_literature/OMalley_et_al/escherichia_coli_motifs
      patterns: ["*.txt"]
      recursive: false
      format_map: {".txt": "MEME"}
      default_format: null
      tf_name_strategy: stem
      matrix_semantics: probabilities
      extract_sites: false
      meme_motif_selector: null  # name_match | MEME-1 | 1 | "<MOTIF label>"
      organism:
        name: Escherichia coli
      citation: "O'Malley et al. 2021 (DOI: 10.1038/s41592-021-01312-2)"
      source_url: https://github.com/e-south/dnadesign-data
      tags:
        doi: 10.1038/s41592-021-01312-2
        title: "Persistence and plasticity in bacterial gene regulation"
        association: "TF-gene interactions"
        comments: "DAP-seq (DNA affinity purification sequencing) motifs across 354 TFs in 48 bacteria (~17,000 binding maps)."
```

Notes:
- `genome_source=ncbi` hydrates coordinate-only sites using NCBI (default).
- `genome_source=fasta` uses `genome_fasta` and optional `genome_assembly`.
- `ca_bundle` overrides the default trust store plus bundled intermediate.
- `motif_matrix_source=alignment` fails if no alignment payload exists.
- `local_sources` adds local motif directories as first-class sources.
  Roots are resolved relative to the config path if they are not absolute.
- Each local source must set `format_map` and/or `default_format` so `.txt` (or other)
  files can be parsed. Missing mappings fail fast.
- Local sources provide motif matrices by default. For MEME text output, set
  `extract_sites=true` to parse BLOCKS sites (training-set occurrences).
- `meme_motif_selector` selects a motif from multi-motif MEME files (by name match,
  MEME-1, numeric index, or exact label). Use this to disambiguate multi-motif files.
- For DAP-seq local datasets (DNA affinity purification sequencing), see the
  `dnadesign-data` repository and cite O'Malley et al. "Persistence and plasticity in
  bacterial gene regulation" (DOI: 10.1038/s41592-021-01312-2). The E. coli motifs are
  provided in MEME format (Supplementary Data 2).

### parse

Logo rendering settings for cached PWMs.

```yaml
parse:
  plot:
    logo: true
    bits_mode: information
    dpi: 150
```

Notes:
- `logo: false` skips logo rendering but still validates cached PWMs and writes a run manifest.
- When `motif_store.pwm_source=sites`, logo subtitles indicate site provenance
  (curated, high-throughput, or combined).

### sample

Sampling and optimizer settings.

```yaml
sample:
  bidirectional: true
  seed: 42
  record_tune: false
  progress_bar: true
  progress_every: 1000
  live_metrics: true
  save_trace: true
  save_sequences: true
  init:
    kind: random
    length: 30
    pad_with: background
  draws: 500
  tune: 200
  chains: 2
  min_dist: 1
  top_k: 5
  moves:
    block_len_range: [2, 6]
    multi_k_range: [2, 6]
    slide_max_shift: 4
    swap_len_range: [2, 8]
    move_probs:
      S: 0.80
      B: 0.10
      M: 0.10
  optimiser:
    kind: gibbs        # gibbs | pt
    scorer_scale: llr  # llr | z | logp | consensus-neglop-sum
    cooling:
      kind: linear
      beta: [0.0001, 0.001]
    swap_prob: 0.10
  pwm_sum_threshold: 0.0
  include_consensus_in_elites: false
```

Notes:
- `save_sequences=true` is required for `analyze` and `report`.
- `save_trace=true` is required for trace-based plots and `report`.
- `live_metrics=true` writes `live_metrics.jsonl` with progress snapshots (used by `cruncher runs watch`).
- `bidirectional=true` scores both strands (reverse complement) when scanning PWMs.
- `min_dist` is the Hamming-distance filter for elite sequences (0 disables).
- `top_k` controls how many top sequences per chain are retained before elite filtering.
- `pwm_sum_threshold` filters elites by summed normalized scores (0 keeps all).
- `include_consensus_in_elites` adds per-TF PWM consensus strings to elites metadata.
- R-hat needs ≥2 chains and ESS needs ≥4 draws; otherwise `report` shows `n/a` and records diagnostics warnings.

### analysis

Diagnostics and plotting settings for existing sample runs.

```yaml
analysis:
  runs: []
  tf_pair: [lexA, cpxR]
  archive: false
  plots:
    trace: true
    autocorr: true
    convergence: true
    scatter_pwm: true
    pair_pwm: true
    parallel_pwm: true
    score_hist: true
    score_box: false
    correlation_heatmap: true
    pairgrid: false
    parallel_coords: true
  scatter_scale: llr
  scatter_style: edges
  subsampling_epsilon: 10.0
```

Notes:
- `analysis.runs` lists sample run directory names. If empty, use `--run` or
  `--latest` when calling `cruncher analyze`.
- `tf_pair` is required for pairwise plots.
- `archive=true` moves the previous analysis into `analysis/_archive/<analysis_id>/`
  before writing the new one.
- `scatter_scale` supports `llr`, `z`, `logp`, or `consensus-neglop-sum`.
- `scatter_style` toggles scatter styling (`edges` or `thresholds`).
- `scatter_style=thresholds` requires `scatter_scale=llr` and uses `sample.pwm_sum_threshold` for the x+y cutoff.
- Threshold plots normalize per-TF LLRs by each PWM's consensus LLR (axes are 0-1).
- `subsampling_epsilon` controls how per-PWM draws are subsampled for scatter plots; it is the minimum Euclidean change in per-TF score space required to keep a draw (must be > 0).
- `cruncher analyze --list-plots` shows the registry and required inputs.
- `pairgrid` produces a pairwise projection grid across TF scores (useful for N>2).
- Analysis tables include `joint_metrics.csv`, summarizing joint score balance and Pareto-front size for elites.

### Inspect resolved config

`cruncher config` prints the resolved settings that will be used by the CLI.

Example output (captured with `CRUNCHER_LOG_LEVEL=WARNING` and `COLUMNS=200`):

```bash
                                                                                        Cruncher config summary
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Key                              ┃ Value                                                                                                                                                             ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ out_dir                          │ runs                                                                                                                                                              │
│ regulator_sets                   │ [['lexA', 'cpxR']]                                                                                                                                                │
│ regulators_flat                  │ lexA, cpxR                                                                                                                                                        │
│ regulator_categories             │ {'Stress': ['lexA'], 'Envelope': ['cpxR'], 'Category1': ['cpxR', 'baeR'], 'Category2': ['lexA', 'rcdA', 'lrp', 'fur'], 'Category3': ['fnr', 'fur', 'acrR',        │
│                                  │ 'soxR', 'soxS', 'lrp']}                                                                                                                                           │
│ campaigns                        │ demo_pair, demo_categories, demo_categories_best                                                                                                                  │
│ io.parsers.extra_modules         │ []                                                                                                                                                                │
│ pwm_source                       │ sites                                                                                                                                                             │
│ site_kinds                       │ None                                                                                                                                                              │
│ combine_sites                    │ False                                                                                                                                                             │
│ dataset_preference               │ []                                                                                                                                                                │
│ dataset_map                      │ {}                                                                                                                                                                │
│ site_window_lengths              │ {'lexA': 15, 'cpxR': 11, 'baeR': 20, 'rcdA': 10, 'lrp': 12, 'fur': 12, 'fnr': 14, 'acrR': 10, 'soxR': 18, 'soxS': 20}                                             │
│ site_window_center               │ midpoint                                                                                                                                                          │
│ min_sites_for_pwm                │ 2                                                                                                                                                                 │
│ source_preference                │ ['regulondb']                                                                                                                                                     │
│ allow_ambiguous                  │ False                                                                                                                                                             │
│ ingest.genome_source             │ ncbi                                                                                                                                                              │
│ ingest.genome_fasta              │ -                                                                                                                                                                 │
│ ingest.genome_cache              │ .cruncher/genomes                                                                                                                                                 │
│ ingest.genome_assembly           │ -                                                                                                                                                                 │
│ ingest.contig_aliases            │ {}                                                                                                                                                                │
│ ingest.ncbi_email                │ -                                                                                                                                                                 │
│ ingest.ncbi_tool                 │ cruncher                                                                                                                                                          │
│ ingest.ncbi_timeout              │ 30                                                                                                                                                                │
│ ingest.http.retries              │ 3                                                                                                                                                                 │
│ ingest.http.backoff_seconds      │ 0.5                                                                                                                                                               │
│ ingest.http.max_backoff_seconds  │ 8.0                                                                                                                                                               │
│ ingest.local_sources             │ demo_local_meme@data/local_motifs                                                                                                                                 │
│ ingest.regulondb.curated_sites   │ True                                                                                                                                                              │
│ ingest.regulondb.ht_sites        │ False                                                                                                                                                             │
│ ingest.regulondb.ht_dataset_type │ TFBINDING                                                                                                                                                         │
│ ingest.regulondb.ht_binding_mode │ tfbinding                                                                                                                                                         │
│ init.kind                        │ random                                                                                                                                                            │
│ init.length                      │ 30                                                                                                                                                                │
│ init.regulator                   │ None                                                                                                                                                              │
│ draws                            │ 500                                                                                                                                                               │
│ tune                             │ 200                                                                                                                                                               │
│ chains                           │ 2                                                                                                                                                                 │
│ top_k                            │ 5                                                                                                                                                                 │
│ min_dist                         │ 1                                                                                                                                                                 │
│ seed                             │ 42                                                                                                                                                                │
│ record_tune                      │ False                                                                                                                                                             │
│ progress_bar                     │ True                                                                                                                                                              │
│ progress_every                   │ 200                                                                                                                                                               │
│ save_trace                       │ True                                                                                                                                                              │
│ save_sequences                   │ True                                                                                                                                                              │
│ bidirectional                    │ True                                                                                                                                                              │
│ pwm_sum_threshold                │ 0.0                                                                                                                                                               │
│ include_consensus_in_elites      │ False                                                                                                                                                             │
│ optimizer.kind                   │ gibbs                                                                                                                                                             │
│ scorer_scale                     │ llr                                                                                                                                                               │
│ cooling                          │ {'kind': 'linear', 'beta': (0.0001, 0.001)}                                                                                                                       │
│ swap_prob                        │ 0.1                                                                                                                                                               │
│ analysis.runs                    │ []                                                                                                                                                                │
│ analysis.plots                   │ {'trace': True, 'autocorr': True, 'convergence': True, 'scatter_pwm': True, 'pair_pwm': True, 'parallel_pwm': True, 'pairgrid': True, 'score_hist': True,         │
│                                  │ 'score_box': False, 'correlation_heatmap': True, 'parallel_coords': True}                                                                                         │
│ analysis.scatter_scale           │ llr                                                                                                                                                               │
│ analysis.subsampling_epsilon     │ 10.0                                                                                                                                                              │
│ analysis.scatter_style           │ edges                                                                                                                                                             │
│ analysis.tf_pair                 │ ['lexA', 'cpxR']                                                                                                                                                  │
│ analysis.archive                 │ False                                                                                                                                                             │
└──────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

@e-south
