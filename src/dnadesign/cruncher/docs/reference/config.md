## cruncher config

This page explains the `config.yaml` and how each block maps to the **cruncher** lifecycle. The YAML root key is `cruncher`, the CLI chooses *what* runs (fetch, lock, sample, analyze), the config defines *how* each stage behaves.

### Contents

1. [Root settings](#root-settings)
2. [Categories and campaigns](#categories-and-campaigns)
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
- `out_dir` is resolved relative to the config file and must be a relative path.
- Each regulator set creates its own `parse_...` and `sample_...` run folders.
- Config parsing is strict: unknown keys are rejected to avoid silent typos.

### Categories and campaigns

Category and campaign blocks let you generate many regulator combinations without hand-writing pairwise sets. They are **additive** and do not change core sampling behavior.

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
  pwm_source: matrix              # matrix | sites
  site_kinds: null                # optional filter: ["curated"], ["ht_tfbinding"], ["ht_peak"]
  combine_sites: false
  pseudocounts: 0.5               # smoothing for site-derived PWMs
  dataset_preference: []          # HT dataset ranking
  dataset_map: {}                 # TF -> dataset ID
  site_window_lengths: {}         # TF or dataset:<id> -> length (bp)
  site_window_center: midpoint    # midpoint | summit
  pwm_window_lengths: {}          # TF or dataset:<id> -> PWM trim length (bp)
  pwm_window_strategy: max_info   # max_info
  min_sites_for_pwm: 2
  allow_low_sites: false
```

Notes:
- `pwm_source=matrix` uses cached motif matrices (default).
- `pwm_source=sites` builds PWMs from cached binding-site sequences at runtime.
- `pseudocounts` controls PWM smoothing when building from sites (Biopython).
- `catalog_root` must be workspace-relative (no absolute paths or `..` segments).
- Local motif sources provide matrices by default. Set `ingest.local_sources[].extract_sites=true`
  to opt into MEME BLOCKS site extraction (training-set occurrences) so they can participate
  when `pwm_source=sites`.
- If site lengths vary for site-derived PWMs, set `site_window_lengths` per TF or dataset. MEME/STREME
  discovery uses raw cached sites unless `motif_discovery.window_sites=true`.
- Window lengths must not exceed the shortest cached site length for a TF; use the
  min length from `cruncher targets stats` if unsure.
- If PWM length must be constrained (e.g., optimization length is shorter), set
  `pwm_window_lengths` to select a contiguous sub-window by information content.
- `combine_sites=false` avoids mixing curated, HT, and local sites unless you opt in.
- When `combine_sites=true`, lockfiles hash all matching site sets for the TF (respecting `site_kinds`); adding/removing site sets requires re-locking.
- `site_window_center=summit` requires per-site summit metadata; use `midpoint` unless your source provides summits.

### motif_discovery

Motif discovery / alignment via MEME Suite (STREME or MEME). This step produces
new motif matrices from cached binding sites and stores them back into the catalog.

```yaml
motif_discovery:
  tool: auto                     # auto | streme | meme
  tool_path: null                # optional path to meme/streme or a bin dir
  window_sites: false            # pre-window binding sites before discovery
  minw: null                      # minimum motif width (auto from site lengths if unset)
  maxw: null                      # maximum motif width (auto from site lengths if unset)
  nmotifs: 1                      # motifs per TF
  meme_mod: null                  # optional MEME -mod setting: oops | zoops | anr
  min_sequences_for_streme: 50    # auto threshold
  source_id: meme_suite           # catalog source ID
  replace_existing: true          # replace prior discovered motifs for same TF/source
```

Notes:
- `tool=auto` chooses STREME when there are enough sequences (>= min_sequences_for_streme), MEME otherwise.
- Discovery requires cached binding sites (run `cruncher fetch sites`) and uses the MEME Suite
  CLI tools (`streme`/`meme`) on PATH. This is independent of `motif_store.pwm_source`.
- Discovery always uses cached binding sites, even when `pwm_source=matrix`.
- By default, discovery uses raw cached site sequences. Set `motif_discovery.window_sites=true`
  to pre-window binding sites using `motif_store.site_window_lengths` (errors if no window lengths are set).
- If `minw`/`maxw` are unset, Cruncher derives them from the min/max site lengths per TF.
- Use `cruncher targets stats` to choose `minw/maxw` based on site-length ranges; avoid narrow caps that force truncated motifs.
- If you run both MEME and STREME, use distinct `motif_discovery.source_id` values between runs so `lock` can disambiguate.
- You can override `motif_discovery.source_id` per run with `cruncher discover motifs --source-id ...`.
- When `replace_existing=true` (default), re-running discovery replaces previous discovered motifs for the
  same TF/source to avoid cache bloat. Set it to false if you want to keep historical runs.
- `meme_mod` applies to MEME only; leave it unset to use MEME defaults.
- Use `motif_discovery.tool_path` (or set `MEME_BIN`) to point at a versioned MEME Suite install
  without modifying your PATH. Relative `tool_path` values are resolved from the config file
  location, so prefer absolute paths when using repo-level `.pixi/` installs.
- When `tool=auto`, prefer a bin directory (not a single executable) so both tools can be resolved.
- To use newly discovered motifs in downstream runs, set `motif_store.pwm_source: matrix`,
  add `meme_suite` (or your chosen `source_id`) to `motif_store.source_preference`, and re-run
  `cruncher lock <config>` to refresh the lockfile.

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
    pseudocounts: 0.5
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
- `genome_cache` must be workspace-relative (no absolute paths or `..` segments).
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
    block_len_range: [3, 12]
    multi_k_range: [2, 8]
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
      beta: [0.01, 0.1]
    swap_prob: 0.10
  auto_opt:
    enabled: true
    pilot_draws: 200
    pilot_tune: 100
    pilot_chains_gibbs: 2
    pilot_chains_pt: 4
    retry_on_warn: true
    retry_draws_factor: 2.0
    retry_tune_factor: 2.0
    cooling_boost: 5.0
    max_rhat: 1.2
    min_ess: 20
    min_unique_fraction: 0.10
    pt_beta_min: 0.2
    pt_beta_max: 1.0
  pwm_sum_threshold: 0.0
  include_consensus_in_elites: false
```

Notes:
- `save_sequences=true` is required for `analyze` and `report`.
- `save_trace=true` is required for trace-based plots and `report`.
- `live_metrics=true` writes `live/metrics.jsonl` with progress snapshots (used by `cruncher runs watch`).
- `bidirectional=true` scores both strands (reverse complement) when scanning PWMs.
- `min_dist` is the Hamming-distance filter for elite sequences (0 disables).
- `top_k` controls how many top sequences per chain are retained before elite filtering.
- `pwm_sum_threshold` filters elites by summed normalized scores (0 keeps all).
- `include_consensus_in_elites` adds per-TF PWM consensus strings to elites metadata.
- R-hat needs ≥2 chains and ESS needs ≥4 draws; otherwise `report` shows `n/a` and records diagnostics warnings.
- `gibbs` expects `cooling.kind` to be `fixed` or `linear`; `pt` expects `geometric` (beta ladder) or `fixed` with a single chain.
- `auto_opt` runs short Gibbs + PT pilots, compares diagnostics (ESS/R-hat/unique_fraction/best_score), logs the decision, and runs a final sample using the selected optimizer.
- Auto-opt pilots always write `trace.nc` + `sequences.parquet` (required for diagnostics) and are stored under `runs/pilot/`.
- Auto-opt is enabled by default; set `auto_opt.enabled: false` or use `--no-auto-opt` to disable.
- If no pilot meets the quality thresholds, auto-opt retries with cooler settings (and raises an error if still unstable).
- Optional: set `auto_opt.pt_beta` to a full beta ladder (must match `sample.chains` for PT finals).
- If the base config uses PT, Gibbs pilots use `auto_opt.gibbs_cooling` (default: linear 0.01→0.1).
PT starting point (optional; requires a beta ladder):
```yaml
optimiser:
  kind: pt
  cooling:
    kind: geometric
    beta: [0.2, 0.4, 0.7, 1.0]
  swap_prob: 0.20
```
Note: `cooling.beta` must be the same length as `sample.chains` (one β per chain).
Tuning hints (use `analysis/tables/diagnostics.json` and `report/report.json`):
- Low ESS / high R-hat → increase `draws`/`tune` first; PT can help but is not always better.
- PT swap acceptance < ~0.05 → widen the beta ladder or increase `swap_prob`.
- Very high/low block or multi-site acceptance → adjust move ranges or cooling strength.

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
  scatter_background: true
  scatter_background_samples: null
  scatter_background_seed: 0
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
- `scatter_background=true` adds a random-sequence baseline cloud to `pwm__scatter`.
- `scatter_background_samples` controls how many random sequences to draw (defaults to the MCMC subsample size).
- `scatter_background_seed` makes the random baseline reproducible.
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
│ regulator_sets                   │ [['lexA', 'cpxR', 'fur']]                                                                                                                                         │
│ regulators_flat                  │ lexA, cpxR, fur                                                                                                                                                   │
│ regulator_categories             │ {'Stress': ['lexA'], 'Envelope': ['cpxR'], 'Category1': ['cpxR', 'baeR'], 'Category2': ['lexA', 'rcdA', 'lrp', 'fur'], 'Category3': ['fnr', 'fur', 'acrR',        │
│                                  │ 'soxR', 'soxS', 'lrp']}                                                                                                                                           │
│ campaigns                        │ demo_pair, demo_categories, demo_categories_best                                                                                                                  │
│ io.parsers.extra_modules         │ []                                                                                                                                                                │
│ pwm_source                       │ sites                                                                                                                                                             │
│ site_kinds                       │ None                                                                                                                                                              │
│ combine_sites                    │ False                                                                                                                                                             │
│ pseudocounts                     │ 0.5                                                                                                                                                               │
│ dataset_preference               │ []                                                                                                                                                                │
│ dataset_map                      │ {}                                                                                                                                                                │
│ site_window_lengths              │ {'lexA': 15, 'cpxR': 11, 'baeR': 20, 'rcdA': 10, 'lrp': 12, 'fur': 12, 'fnr': 14, 'acrR': 10, 'soxR': 18, 'soxS': 20}                                             │
│ site_window_center               │ midpoint                                                                                                                                                          │
│ pwm_window_lengths               │ {}                                                                                                                                                                │
│ pwm_window_strategy              │ max_info                                                                                                                                                          │
│ min_sites_for_pwm                │ 2                                                                                                                                                                 │
│ source_preference                │ ['regulondb']                                                                                                                                                     │
│ allow_ambiguous                  │ False                                                                                                                                                             │
│ motif_discovery.tool             │ auto                                                                                                                                                              │
│ motif_discovery.tool_path        │ -                                                                                                                                                                 │
│ motif_discovery.window_sites     │ False                                                                                                                                                             │
│ motif_discovery.minw             │ -                                                                                                                                                                 │
│ motif_discovery.maxw             │ -                                                                                                                                                                 │
│ motif_discovery.nmotifs          │ 1                                                                                                                                                                 │
│ motif_discovery.min_sequences_for_streme │ 50                                                                                                                                                          │
│ motif_discovery.source_id        │ meme_suite                                                                                                                                                        │
│ motif_discovery.replace_existing │ True                                                                                                                                                              │
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
│ cooling                          │ {'kind': 'linear', 'beta': (0.01, 0.1)}                                                                                                                           │
│ swap_prob                        │ 0.1                                                                                                                                                               │
│ analysis.runs                    │ []                                                                                                                                                                │
│ analysis.plots                   │ {'trace': True, 'autocorr': True, 'convergence': True, 'scatter_pwm': True, 'pair_pwm': True, 'parallel_pwm': True, 'pairgrid': True, 'score_hist': True,         │
│                                  │ 'score_box': False, 'correlation_heatmap': True, 'parallel_coords': True}                                                                                         │
│ analysis.scatter_scale           │ llr                                                                                                                                                               │
│ analysis.subsampling_epsilon     │ 10.0                                                                                                                                                              │
│ analysis.scatter_style           │ edges                                                                                                                                                             │
│ analysis.scatter_background      │ True                                                                                                                                                              │
│ analysis.scatter_background_samples │ None                                                                                                                                                            │
│ analysis.scatter_background_seed │ 0                                                                                                                                                                 │
│ analysis.tf_pair                 │ None                                                                                                                                                              │
│ analysis.archive                 │ False                                                                                                                                                             │
└──────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

@e-south
