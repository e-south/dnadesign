## cruncher config


## Contents
- [cruncher config](#cruncher-config)
- [Root settings](#root-settings)
- [Categories and campaigns](#categories-and-campaigns)
- [io](#io)
- [motif_store](#motif_store)
- [motif_discovery](#motif_discovery)
- [ingest](#ingest)
- [parse](#parse)
- [sample](#sample)
- [analysis](#analysis)
- [Inspect resolved config](#inspect-resolved-config)

This reference lists the current Cruncher config schema and documents how each block maps to runtime behavior.

### Root settings

```yaml
cruncher:
  out_dir: outputs/
  regulator_sets:
    - [lexA, cpxR]
  regulator_categories: {}
  campaigns: []
```

Notes:
- `out_dir` is resolved relative to the config file and must be a relative path.
- Each regulator set creates its own run folder under `outputs/<stage>/`.
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
- `allow_overlap=false` rejects TFs shared across categories (campaign construction only; it does not constrain motif overlap in sequences).
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
  pwm_window_lengths: {}          # TF or dataset:<id> -> PWM window length (bp)
  pwm_window_strategy: max_info   # max_info
  min_sites_for_pwm: 2
  allow_low_sites: false
```

Notes:
- `pwm_source=matrix` uses cached motif matrices (default).
- `pwm_source=sites` builds PWMs from cached binding-site sequences at runtime.
- `pseudocounts` controls PWM smoothing when building from sites (Biopython).
- `catalog_root` can be absolute or relative to the cruncher root (`src/dnadesign/cruncher`); relative paths must not include `..`.
- Local motif sources provide matrices by default. Set `ingest.local_sources[].extract_sites=true`
  to opt into MEME BLOCKS site extraction for site‑derived PWMs.
- If site lengths vary for site-derived PWMs, set `site_window_lengths` per TF or dataset. MEME/STREME
  discovery uses raw cached sites unless `motif_discovery.window_sites=true`.
- If PWM length must be constrained (e.g., optimization length is shorter), set
  `pwm_window_lengths` to select a contiguous sub-window by information content.
- `combine_sites=false` avoids mixing curated, HT, and local sites unless you opt in.
- When `combine_sites=true`, lockfiles hash all matching site sets for the TF; adding/removing site sets requires re‑locking.
- `site_window_center=summit` requires per‑site summit metadata.

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
  meme_prior: null                # optional MEME -prior setting: dirichlet | dmix | mega | megap | addone
  min_sequences_for_streme: 50    # auto threshold
  source_id: meme_suite           # catalog source ID
  replace_existing: true          # replace prior discovered motifs for same TF/source
```

Notes:
- Discovery requires cached binding sites and MEME Suite binaries on PATH (or `motif_discovery.tool_path`).
- `tool=auto` chooses STREME when `min_sequences_for_streme` is met, MEME otherwise.
- `minw/maxw` default to per‑TF site length ranges unless set explicitly.
- Use distinct `source_id` values for MEME vs STREME so `lock` can disambiguate.
- Re‑run `cruncher lock` after discovery to pin the new matrices.

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
  site_sources:
    - source_id: baer_chip_exo
      description: Choudhary et al. BaeR ChIP-exo binding sites (processed FASTA)
      path: /path/to/dnadesign-data/primary_literature/Choudhary_et_al/processed/BaeR_binding_sites.fasta
      tf_name: BaeR
      record_kind: chip_exo
      organism:
        name: Escherichia coli
        strain: K-12 MG1655
        assembly: NC_000913.3
      citation: "Choudhary et al. 2020 (DOI: 10.1128/mSystems.00980-20)"
      source_url: https://doi.org/10.1128/mSystems.00980-20
      tags:
        assay: chip_exo
        doi: 10.1128/mSystems.00980-20
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
- `site_sources` adds local FASTA binding-site sources.
  Headers should start with a TF name (e.g., `BaeR|peak_0|NC_000913.3:1-5|strand=-|...`).
  `record_kind` becomes the cached `site_kind` label for filtering or display.
- `meme_motif_selector` selects a motif from multi-motif MEME files (by name match,
  MEME-1, numeric index, or exact label). Use this to disambiguate multi-motif files.
- For DAP-seq local datasets (DNA affinity purification sequencing), see the
  `dnadesign-data` repository and cite O'Malley et al. "Persistence and plasticity in
  bacterial gene regulation" (DOI: 10.1038/s41592-021-01312-2). The E. coli motifs are
  provided in MEME format (Supplementary Data 2).

### parse

Logo rendering defaults for `cruncher catalog logos` (legacy location under `parse.plot`).

```yaml
parse:
  plot:
    logo: true
    bits_mode: information
    dpi: 150
```

Notes:
- `cruncher parse` no longer renders logos; it only validates locked motifs.
- `plot.logo` is currently unused (reserved for future toggles).
- When `motif_store.pwm_source=sites`, logo subtitles indicate site provenance (curated, high-throughput, or combined).

### sample

Sampling settings.

```yaml
sample:
  mode: optimize          # optimize | sample
  rng:
    seed: 42
    deterministic: true
  sequence_length: 30
  compute:
    total_sweeps: 3000
    adapt_sweep_frac: 0.34
  early_stop:
    enabled: true
    patience: 500
    min_delta: 0.01
    require_min_unique: false
    min_unique: 20
    success_min_per_tf_norm: 0.80
  init:
    kind: random
    pad_with: background
  objective:
    bidirectional: true
    score_scale: normalized-llr
    combine: min
    allow_unscaled_llr: false
    scoring:
      pwm_pseudocounts: 0.10
      log_odds_clip: null
    softmin:
      enabled: true
      kind: linear
      beta: [0.5, 10.0]
    length_penalty_lambda: 0.0
  elites:
    k: 5
    min_per_tf_norm: null
    require_all_tfs_over_min_norm: true
    mmr_alpha: 0.85
  moves:
    profile: balanced
    overrides:
      block_len_range: [3, 12]
      multi_k_range: [2, 8]
      slide_max_shift: 4
      swap_len_range: [2, 8]
      move_probs:
        S: 0.85
        B: 0.05
        M: 0.05
        L: 0.03
        W: 0.01
        I: 0.01
  output:
    save_sequences: true
    include_consensus_in_elites: false
    live_metrics: true
    trace:
      save: true
      include_tune: false
  ui:
    progress_bar: true
    progress_every: 0
```

Notes:
- `output.save_sequences=true` is required for `analyze` and `report`.
- `output.trace.save=true` is required for trace-based plots and `report`.
- `output.live_metrics=true` writes `live/metrics.jsonl` with progress snapshots (used by `cruncher runs watch`).
- `output.trace.include_tune=true` includes adaptation samples in `sequences.parquet` (trace.nc always contains draws only).
- `sequences.parquet` includes `min_per_tf_norm` (alias of `min_norm`) for the per‑TF normalized minimum.
- `sample.rng.deterministic=true` derives a stable RNG stream from config + locks.
- `objective.bidirectional=true` scores both strands (reverse complement) when scanning PWMs.
- When `objective.bidirectional=true`, dsDNA equivalence is always on for uniqueness and MMR selection (reverse complements are treated as identical).
- Best‑hit tie‑breaking is deterministic: highest score wins, then smallest start index, then `+` strand.
- `objective.combine` controls how per-TF scores are combined (`min` for weakest-TF optimization, `sum` for sum-based).
- `objective.allow_unscaled_llr=true` allows `score_scale=llr` in multi-TF runs (otherwise validation fails).
- `objective.score_scale=logp` is FIMO‑like: it uses a DP‑derived null
  distribution under a 0‑order background to compute a tail p‑value for the
  best window, then converts to a sequence‑level p via
  `p_seq = 1 − (1 − p_win)^n_windows` before reporting `−log10(p_seq)`.
  When `objective.bidirectional=true`, `n_windows` counts both strands
  (effective tests = `2 * (L − w + 1)`).
- `sample.sequence_length` is fixed for sampling and must be >= the widest PWM length.
- Cruncher runs are fixed-length only; there is no trim/polish or length ladder in the schema.
- `sample.compute.total_sweeps` is the total sweep budget across adaptation + draws.
- `sample.compute.adapt_sweep_frac` controls the adaptation share; draws are the remainder.
- `sample.compute.adapt_sweep_frac` must leave at least one draw (validation fails otherwise).
- `elites.k` controls how many sequences are retained after selection (must be >= 1).
- TFBS‑core MMR behavior: for each sequence we extract the best‑hit window for each TF (oriented to the PWM); when comparing two sequences we compute same‑TF core distances (e.g., LexA vs LexA, CpxR vs CpxR) and then average across TFs. We never compare different TFs within the same sequence.
- “Tolerant” weights mean low‑information PWM positions are weighted more when measuring diversity (`weight = 1 - info_norm`), preserving consensus‑critical positions while encouraging diversity at flexible bases.
- `elites.mmr_alpha` controls relevance vs diversity (1.0 = relevance-only).
- `objective.scoring.pwm_pseudocounts` smooths matrix-derived PWMs (set to 0 for raw matrices).
- `objective.scoring.log_odds_clip` caps log-odds magnitudes (use to avoid extreme cliffs).
- `objective.length_penalty_lambda` subtracts `lambda * (L - sequence_length)` from combined scores (mitigates length bias).
- `elites.min_per_tf_norm` filters elites by per-TF normalized minimum. With `normalized-llr`,
  0.0–1.0 corresponds to background‑like → consensus‑like; values around 0.05–0.2 are a common starting band.
- `elites.require_all_tfs_over_min_norm` controls whether the per‑TF threshold must be met by every TF.
- `output.include_consensus_in_elites` adds per-TF PWM consensus strings to elites metadata.
- `objective.softmin` controls the min-approximation hardness separately from MCMC temperature.
- `early_stop` halts sampling when the best score fails to improve by `min_delta` for `patience` sweeps. If `require_min_unique=true`, early-stop can only trigger after `min_unique` canonical successes above `success_min_per_tf_norm`. For `score_scale=normalized-llr` (0-1), `min_delta` must be <= 0.1; use ~0.01 to detect plateaus.
- `moves.overrides.move_schedule` interpolates between `moves.overrides.move_probs` (start) and `move_schedule.end` (end).
- `moves.overrides.target_worst_tf_prob` biases proposals toward the current worst TF window (0 disables).
- `moves.overrides.insertion_consensus_prob` controls whether insertion moves use consensus vs PWM sampling.

### analysis

Diagnostics and plotting settings for existing sample runs.

```yaml
analysis:
  runs: []
  extra_plots: false
  extra_tables: false
  mcmc_diagnostics: false
  dashboard_only: true
  table_format: parquet
  plot_format: png
  plot_dpi: 150
  png_compress_level: 9
  include_sequences_in_tables: false
  tf_pair: [lexA, cpxR]
  archive: false
  plots:
    dashboard: true
    worst_tf_trace: true
    worst_tf_identity: true
    elite_filter_waterfall: true
    overlap_heatmap: true
    overlap_bp_distribution: true
  scatter_scale: llr
  scatter_style: edges
  scatter_background: true
  scatter_background_samples: null
  scatter_background_seed: 0
  subsampling_epsilon: 10.0
```

Notes:
- `analysis.runs` lists sample run directory names. If empty, `cruncher analyze`
  defaults to the latest sample run (same as `--latest`); use `--run` to target
  specific runs.
- `extra_plots=true` enables non-Tier‑0 plots (scatter, pairwise, score histograms, overlap strand combos).
- If any non‑Tier‑0 plot keys are explicitly set `true` under `analysis.plots`, cruncher
  auto‑enables `extra_plots` and records the adjustment in `analysis_used.yaml`.
- `extra_tables=true` enables optional tables like per-PWM scatter tables and MMR summary tables.
- `mcmc_diagnostics=true` enables trace-based diagnostics and move/pt swap plots/tables.
- If any MCMC diagnostic plots are explicitly set `true`, cruncher auto‑enables
  `mcmc_diagnostics` and records the adjustment in `analysis_used.yaml`.
- Trace-based diagnostics require `output.trace.save=true`; analyze fails if `trace.nc` is missing.
- Tier‑0 plots (dashboard + worst‑TF + overlap summaries) default to `true`;
  all other plot keys default to `false`. Use `cruncher analyze --plots all`
  to generate the full plot suite.
- The canonical artifact summary is `analysis/summary.json`. A detailed inventory
  with reasons (default/extra/mcmc) is written to `analysis/manifest.json`.
- Analysis reports live in `analysis/report.json` and `analysis/report.md`.
- `tf_pair` is optional; if omitted, cruncher auto-selects a deterministic pair for pairwise plots.
- `archive=true` moves the previous analysis into `analysis/_archive/<analysis_id>/`
  before writing the new one.
- `dashboard_only=true` suppresses redundant component plots when the dashboard is enabled.
- `table_format` defaults to `parquet` and controls analysis table extensions.
- `scatter_scale` supports `llr`, `z`, `logp`, `normalized-llr`, or `consensus-neglop-sum`.
- `scatter_style` toggles scatter styling (`edges` or `thresholds`).
- `scatter_style=thresholds` requires `scatter_scale=llr` and uses `2 * sample.elites.min_per_tf_norm` for the x+y cutoff.
- `plot_format` selects the single output format for all plots (`png`, `pdf`, or `svg`).
- `plot_dpi` and `png_compress_level` control plot output size.
- Threshold plots normalize per-TF LLRs by each PWM's consensus LLR (axes are 0-1).
- `scatter_background=true` adds a random-sequence baseline cloud to `pwm__scatter`.
- `scatter_background_samples` controls how many random sequences to draw (defaults to the MCMC subsample size).
- `scatter_background_seed` makes the random baseline reproducible.
- `subsampling_epsilon` controls how per-PWM draws are subsampled for scatter plots; it is the minimum Euclidean change in per-TF score space required to keep a draw (must be > 0).
- `cruncher analyze --list-plots` shows the registry and required inputs.
- `pairgrid` produces a pairwise projection grid across TF scores (useful for N>2).
- Analysis tables include `score_summary.parquet`, `joint_metrics.parquet`, overlap summaries
  (`overlap_summary.parquet`, `elite_overlap.parquet`), `objective_components.json`, and
  optional move/ladder tables (`move_stats_summary.parquet`, `pt_swap_pairs.parquet`).

### Inspect resolved config

`cruncher config` prints the resolved settings that will be used by the CLI.

The output is a flattened key/value table; use it to confirm the resolved
`sample.*` and `motif_store.*` settings match your intent.

---

@e-south
