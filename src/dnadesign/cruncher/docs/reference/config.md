## cruncher config

This page explains the `config.yaml` and how each block maps to the **cruncher** lifecycle. The YAML root key is `cruncher`, the CLI chooses *what* runs (fetch, lock, sample, analyze), the config defines *how* each stage behaves.

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
- Each regulator set creates its own run folder under `runs/<stage>/`.
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

Sampling and optimizer settings.

```yaml
sample:
  mode: optimize          # optimize | sample
  rng:
    seed: 42
    deterministic: true
  budget:
    tune: 200
    draws: 500
    restarts: 2
  early_stop:
    enabled: true
    patience: 500
    min_delta: 0.5
  init:
    kind: random
    length: 30
    pad_with: background
  objective:
    bidirectional: true
    score_scale: normalized-llr
    scoring:
      pwm_pseudocounts: 0.10
      log_odds_clip: null
    softmin:
      enabled: true
      kind: linear
      beta: [0.5, 10.0]
  elites:
    k: 5
    min_hamming: 1
    filters:
      pwm_sum_min: 0.0
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
  optimizer:
    name: auto          # auto | gibbs | pt
  optimizers:
    gibbs:
      beta_schedule:
        kind: linear
        beta: [0.05, 0.5]
      apply_during: tune
      schedule_scope: per_chain
      adaptive_beta:
        enabled: true
        target_acceptance: 0.40
        window: 100
        k: 0.50
        min_beta: 1.0e-3
        max_beta: 10.0
        moves: [B, M]
        stop_after_tune: true
    pt:
      beta_ladder:
        kind: geometric
        betas: [0.05, 0.1, 0.2, 0.4]
      swap_prob: 0.10
      ladder_adapt:
        enabled: false
        target_swap: 0.25
        window: 50
        k: 0.50
        min_scale: 0.25
        max_scale: 4.0
        stop_after_tune: true
  auto_opt:
    enabled: true
    budget_levels: [200, 800]
    eta: 3
    replicates: 1
    keep_pilots: ok
    prefer_simpler_if_close: true
    tolerance:
      score: 0.01
    length:
      enabled: true
      min_length: null
      max_length: null
      step: 2
      max_candidates: 4
    policy:
      allow_warn: false
      retry_on_warn: true
      retry_draws_factor: 2.0
      retry_tune_factor: 2.0
      cooling_boost: 5.0
      max_rhat: 1.2
      min_ess: 20
      min_unique_fraction: 0.10
      max_unique_fraction: null
      scorecard:
        top_k: 10
        min_balance: 0.25
        min_diversity: 0.10
        acceptance_target: 0.40
        acceptance_tolerance: 0.25
        swap_target: 0.25
        swap_tolerance: 0.20
  output:
    save_sequences: true
    include_consensus_in_elites: false
    live_metrics: true
    trace:
      save: true
      include_tune: false
    trim:
      enabled: true
      padding: 1
      require_non_decreasing: true
    polish:
      enabled: true
      max_rounds: 2
      improvement_tol: 0.0
  ui:
    progress_bar: true
    progress_every: 0
```

Notes:
- `output.save_sequences=true` is required for `analyze` and `report`.
- `output.trace.save=true` is required for trace-based plots and `report`.
- `output.live_metrics=true` writes `live/metrics.jsonl` with progress snapshots (used by `cruncher runs watch`).
- `output.trace.include_tune=true` includes burn-in samples in `trace.nc` and `sequences.parquet`.
- `objective.bidirectional=true` scores both strands (reverse complement) when scanning PWMs.
- `elites.min_hamming` is the Hamming-distance filter for elites (0 disables). If `output.trim.enabled=true` yields variable lengths, the distance is computed over the shared prefix plus the length difference.
- `elites.k` controls how many sequences are retained before diversity filtering.
- `objective.scoring.pwm_pseudocounts` smooths matrix-derived PWMs (set to 0 for raw matrices).
- `objective.scoring.log_odds_clip` caps log-odds magnitudes (use to avoid extreme cliffs).
- `elites.filters.pwm_sum_min` filters elites by summed normalized scores (0 keeps all).
- `output.include_consensus_in_elites` adds per-TF PWM consensus strings to elites metadata.
- `objective.softmin` controls the min-approximation hardness separately from MCMC temperature.
- `optimizers.gibbs.beta_schedule` sets the MCMC temperature schedule; use `apply_during: tune` to anneal only during burn-in.
- `optimizers.gibbs.schedule_scope` controls whether the beta schedule is applied per chain (`per_chain`) or across all chains (`global`). Global schedules require `apply_during: all`.
- `optimizers.pt.beta_ladder` defines the temperature ladder; PT does not support `budget.restarts>1` (use ladder size for chain count).
- `adaptive_beta` tunes Gibbs acceptance (B/M moves) toward a target band; `ladder_adapt` tunes PT ladder scale.
- `auto_opt` runs short Gibbs + PT pilots, compares balance/diversity/acceptance signals, logs the decision, and runs a final sample using the selected optimizer.
- Auto-opt pilots always write `trace.nc` + `sequences.parquet` (required for diagnostics) and are stored under `runs/auto_opt/`.
- The selected pilot is recorded in `runs/auto_opt/best_<run_group>.json` (run_group is the TF slug; it uses a `setN_` prefix only when multiple regulator sets are configured) and marked with a leading `*` in `cruncher runs list`.
- Auto-opt is enabled by default when `optimizer.name=auto`; set `auto_opt.enabled: false` or pass `--no-auto-opt` to disable.
- If no pilot meets the quality thresholds, auto-opt retries with cooler settings. By default it **fails fast**; set `auto_opt.policy.allow_warn: true` to proceed with the best available candidate (logging warnings). When a boosted-cooling retry wins, the final run uses that boosted schedule (recorded in `config_used.yaml` and auto-opt notes).
- `early_stop` halts sampling when the best score fails to improve by `min_delta` for `patience` draws (per chain for Gibbs, per sweep for PT).
- Auto-opt selection details are stored in each pilot's `meta/run_manifest.json`; `cruncher analyze` writes `analysis/tables/auto_opt_pilots.csv` and `analysis/plots/auto_opt_tradeoffs.png`.
- `sample.rng.deterministic=true` isolates a stable RNG stream per pilot config.
- `auto_opt.length` searches candidate lengths; the shortest passing the scorecard is preferred, and length comparisons are ranked by normalized balance to reduce length bias.
- `auto_opt.policy.scorecard.top_k` controls how many pilot elites are used for balance/diversity scoring.
- `auto_opt.keep_pilots` controls pilot retention after selection: `all` keeps everything, `ok` keeps candidates that met thresholds (plus the selected pilot), and `best` keeps only the selected pilot.
- `moves.overrides.move_schedule` interpolates between `moves.overrides.move_probs` (start) and `move_schedule.end` (end).
- `moves.overrides.target_worst_tf_prob` biases proposals toward the current worst TF window (0 disables).
- `moves.overrides.insertion_consensus_prob` controls whether insertion moves use consensus vs PWM sampling.
- `output.trim` shrinks elites to the minimal best-hit window union; it fails fast if a trim would be shorter than the widest PWM. `output.polish` runs deterministic coordinate ascent.
PT starting point (optional; uses a beta ladder):
```yaml
sample:
  optimizer:
    name: pt
  budget:
    restarts: 1
  optimizers:
    pt:
      beta_ladder:
        kind: geometric
        betas: [0.2, 0.4, 0.7, 1.0]
      swap_prob: 0.20
```
Tuning hints (use `analysis/tables/diagnostics.json` and `report/report.json`):
- Low ESS / high R-hat → increase `budget.draws`/`budget.tune` first; PT can help but is not always better.
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
- `scatter_scale` supports `llr`, `z`, `logp`, `normalized-llr`, or `consensus-neglop-sum`.
- `scatter_style` toggles scatter styling (`edges` or `thresholds`).
- `scatter_style=thresholds` requires `scatter_scale=llr` and uses `sample.elites.filters.pwm_sum_min` for the x+y cutoff.
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

The output is a flattened key/value table; use it to confirm the resolved
`sample.*` and `motif_store.*` settings match your intent.

---

@e-south
