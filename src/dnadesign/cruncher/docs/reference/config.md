## Cruncher config

**Last updated by:** cruncher-maintainers on 2026-02-23

### Contents
- [Overview](#overview)
- [Root](#root)
- [Workspace](#workspace)
- [io](#io)
- [catalog](#catalog)
- [discover](#discover)
- [ingest](#ingest)
- [sample](#sample)
- [analysis](#analysis)
- [Inspect resolved config](#inspect-resolved-config)
- [Related docs](#related-docs)

### Overview

Cruncher uses a single root key and strict validation. Unknown keys and missing required keys are errors.

- Root key: `cruncher`
- Schema version: `schema_version: 3`
- Fixed-length invariant: `sample.sequence_length >= max_pwm_width`

Use this doc as a *schema map*. If you only change a few knobs, start with:
`workspace.regulator_sets`, `catalog.pwm_source`, `sample.sequence_length`,
`sample.budget.*`, and `sample.elites.*`.

### Root

```yaml
cruncher:
  schema_version: 3
  workspace: { ... }
  io: { ... }
  catalog: { ... }
  discover: { ... }
  ingest: { ... }
  sample: { ... }            # optional unless running `cruncher sample`
  analysis: { ... }          # optional; analyze uses schema defaults when omitted
```

Notes:
- `sample` is required for `cruncher sample`, but analyze/reporting consume run artifacts and do not require `sample` in the current config.
- `analysis` is optional; when omitted, analyze uses default settings (`run_selector=latest`, `pairwise=auto`, `plot_format=pdf`, `table_format=parquet`).

### Workspace

```yaml
workspace:
  out_dir: outputs/
  regulator_sets:
    - [lexA, cpxR]
```

Notes:
- `out_dir` is resolved relative to the config file and must be a relative path.
- `regulator_sets` defines direct run targets (each set becomes a run).
- `regulator_sets` must be non-empty.

### io

```yaml
io:
  parsers:
    extra_modules: []
```

Notes:
- `extra_modules` is a list of importable Python modules that register parsers via
  `@register("FMT")` decorators in `dnadesign.cruncher.io.parsers.backend`.

### catalog

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
  min_sites_for_pwm: 2
  allow_low_sites: false
```

Notes:
- `catalog.root` accepts absolute paths or workspace-relative paths (relative paths are resolved from the config workspace root and cannot include `..`).
- `pwm_source=matrix` uses cached motif matrices (default).
- `pwm_source=sites` builds PWMs from cached binding-site sequences at runtime.
- `source_preference` is strict when set: lock resolution fails if no candidate matches the preferred source list.
- If site lengths vary, set `site_window_lengths` per TF or dataset.

### discover

Motif discovery/alignment via MEME Suite.

```yaml
discover:
  enabled: false
  tool: auto                     # auto | streme | meme
  tool_path: null                # optional path to meme/streme or a bin dir
  window_sites: false            # pre-window binding sites before discovery
  minw: null                     # minimum motif width (tool default if unset)
  maxw: null                     # maximum motif width (tool default if unset)
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
  local_sources: []
  site_sources: []
```

Notes:
- `ht_sites: true` is strict. HT discovery/fetch failures are not downgraded to curated-only results.
- `ht_binding_mode: tfbinding` may return zero rows for some datasets; use `peaks` for peak-only datasets.
- `dataset_source` filtering is applied to each returned dataset row (not only request parameters).
- With `curated_sites: true` and `ht_sites: true`, `fetch sites --limit` requires explicit mode (`--dataset-id` or one source class disabled).

### sample

Fixed-length sampling with gibbs annealing optimization and MMR elite selection.

```yaml
sample:
  seed: 42
  sequence_length: 30
  motif_width:
    minw: null
    maxw: 30
    strategy: max_info

  budget:
    tune: 1000
    draws: 2000

  objective:
    bidirectional: true
    score_scale: normalized-llr   # normalized-llr | llr | logp | z | consensus-neglop-sum
    combine: min                  # min | sum
    softmin:
      enabled: true
      schedule: fixed             # fixed | linear
      beta_start: 0.5
      beta_end: 6.0
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
        freeze_after_sweep: null   # int | null
        freeze_after_beta: null    # float | null
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
        freeze_after_sweep: null   # int | null
        freeze_after_beta: null    # float | null
      gibbs_inertia:
        enabled: false
        kind: linear               # fixed | linear
        p_stay_start: 0.0
        p_stay_end: 0.0

  optimizer:
    kind: gibbs_anneal
    chains: 3
    cooling:
      kind: linear
      beta_start: 0.20
      beta_end: 4.0
    early_stop:
      enabled: false
      patience: 0
      min_delta: 0.0
      require_min_unique: false
      min_unique: 0
      success_min_per_tf_norm: 0.0

  elites:
    k: 10
    select:
      diversity: 0.0              # 0..1
      pool_size: auto             # auto | all | int>=1

  output:
    save_sequences: true
    save_trace: true
    include_tune_in_sequences: false
    live_metrics: true
```

Notes:
- `sequence_length` must be at least the widest PWM after applying `sample.motif_width` bounds.
- `motif_width.maxw` enforces a contiguous max-information trim during sampling only.
- Canonicalization is automatic when `objective.bidirectional=true`.
- MMR uses a hybrid distance: full-sequence Hamming + motif-core weighted Hamming (low-information core positions get higher weight).
- `moves.overrides.*` contains optional expert controls (operator mix + adaptation). Leave unset unless you are actively tuning proposals.
- Default operator mix is `S=0.85, B=0.07, M=0.04, I=0.04, L=0, W=0` (not `P(S)=1`).
- `S` is a Gibbs single-site update (accepted by construction); `B/M/L/W/I` are MH proposals (accept/reject).
- `moves.overrides.gibbs_inertia` dampens late single-site Gibbs flips (`p_stay_*`), which can reduce raw-trajectory jitter.
- `moves.overrides.adaptive_weights.freeze_after_*` and `moves.overrides.proposal_adapt.freeze_after_*` freeze adaptation late so the kernel stops drifting.
- `sample.optimizer.kind` currently supports `gibbs_anneal`.
- `sample.optimizer.chains` controls the number of independently initialized chains.
- `sample.optimizer.cooling.kind` controls the MCMC beta schedule (`fixed`, `linear`, or `piecewise`).
- Cooling keys are kind-specific and fail fast:
  - `fixed`: `beta`
  - `linear`: `beta_start`, `beta_end`
  - `piecewise`: `stages` (strictly increasing `sweeps`)
- When beta changes over sweeps, behavior is simulated annealing. With fixed beta, it is fixed-temperature hybrid MCMC.
- `sample.elites.select.diversity` is the primary quality-vs-diversity knob (`0..1`):
  - `0.0`: disables MMR diversity pressure and uses greedy top-k selection by final optimizer score (`combined_score_final`), with normal uniqueness dedupe.
  - `1.0`: strongest diversity pressure.
  - For `diversity > 0`, Cruncher uses direct weights:
    `score_weight = 1 - diversity`, `diversity_weight = diversity`, plus minimum full/core Hamming constraints derived from `diversity`.
- `sample.elites.select.pool_size` controls the MMR candidate sandbox:
  - `auto`: `min(candidate_count, min(20000, max(4000, 500*k)))`
  - `all`: use every candidate draw
  - integer: clamp to available candidates

### analysis

Curated plot + table suite with explicit contracts (no per-plot enable toggles).

```yaml
analysis:
  enabled: true
  run_selector: latest      # latest | explicit
  runs: []                  # used only if run_selector=explicit
  pairwise: auto            # off | auto | all_pairs_grid | [tf1, tf2]
  plot_format: pdf          # pdf | png
  plot_dpi: 300
  table_format: parquet     # parquet | csv
  archive: false
  max_points: 5000
  trajectory_stride: 5
  trajectory_scatter_scale: llr   # normalized-llr | llr
  trajectory_scatter_retain_elites: true
  trajectory_sweep_y_column: objective_scalar  # objective_scalar | raw_llr_objective | norm_llr_objective
  trajectory_sweep_mode: best_so_far  # best_so_far | raw | all
  trajectory_particle_alpha_min: 0.25
  trajectory_particle_alpha_max: 0.45
  trajectory_chain_overlay: false
  trajectory_summary_overlay: false
  elites_showcase:
    max_panels: 12
  fimo_compare:
    enabled: false
  mmr_sweep:
    enabled: false
    pool_size_values: [auto, all]
    diversity_values: [0.0, 0.25, 0.50, 0.75, 1.0]
```

Notes:
- `analysis.pairwise` controls elite score-space projection:
  - `auto`: explicit TF pair for 2-TF runs; for 3+ TF runs, selects TF-specific axes from elite worst/second-worst rankings and renders those TF raw-LLR axes.
  - `all_pairs_grid`: render all TF pair projections in one figure (edge-only axis labels in the grid).
  - `[tf1, tf2]`: explicit TF pair projection.
- In `elite_score_space_context`, consensus markers are per-TF consensus anchors for the active axes (`<tf> consensus`), not theoretical maxima bounds.
- `analysis.trajectory_scatter_scale` controls elite-context axis scale for explicit TF-pair projection; `analysis.pairwise=auto` with 3+ TFs requires `llr`.
- In `analysis.pairwise=all_pairs_grid`, shared limits are automatic only for `analysis.trajectory_scatter_scale=normalized-llr`.
- `analysis.trajectory_scatter_retain_elites` keeps exact-mapped elite provenance in metadata.
- `analysis.trajectory_sweep_y_column` controls the y-axis for `chain_trajectory_sweep.*`:
  - `objective_scalar`: optimizer scalar objective at each sweep (`min`/`sum` over TF best-window scores, with soft-min shaping when enabled).
  - `raw_llr_objective`: replay objective on raw-LLR per-TF scores.
  - `norm_llr_objective`: replay objective on normalized-LLR per-TF scores.
- `analysis.trajectory_sweep_mode` controls sweep-plot narrative: `best_so_far` (default optimizer narrative), `raw`, or `all` (raw exploration + best-so-far envelope).
- `analysis.trajectory_stride` applies deterministic point decimation for `chain_trajectory_sweep.*`.
  - Scatter always retains first/last sweeps, best-update sweeps, and exact-mapped elite sweeps (when enabled).
  - Sweep `best_so_far|all` always retains improvement sweeps in addition to stride points.
- `analysis.trajectory_summary_overlay=true` adds a median-across-chains line and IQR band in `chain_trajectory_sweep.*`, computed from real per-sweep values (no fitted smoothing).
  - `trajectory_sweep_mode=raw` summarizes raw chain values.
  - `trajectory_sweep_mode=best_so_far|all` summarizes best-so-far chain values.
  - Overlay renders only when at least two chains are present; default is disabled.
- Trajectory plots are chain-centric: scatter backbones follow visited states; best markers highlight record updates without replacing the backbone.
- `analysis.trajectory_chain_overlay=true` adds diagnostic chain markers (scatter: sampled points, sweep: start/end markers).
- Required analysis plots fail fast on plotting/data contract errors (`elite_score_space_context`, `chain_trajectory_sweep`, `elites_nn_distance`, `elites_showcase`); only explicitly optional plots can be skipped by policy.
- `analysis.elites_showcase.max_panels` sets a hard cap for the baserender-backed elites showcase panel count; analyze fails fast when elites exceed this cap.
  - Cruncher-to-baserender handoff is contract-first: Cruncher emits `Record` primitives and baserender renders them.
  - Integration must use baserender public API (`dnadesign.baserender`) only; internal `dnadesign.baserender.src.*` imports are non-contractual.
- `analysis.fimo_compare.enabled=true` adds `optimizer_vs_fimo.*`, a descriptive QA scatter:
  - x: Cruncher joint optimizer score (same scalar used during optimization)
  - y: FIMO weakest-TF score (`min_tf(-log10 p_seq_tf)`), where each TF score is based on the best FIMO hit p-value corrected to sequence-level.
  - no-hit/poor-hit rows are retained with score `0` (not dropped), so the comparison shows full sampled coverage.
- `analysis.mmr_sweep.enabled=true` writes `analysis/tables/table__elites_mmr_sweep.*` by replaying MMR over pool-size and diversity grids.
- `health_panel.*` reports MH acceptance only (Gibbs `S` moves are excluded by design) and shows attempted move mix over sweeps.
- `elites_nn_distance.*` uses the final optimizer scalar score (`combined_score_final`) on the y-axis and summarizes full-sequence diversity as score-vs-distance plus a pairwise distance matrix, while retaining motif-core NN context.
  - In the score-vs-distance panel, x is nearest-neighbor full-sequence Hamming distance (bp) to the closest other selected elite (not average pairwise distance).
- If the `analysis` block is omitted, analyze resolves this section from schema defaults.

### Inspect resolved config

```bash
cruncher config summary -c path/to/workspace/configs/config.yaml
```

### Related docs

- [Sampling + analysis](../guides/sampling_and_analysis.md)
- [Intent + lifecycle](../guides/intent_and_lifecycle.md)
- [CLI reference](cli.md)
- [Architecture and artifacts](architecture.md)
