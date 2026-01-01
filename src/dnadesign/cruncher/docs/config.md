# Cruncher config (v2)

`cruncher.mode` is removed. Commands define the action.

Key blocks:

- `motif_store`: local catalog/cache settings
- `ingest`: source adapters and RegulonDB options
- `parse`: plots for cached PWMs
- `sample`: optimizer settings
- `analysis`: diagnostics for existing runs

Lockfiles: `.cruncher/locks/<config>.lock.json` (required for parse/sample/analyze/report).

Regulator sets: each entry in `regulator_sets` is sampled independently (separate run folders). Run outputs include
`active_regulator_set` in `config_used.yaml` and `regulator_set` in `run_manifest.json`.

## Catalog settings

```
motif_store:
  catalog_root: .cruncher
  source_preference: [regulondb]
  allow_ambiguous: false
  pwm_source: matrix   # matrix | sites
  site_kinds: null     # optional filter: ["curated"], ["ht_tfbinding"], ["ht_peak"]
  combine_sites: false # true combines multiple site sets for a TF (opt-in only)
  dataset_preference: [] # optional dataset IDs (first match wins)
  dataset_map: {}        # optional TF -> dataset_id mapping
  site_window_lengths: {}  # map TF name or dataset:<id> -> fixed window length (bp) for HT sites
  site_window_center: midpoint # midpoint | summit (summit requires summit metadata)
  min_sites_for_pwm: 2
  allow_low_sites: false
```

Notes:

- `pwm_source=matrix` uses stored motif matrices (default).
- `pwm_source=sites` builds PWMs from cached binding-site sequences at runtime.
- `site_kinds` narrows which site sets are eligible when `pwm_source=sites` (useful when curated + HT are cached).
- `combine_sites=true` concatenates site sets for a TF before PWM creation (explicit opt-in; default is no combining).
- `dataset_preference` lets you prefer specific HT datasets when multiple exist for a TF.
- `dataset_map` locks a TF to a specific HT dataset ID (stronger than preference; avoids ambiguity).
- `min_sites_for_pwm` sets the minimum number of binding-site sequences required to build a PWM.
- `allow_low_sites=false` enforces `min_sites_for_pwm` as a hard minimum (sampling will fail if below).
- Lockfile resolution filters candidates based on `pwm_source` (matrix vs sites).
- `catalog_root` is project-local; Cruncher resolves it relative to the config file.
- `cruncher targets status <config>` reports readiness based on these settings.
- `cruncher config summary <config>` prints the effective sampling settings before you run `sample`.
- `site_window_lengths` has **no default**; if binding-site lengths vary, set a per‑TF or per‑dataset window length.
- `site_window_center=midpoint` slices windows symmetrically; `summit` is reserved for sources that expose summits.

## RegulonDB adapter configuration

```
ingest:
  genome_source: ncbi              # ncbi | fasta | none
  genome_fasta: null               # optional FASTA for hydrating coordinate-only sites
  genome_cache: .cruncher/genomes  # cache for downloaded genomes
  genome_assembly: null            # optional assembly/contig accession for local FASTA validation
  contig_aliases: {}               # optional mapping for contig aliases (e.g., chr -> U00096.3)
  ncbi_email: null                 # recommended for NCBI E-utilities
  ncbi_tool: cruncher              # NCBI tool identifier
  ncbi_api_key: null               # optional NCBI API key
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
    ca_bundle: null                # optional CA bundle (loaded alongside certifi)
    timeout_seconds: 30
    motif_matrix_source: alignment   # alignment | sites
    alignment_matrix_semantics: probabilities  # probabilities | counts
    min_sites_for_pwm: 2
    allow_low_sites: false
    curated_sites: true
    ht_sites: false
    ht_dataset_sources: null         # list of HT sources to scan, or null for all
    ht_dataset_type: TFBINDING
    ht_binding_mode: tfbinding       # tfbinding | peaks
    uppercase_binding_site_only: true
```

Notes:

- `motif_matrix_source=alignment` is the default and **fails** if no alignment payload is available.
- `motif_matrix_source=sites` computes a PWM from curated binding-site sequences and requires equal-length sites.
- `min_sites_for_pwm` and `allow_low_sites` control PWM creation during ingestion (independent of motif_store).
- `ca_bundle` adds an extra CA bundle for SSL verification (certifi remains the default trust store).
  As of January 1, 2026, RegulonDB sometimes omits the intermediate certificate; use `ca_bundle`
  to supply it without disabling verification.
- `ht_sites=true` enables HT dataset retrieval (`TFBINDING` and/or peaks depending on `ht_binding_mode`).
- `ht_binding_mode` is explicit: choose `tfbinding` for curated sites or `peaks` for peak-only HT data.
- `genome_source=ncbi` uses NCBI E-utilities to fetch RefSeq/GenBank FASTA by accession (default).
- `genome_source=fasta` uses `ingest.genome_fasta` and (optionally) `ingest.genome_assembly` for validation.
- `genome_cache` is project-local; downloads are cached by accession and reused across runs.
- `contig_aliases` lets you map adapter contig labels to FASTA contig names explicitly.
- `ingest.genome_fasta` and `genome_cache` paths are resolved relative to the config file.
- `ncbi_email` is recommended for polite use of NCBI E-utilities (and helps with throttling).
- `ncbi_api_key` (optional) raises NCBI request limits for large HT datasets.

## Sampling settings

Two additional runtime controls are provided for reproducibility and performance:

```
sample:
  seed: 42          # deterministic RNG seed
  record_tune: false  # store burn-in states in sequences.parquet
  progress_bar: true  # show progress bars
  progress_every: 1000 # log progress summary every N steps
  save_trace: true    # write trace.nc (requires netCDF backend)
  save_sequences: true  # required for analyze/report
  pwm_sum_threshold: 0.0 # filter elites by sum of per-TF scaled scores
  include_consensus_in_elites: false # add PWM consensus to elites metadata
```

Notes:

- `seed` drives all RNG used by MCMC initialisation and move proposals.
- `record_tune=false` reduces memory and output size by skipping burn-in states.
- `save_sequences=true` is required for `analyze` and `report` (no fallback).
- `progress_every=0` disables periodic progress logging.
- `save_trace=false` skips NetCDF output (analyze/report require trace.nc).
- If `init.kind=consensus`, `init.regulator` must be a TF in the active regulator set.

## Optimizer settings

```
optimiser:
  kind: gibbs
  scorer_scale: llr
  cooling:
    kind: linear
    beta: [0.0001, 0.001]
  swap_prob: 0.10
```

Notes:

- `swap_prob` controls swap attempts for the PT optimizer (ignored by single-chain Gibbs).

## Analysis settings

```
analysis:
  plots:
    trace: true
    autocorr: true
    convergence: true
    scatter_pwm: true
  scatter_scale: llr
  scatter_style: edges # edges | thresholds
  subsampling_epsilon: 10.0
```

Notes:

- `scatter_style=edges` draws pairwise KDE contours; `thresholds` draws percentile cutoffs.

## Scoring scales

`optimiser.scorer_scale` and `analysis.scatter_scale` must be one of:

```
llr | z | logp | consensus-neglop-sum
```
