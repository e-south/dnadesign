## cruncher

**cruncher** is a pipeline that automates the design of short DNA sequences embedding strong matches for all user-supplied transcription-factor PWMs:

1. **Parse**
   Read cached PWMs (or build them from cached TFBS) and generate sequence-logo plots.

2. **Sample**
   Initialize a DNA sequence and run a **simulated-annealing MCMC** optimizer to discover sequences whose best motif matches are jointly maximized.

3. **Analyze**
   Reload any past batch, regenerate plots (score distributions, PWM scatter) without re-sampling.

> “Generate short DNA sequences that contain statistically significant sites for every requested PWM, possibly with overlapping motifs on either strand.”

---

### Quick Start

Lockfiles are mandatory for parse/sample/analyze/report (no implicit TF resolution).

```bash
# 1. Populate catalog (RegulonDB)
cruncher fetch sites  --tf lexA --tf cpxR src/dnadesign/cruncher/config.yaml
cruncher fetch motifs --tf lexA --dry-run src/dnadesign/cruncher/config.yaml
cruncher catalog list src/dnadesign/cruncher/config.yaml

# Offline verification (no network)
cruncher fetch motifs --tf lexA --offline src/dnadesign/cruncher/config.yaml

# 2. Lock TF names
cruncher lock src/dnadesign/cruncher/config.yaml

# 3. Check targets + preview motifs
cruncher targets status src/dnadesign/cruncher/config.yaml
cruncher targets stats src/dnadesign/cruncher/config.yaml
cruncher targets candidates src/dnadesign/cruncher/config.yaml
cruncher catalog search src/dnadesign/cruncher/config.yaml lex --fuzzy
cruncher parse src/dnadesign/cruncher/config.yaml

# 4. Run optimizer
cruncher sample  src/dnadesign/cruncher/config.yaml

# 5. Analyze + report
cruncher analyze src/dnadesign/cruncher/config.yaml
cruncher report  src/dnadesign/cruncher/config.yaml sample_<tfset>_<timestamp>

# 6. Inspect past runs
cruncher runs list src/dnadesign/cruncher/config.yaml
cruncher runs watch src/dnadesign/cruncher/config.yaml <run_name>
# If the run index is missing or stale:
cruncher runs rebuild-index src/dnadesign/cruncher/config.yaml
```

Source-specific details (RegulonDB TLS, HT hydration, windowing rules) live in
`docs/ingestion.md` and `docs/troubleshooting.md`. The short version:

- `motif_store.pwm_source: sites` builds PWMs from cached binding sites.
- Coordinate-only HT data is hydrated via NCBI (default) or `--genome-fasta`.
- Variable-length sites require `motif_store.site_window_lengths`.
- If NetCDF backends are missing, set `sample.save_trace: false` (analyze/report require trace.nc).

To inspect available optimizers:

```bash
cruncher optimizers list
```

To inspect config settings before sampling:

```bash
cruncher config summary cruncher/config.yaml
```

To verify cache integrity (no missing motif/site files):

```bash
cruncher cache verify cruncher/config.yaml
```

---

### Developer docs

- `docs/spec.md` — end-to-end architecture + requirements
- `docs/architecture.md` — component boundaries and run artifacts
- `docs/config.md` — config schema and examples
- `docs/cli.md` — full CLI command list
- `docs/demo.md` — end-to-end RegulonDB workflow (LexA + CpxR)

### Project Layout

```
dnadesign/
└─ cruncher/
   ├─ README.md         # Overview and usage
   ├─ config.yaml       # Runtime settings
   ├─ cli/              # Typer CLI entry point
   ├─ core/             # PWM/scoring/state/optimizers
   ├─ ingest/           # Source adapters + normalization
   ├─ store/            # catalog cache access + lockfiles
   ├─ services/         # fetch/lock/catalog/targets services
   ├─ workflows/        # parse/sample/analyze/report orchestration
   ├─ io/               # parsers + plots
   ├─ config/           # v2 config schema + loader
   ├─ results/          # Generated batch folders (CSV, plots, trace files)
   └─ tests/            # Unit tests for each component (to be rebuilt)
```

---

#### Core Concepts

#### 1. PWM-Based Scoring (How We Measure Sequence “Goodness”)

* **PWM (Position Weight Matrix)**
  A PWM encodes a transcription factor’s preferred DNA motif as a matrix of nucleotide probabilities.
* **Sliding Window**
  To score a candidate sequence, slide a window equal to the PWM’s width along both strands. At each position, compute the log-likelihood-ratio (LLR) comparing “this window matches the PWM” vs. “this window is random DNA.”
* **Null Distribution & p-Value**
  Before sampling, build, for each PWM, the distribution of LLR scores one would see on purely random DNA of the same length. When the sliding window finds the best LLR in our sequence, we convert that LLR into a p-value (or z-score) via the null distribution. In plain terms: “If DNA were random, how surprising is this match?”
* **Combining Multiple PWMs**
  If using N PWMs, each sequence obtains N p-values (one per PWM). We then take the worst (largest) p-value as the single “fitness” measure. This forces the sequence to contain a strong match for every PWM, not just one.

#### 2. MCMC Sampling with Markov Chains (How We Search for High-Scoring Sequences)

* **Seed Sequence**
  Begin with an initial DNA string—either purely random or seeded by embedding a PWM’s consensus motif.
* **Parallel Markov Chains**
  Run multiple chains in parallel. Each chain holds its own DNA sequence.
* **Proposing Moves**
  At each iteration, a chain proposes a small edit:

  * **Single-base flip:** change one nucleotide.
  * **Contiguous block replacement:** select a random segment and rewrite it.
  * **Multi-site flip:** flip several positions at once.
* **Acceptance via Metropolis Criterion**
  After editing, recompute the sequence’s fitness by rescanning every PWM (sliding window + null-distribution lookup). Compare old vs. new fitness using

  $$
    \text{accept probability} = \min\bigl(1,\;e^{\beta\,(f_{\text{new}}-f_{\text{old}})}\bigr),
  $$

  where β (inverse temperature) starts small (exploration) and increases over time (exploitation).
* **Tune vs. Draw Phases**

  * **Tune (burn-in):** chains explore broadly without saving to output.
  * **Draw (sampling):** every accepted sequence is recorded, along with its per-PWM p-values and overall fitness.
  * **record_tune:** when false, burn-in states are not stored in `sequences.parquet`.
* **Output Files**

  * **config\_used.yaml:** exact runtime settings plus each PWM’s matrix and consensus. Includes `active_regulator_set`.
  * **trace.nc** (if enabled): an ArviZ-format record of every sampled fitness.
  * **sequences.parquet:** one row per saved draw, with columns
    `chain, iteration, phase (tune/draw), sequence, score_<TF1>, score_<TF2>, …`.
  * **cruncher_elites_*/<name>.parquet:** machine-friendly elites table (plus a JSON copy for readability).
  * **cruncher_elites_*/<name>.json:** the top K sequences by fitness across all chains. Each elite entry includes:

    * rank, chain, iteration, and DNA string
    * for each PWM: raw LLR, best match position (offset), strand, a simple “motif\_diagram” (e.g. `15_[+1]_8`), and the scaled score.
  * **run_manifest.json:** resolved TFs, hashes, and optimizer stats for reproducibility.
  * **report.json / report.md:** generated by `cruncher report`.

If multiple `regulator_sets` are configured, Cruncher runs **one parse/sample/analyze per set** and
creates separate run directories (e.g., `sample_set2_lexA-oxyR_20250101_120000`).

---

#### Example Configuration (`config.yaml`)
```yaml
# dnadesign/cruncher/config.yaml
cruncher:
  # GLOBAL SETTINGS
  out_dir: results/                         # relative path under cruncher/ where batches go
  regulator_sets:                           # each entry is a list of TF names (each set is sampled separately)
    - [cpxR, soxR]

  # PARSE MODE (draw logos, print log‐odds from cached PWMs)
  parse:
    plot:
      logo: true                            # whether to generate a logo (PNG) per PWM
      bits_mode: information                # “information” (bits) vs “probability” mode
      dpi: 200                              # resolution for output PNG

  # SAMPLE MODE (MCMC‐based sequence search)
  sample:
    bidirectional: true                     # scan both strands (forward + reverse)
    seed: 42                                # RNG seed for reproducibility
    record_tune: false                      # whether to store burn-in states

    init:
      kind: random                          # “random” | “consensus” | “consensus_mix”
      length: 30                            # overall length of the output sequence (must be ≥ 1)
      pad_with: background                  # “background” (uniform-random pad) or “A”|“C”|“G”|“T”
      regulator: soxR                       # If kind == “consensus”, supply a regulator name within the active set

    draws: 20000                            # number of MCMC draws (after tune)
    tune: 10000                             # number of burn‐in sweeps
    chains: 4                               # number of parallel chains (Gibbs or PT)
    min_dist: 1                             # Hamming‐distance threshold for “diverse elites”
    top_k: 10                               # how many top sequences to save

    # Move‐kernel parameters
    moves:
      block_len_range: [2, 6]               # contiguous block proposals ∈ [5,25] bp
      multi_k_range:   [2, 6]               # number of disjoint flips ∈ [2,8] sites
      slide_max_shift: 4                    # maximum shift for “slide” moves (reserved)
      swap_len_range:  [2, 8]               # length of blocks to swap ∈ [8,20] (reserved)

      move_probs:
        S: 0.80                             # probability of a single‐base‐flip move
        B: 0.10                             # probability of a contiguous block replacement
        M: 0.10                             # probability of a multi‐site flip (k sites)

    optimiser:
      kind: gibbs                           # “gibbs” | “pt”
      scorer_scale: llr                     # “llr" | "z" | “logp” | "consensus-neglop-sum"

      # GIBBS (LINEAR‐RAMP COOLING) BUILT‐IN
      cooling:
        kind: linear                        # "fixed" | “linear” | “geometric” (geometric is for PT only)
        beta: [0.0001, 0.001]               # [β_start, β_end]

      # replica exchange MCMC sampling
      # If kind == “pt”, uncomment & use the block below instead:
      # cooling:
      #   kind: geometric
      #   beta: [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 6.0]

      swap_prob: 0.10                       # intra‐chain block‐swap probability (Gibbs); inter‐chain exchange prob (PT)

    save_sequences: true                    # whether to write sequences.parquet for downstream analysis

  # ANALYSIS MODE
  analysis:
    runs:                                   # list of batch names to re‐analyse
      - sample_cpxR-soxR_20250603
    plots:
      trace:       true                     # plot MCMC trace
      autocorr:    true                     # plot autocorrelation
      convergence: true                     # convergence diagnostics
      scatter_pwm: true                     # PWM‐score scatter (requires gathered_per_pwm.csv)
    scatter_scale: llr
    subsampling_epsilon: 10.0              # minimum per-TF distance to keep a draw

  motif_store:
    catalog_root: .cruncher
    source_preference: [regulondb]
    allow_ambiguous: false
    pwm_source: matrix                        # matrix | sites
    min_sites_for_pwm: 2

  ingest:
    genome_source: ncbi
    genome_fasta: null
    genome_cache: .cruncher/genomes
    genome_assembly: null
    contig_aliases: {}
    ncbi_email: null
    ncbi_tool: cruncher
    ncbi_api_key: null
    ncbi_timeout_seconds: 30
    regulondb:
      base_url: https://regulondb.ccg.unam.mx/graphql
      verify_ssl: true
      ca_bundle: null                        # optional CA bundle (loaded alongside certifi)
      timeout_seconds: 30
      motif_matrix_source: alignment         # alignment | sites
      alignment_matrix_semantics: probabilities
      curated_sites: true
      ht_sites: false
      ht_dataset_sources: null
      ht_dataset_type: TFBINDING
      uppercase_binding_site_only: true
```
