## cruncher

**cruncher** is a pipeline that automates the design of short DNA sequences embedding strong matches for all user-supplied transcription-factor PWMs:

1. **Parse**
   Read one or many PWMs (MEME, JASPAR, …) and generate sequence-logo plots.

2. **Sample**
   Initialize a DNA sequence and run a **simulated-annealing MCMC** optimizer to discover sequences whose best motif matches are jointly maximized.

3. **Analyze**
   Reload any past batch, regenerate plots (score distributions, PWM scatter) without re-sampling.

> “Generate short DNA sequences that contain statistically significant sites for every requested PWM, possibly with overlapping motifs on either strand.”

---

### Quick Start

```bash
# 1. Preview motifs
cruncher parse   cruncher/config.yaml

# 2. Run optimizer
cruncher sample  cruncher/config.yaml

# 3. View diagnostics
open results/batch_<timestamp>/plots/score_kde.png
```

---

### Project Layout

```
dnadesign/
└─ cruncher/
   ├─ README.md         # Overview and usage
   ├─ config.yaml       # Runtime settings
   ├─ main.py           # CLI entry point (“parse”, “sample”, “analyse”)
   ├─ parse/            # PWM parsing and null-distribution setup
   ├─ sample/           # SequenceState, Scorer/Evaluator, and optimizers
   ├─ analyse/          # Post-sampling transforms and plotting
   ├─ utils/            # Shared helpers (config loading, trace I/O, etc.)
   ├─ results/          # Generated batch folders (CSV, plots, trace files)
   └─ tests/            # Unit tests for each component
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
* **Output Files**

  * **config\_used.yaml:** exact runtime settings plus each PWM’s matrix and consensus.
  * **trace.nc** (if enabled): an ArviZ-format record of every sampled fitness.
  * **sequences.csv:** one row per saved draw, with columns
    `chain, iteration, phase (tune/draw), sequence, score_<TF1>, score_<TF2>, …`.
  * **elites.json:** the top K sequences by fitness across all chains. Each elite entry includes:

    * rank, chain, iteration, and DNA string
    * for each PWM: raw LLR, best match position (offset), strand, a simple “motif\_diagram” (e.g. `15_[+1]_8`), and the scaled score.

---

#### Example Configuration (`config.yaml`)
```yaml
# dnadesign/cruncher/config.yaml
cruncher:
  # GLOBAL SETTINGS
  mode: sample-analyse                      # “parse” | “sample” | “analyse” | “sample-analyse”
  out_dir: results/                         # relative path under cruncher/ where batches go
  regulator_sets:                           # each entry is a list of TF names
    - [cpxR, soxR]

  # PARSE MODE (sanity‐check PWMs, draw logos, print log‐odds)
  parse:
    formats:                                # map file‐extension → parser name (MEME, JASPAR, …)
      .txt: MEME
      .pfm: JASPAR
    plot:
      logo: true                            # whether to generate a logo (PNG) per PWM
      bits_mode: information                # “information” (bits) vs “probability” mode
      dpi: 200                              # resolution for output PNG

  # SAMPLE MODE (MCMC‐based sequence search)
  sample:
    bidirectional: true                     # scan both strands (forward + reverse)

    init:
      kind: random                          # “random” | “consensus” | “consensus_mix”
      length: 30                            # overall length of the output sequence (must be ≥ 1)
      pad_with: background                  # “background” (uniform-random pad) or “A”|“C”|“G”|“T”
      regulator: soxR                       # If kind == “consensus”, supply a regulator name that exists in regulator_sets

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
      # softmax_beta: 0.20                  # only used by PT (must be a positive float)

    save_sequences: true                    # whether to write sequences.csv for downstream analysis

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
    gather_nth_iteration_for_scaling: 10    # how many draws to skip between per‐PWM scoring
```

e-south



###### Notes

not all PWMs are created equal in terms of realtive information, how can this be controlled for?
hamming constraints to isolate multiple "diverse" hits
