## cruncher


How Scorer + SequenceEvaluator Fit Into the Gibbs Sampler
	1.	SequenceState ➔ Scoring
The optimizer maintains a candidate sequence as a SequenceState (an integer‐encoded DNA string). At each MCMC step, it proposes a small change (e.g. flipping one base, replacing a block, or flipping multiple sites).
	2.	Evaluator as a Simple Adapter
Rather than calling Scorer directly, the optimizer always invokes evaluator(state). Internally, the SequenceEvaluator does two things:
	•	It asks Scorer for each PWM’s best log-odds (LLR) score on that sequence.
	•	It converts those per-PWM LLRs into one unified “fitness” number (by applying the chosen scale—e.g. “log-p” or “z” or “raw LLR”—and then taking the minimum across all PWMs).
	3.	Scorer’s Responsibilities
	•	Per-PWM null distributions are built once at startup: for each PWM, we precompute a grid of possible LLR sums (lom) and the exact tail‐probabilities (p-values).
	•	When asked to score a sequence, Scorer scans every possible window (forward and reverse complement) for each PWM, finds the single highest LLR, and—if needed—turns it into a p-value or z-score or normalized “log-p.”
	•	The result is that each PWM ends up with one number (e.g. “−log₁₀(p_seq)” or “z-score”) for the current sequence.
	4.	Reduction to a Single Fitness
Once every PWM has its own scaled score, the SequenceEvaluator simply takes the minimum of those numbers. The Gibbs sampler treats that minimum as its target fitness (higher = better). In other words, optimizing means “raise the worst PWM match” until all PWMs appear strongly anywhere in the sequence.
	5.	Optimizer Logic (in cgm.py)
	•	The Gibbs loop proposes a move → calls fitness_old = evaluator(old_state) → applies the change → calls fitness_new = evaluator(new_state) → accepts or rejects based on the Metropolis rule at the current inverse temperature (β).
	•	Because SequenceEvaluator hides all the per-PWM bookkeeping, the optimizer code never needs to know about null distributions, p-values, or even how many PWMs exist. It just asks “What is the fitness of this sequence?” and uses that to guide acceptance.

By cleanly separating “how to turn a DNA string into a single number” (Scorer + Evaluator) from “how to propose and accept/reject moves” (GibbsOptimizer), we keep each component focused, easy to read, and modular.


The score is 0 if the sequence has the same probability of being a functional site and of being a random site. The score is greater than 0 if it is more likely to be a functional site than a random site, and less than 0 if it is more likely to be a random site than a functional site.[1] The sequence score can also be interpreted in a physical framework as the binding energy for that sequence.

The p-value already tells you “how surprising is the best window of this PWM in this sequence compared with random DNA of the same length?”  


A more informative axis: “consensus-normalised logp”
Below is an easy drop-in normalisation that preserves all the good properties of the FIMO p-value but spreads the dynamic range uniformly between 0 and 1:

score
norm
  
=
  
−
log
⁡
10
(
𝑝
seq
)
−
log
⁡
10
(
𝑝
consensus
)
  
,
clipped to 
[
0
,
1
]
.
score 
norm
​
 = 
−log 
10
​
 (p 
consensus
​
 )
−log 
10
​
 (p 
seq
​
 )

 
​
 ,clipped to [0,1].
p_seq = Bonferroni-corrected p-value of the best window in this sequence

p_consensus = Bonferroni-corrected p-value of the PWM’s own consensus embedded in the same L-bp background

interpretation	value
sequence has no convincing hit	0
sequence’s best hit is as good as the consensus	1
halfway in log-space between random and consensus	0.5

Because both numerator and denominator are bona-fide FIMO p-values, the ratio stays monotonic in the raw evidence. You keep the nice cross-PWM comparability and you never suffer the “all zeros” collapse.
	•	Keep using p-values – they are the right way to compare heterogeneous PWMs.
	•	Normalise them by the PWM’s own consensus strength to map everything into 0, 1.
	•	Update the config to scorer_scale: logp_norm; the plot will immediately give you an intuitive “fraction-of-perfect” picture for every sequence in every chain.


**cruncher** is a pipeline that automates the design of short DNA sequences embedding strong matches for all user-supplied transcription-factor PWMs:

1. **Parse**
   Read one or many PWMs (MEME, JASPAR, …) and generate sequence‐logo plots.

2. **Sample**
   Initialize a random DNA sequence and run a **simulated-annealing MCMC** optimizer to discover sequences whose best motif‐match p-values are jointly maximized.

3. **Analyze**
   Reload any past batch, regenerate plots (score distributions, PWM scatter, logo overlays) without re-sampling.

> "Generate short DNA sequences that contain strong matches for all user-supplied TF PWMs, possibly highly overlapping on either strand."

---

#### Quick Start

```bash
# 1. Preview motifs
cruncher parse   cruncher/config.yaml

# 2. Run optimizer
cruncher sample  cruncher/config.yaml

# 3. View diagnostics
open results/batch_<timestamp>/plots/score_kde.png
```

---

#### Core Concept

> “If I draw uniform random DNA of this length, what is the chance I’d see a window scoring ≥ my best motif alignment?”

That tail p-value is exactly what FIMO reports per hit.  We implement it in-house via:

1. **Log-odds scan**
   L<sub>i,b</sub> = log₂(p<sub>i,b</sub>/0.25)
   s = ∑₁ᵂ L<sub>i, x<sub>i</sub></sub>
   (sliding both strands via a Numba inner loop)

2. **Exact null distribution (DP)**

   * Scale L to integers (0.001-bit resolution)
   * Convolve column-wise over bases (iid 0.25 each) → P(score = k)
   * Compute tail P(S ≥ k) once per PWM → `(scores[], tail_p[])` lookup

3. **Fitness**
   For sequence x,
   p<sub>i</sub> = tail\_p<sub>i</sub>(max s<sub>i</sub>)
   **fitness(x) = −log₁₀(min<sub>i</sub> p<sub>i</sub>)**

Maximizing this ensures **all** TF motifs match significantly, on either strand, fairly across varying motif lengths/content.

> **Citation:**
> FIMO-style exact score distribution (Grant *et al.* 2011; Staden 1994) implemented in-house for speed & Numba integration.

---

#### Optimization Logic

We perform a **temperature-controlled Gibbs/Metropolis mixture** (simulated annealing):

| Phase       | β (“inverse temperature”) | Moves            | Goal                         |
| ----------- | ------------------------- | ---------------- | ---------------------------- |
| **Explore** | 0 → 0.1                   | B-locks, M-multi | Traverse basins, avoid traps |
| **Refine**  | 0.1 → 1.0                 | S/B/M blend      | Focus on genuine peaks       |
| **Freeze**  | ≥ 1.0 (up to 2–3)         | S-single flips   | Polish top hits              |

β follows your config (`sample.optimiser.gibbs.cooling`): either **fixed** or **piece-wise linear**.

#### Move Catalogue

| Code    | Type                           | Step size                          | Intuition                   |
| ------- | ------------------------------ | ---------------------------------- | --------------------------- |
| **S**   | Single‐nucleotide Gibbs flip   | 1 bp                               | Fine-tune; very high accept |
| **B**   | Contiguous block replacement   | random length ∈ `[min,max]` bp     | Cross shallow minima        |
| **M**   | K disjoint flips (multi‐Gibbs) | K ∈ `[kmin,kmax]`, scattered sites | Mix distant regions         |
| (SL/SW) | Slide / Swap windows (MH)      | reserved for future enhancement    | —                           |

Move probabilities adapt with β (see `_sample_move_kind()`), shifting weight from **coarse** (B/M) to **precise** (S) as temperature cools.

#### One Sweep

1. **Determine β** from schedule.
2. **Select move** type by β-dependent weights.
3. **Propose** a fragment change.
4. **Accept** with probability ∝ exp(β·Δfitness) (Gibbs or Metropolis).
5. **Record** diagnostics; after burn-in, save each full sequence as a draw.

---

#### Project Layout

```
dnadesign/
└─ cruncher/
   ├─ config.yaml      # Runtime configuration settings
   ├─ main.py          # CLI entry point
   ├─ parse/           # PWM parsers, model, DP p-value lookup
   ├─ sample/          # SequenceState, scoring, optimisers, plots
   ├─ utils/           # Config loader, trace persistence
   └─ results/         # Auto-generated batch subfolders
```

---

#### Minimal Configuration (config.yaml)

```yaml
cruncher:
  mode: sample
  out_dir: results/
  regulator_sets:
    - [cpxR, soxR]
  motif:
    formats:
      .txt: MEME
      .pfm: JASPAR
    plot:
      logo: true
      bits_mode: information
      dpi: 200
  sample:
    bidirectional: true
    init:
      length: 30
    optimiser:
      kind: gibbs
      gibbs:
        draws: 400
        tune: 100
        chains: 4
        cores: 4
        min_dist: 1
        cooling:
          kind: piecewise
          stages:
            - {sweeps: 0,   beta: 0.01}
            - {sweeps: 200, beta: 0.10}
            - {sweeps: 500, beta: 1.00}
        moves:
          block_len_range: [3, 15]
          multi_k_range:   [2, 6]
        top_k: 200
    plots:
      trace:       true
      autocorr:    true
      convergence: true
      scatter_pwm: true
  analysis:
    runs: []
    plots:
      - score_kde
      - scatter_pwm
      - logo_elites
```

---

#### Usage Summary

| Task        | Command                                                 |
| ----------- | ------------------------------------------------------- |
| **Parse**   | `cruncher parse configs/example.yaml`                   |
| **Sample**  | `cruncher sample configs/example.yaml`                  |
| **Analyze** | `cruncher analyse configs/example.yaml [--run <batch>]` |

Results appear under `results/batch_<timestamp>/`, including:

* `config_used.yaml` (frozen settings)
* `hits.csv` (ranked sequences + per-PWM scores)
* `trace.nc` (ArviZ MCMC trace)
* `plots/` (trace, autocorr, scatter, logos…)
* `README.txt` (run metadata)
