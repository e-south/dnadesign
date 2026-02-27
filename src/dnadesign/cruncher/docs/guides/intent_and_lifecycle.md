## Cruncher intent and lifecycle

**Last updated by:** cruncher-maintainers on 2026-02-23

### Contents
- [Intent](#intent)
- [Intended use cases](#intended-use-cases)
- [Non-goals](#non-goals)
- [Core behavior](#core-behavior)
- [Math and scoring](#math-and-scoring)
- [Per-sweep optimizer mechanics](#per-sweep-optimizer-mechanics)
- [Why raw trajectories look noisy](#why-raw-trajectories-look-noisy)
- [Diversity and elites](#diversity-and-elites)
- [Run lifecycle](#run-lifecycle)
- [Artifacts and outputs](#artifacts-and-outputs)
- [Config mapping](#config-mapping)
- [Architecture mapping](#architecture-mapping)
- [Related docs](#related-docs)

### Intent

Cruncher is an **artifact-first, reproducible DNA sequence optimizer** for
**fixed-length** designs under **multi-transcription-factor PWM objectives**.
It produces a **small, diverse elite set** of candidate sequences while
preserving strict operational contracts. The system favors:

- deterministic, auditable runs
- strict input validation (no silent fallbacks)
- reproducible artifacts and analysis

**Practical mental model:** deterministic data prep + strict optimization +
artifact-native analytics.

### Intended use cases

- Multi-TF promoter/operator design under tight length constraints.
- Tradeoff exploration when multiple TF motifs must co-exist in a constrained length.
- Producing a small, diverse, high-quality candidate set for downstream assays.
- Workspace-scoped studies for sweeps, then aggregate comparison.

### Non-goals

- **Posterior inference:** Gibbs annealing MCMC here is used as an optimization engine + diagnostics, not Bayesian inference.
- **Variable-length design:** sequences are fixed length by contract.
- **Motif discovery as the primary workflow:** discovery is supported (MEME/STREME integration), but Cruncher's core is sequence design under pinned PWMs.

### Core behavior

- **Fixed length** is a hard invariant: every candidate is exactly
  `sample.sequence_length`, and must be at least the widest PWM length.
- **Gibbs annealing optimization**: chain-based MCMC explores sequence space under an explicit cooling schedule.
- **Best-hit per TF**: each TF is scored by the best-scoring window in the
  sequence; bidirectional scans include both strands.
- **Optimization objective**: the default objective prioritizes the weakest TF
  (`objective.combine=min`) so all TFs must improve together.
- **Artifact-only analysis**: analysis runs do not rescan sequences; they read
  sample artifacts and hit metadata only.

### Math and scoring

Cruncher uses a FIMO-like scanning model without calling FIMO:

1. Convert each PWM to log-odds against a 0-order background.
   For `score_scale=logp`, Cruncher builds an exact best-window tail lookup via DP convolution under the same background.
2. Scan all windows of width `w` in each sequence.
3. Pick the best window per TF (deterministic tie-break).

If `objective.bidirectional=true`, the scan considers both strands. When
`score_scale=logp`, the best-window tail probability is converted to a
sequence-level p-value:

```
p_seq = 1 - (1 - p_win)^n_windows
```

where `n_windows = L - w + 1`, and `n_windows` counts both strands under
bidirectional scanning.

Normalized scores use the PWM's theoretical max to map best-hit values into a
0-1 scale (`normalized-llr`), enabling consistent thresholds across TFs.

Per-TF scores are combined into a single optimization objective using
`objective.combine` (`min` or `sum`). Optional soft-min shaping under
`objective.softmin` uses:

```
softmin(v; beta) = -(1 / beta) * log(sum_i exp(-beta * v_i))
```

where larger `beta` makes the objective closer to hard `min`.

### Per-sweep optimizer mechanics

For each chain, Cruncher maintains a discrete sequence state:

```
x in {A, C, G, T}^L
```

with objective `f(x)`. At each sweep, the optimizer uses the configured inverse
temperature `beta_mcmc` from `sample.optimizer.cooling`.

Move behavior:

1. `S` (single-site Gibbs): sample a base from a conditional distribution
   proportional to `exp(beta_mcmc * f(.))` at that position. This move is
   accepted by construction.
2. `B/M/L/W/I` (block/multi/local rewrite variants): propose `x'` and accept
   with Metropolis probability:

```
alpha = min(1, exp(beta_mcmc * (f(x') - f(x))))
```

This is why cooling and move mix jointly determine stability: temperature
controls downhill tolerance, and move scale controls proposal jump size.

### Why raw trajectories look noisy

Raw chain paths are expected to jitter even when optimization is healthy:

- Gibbs single-site moves always accept and can flip among near-tie bases.
- The objective is rugged (`max` over windows, often `min` across TFs), so
  small sequence edits can cause abrupt score changes.
- Non-zero tail temperature still permits some downhill MH moves.
- If adaptation is active, proposal behavior can drift during a run.

For optimizer storytelling, interpret trajectory plots together with tail
acceptance and elite outcomes. A monotone "best-so-far" overlay is often more
informative than raw per-step score alone.

### Diversity and elites

Elites are selected with **TFBS-core MMR**:

- For each TF, Cruncher extracts the best-hit core sequence oriented to the PWM.
- Distances are computed **per TF**, then averaged across TFs.
- Weights are **tolerant**: low-information PWM positions contribute more to
  distance, preserving consensus-critical bases while encouraging diversity in
  flexible positions.

When bidirectional scoring is enabled, Cruncher uses **standard identity**
(lexicographic min of sequence and reverse complement) for uniqueness, elite
dedupe, and success counters.

### Run lifecycle

1. **fetch** -> cache motif/site records in the local catalog
2. **lock** -> resolve TF names to exact records + hashes (reproducibility pin)
3. **parse** -> validate/standardize locked PWMs into parse artifacts
4. **sample** -> run multi-chain Gibbs annealing and persist optimize artifacts
5. **analyze** -> replay from artifacts only and write diagnostics/reports/plots

The demo workflows in `docs/demos/` follow this lifecycle end-to-end.

### Artifacts and outputs

Sampling writes (under `optimize/` unless noted):

- `optimize/tables/sequences.parquet` (per-draw scores + metadata)
- `optimize/tables/elites.parquet` (elite sequences + summaries)
- `optimize/tables/elites_hits.parquet` (per-elite x per-TF best-hit/core metadata)
- `optimize/tables/random_baseline.parquet` (baseline score cloud; default on via `sample.output.save_random_baseline=true` and `sample.output.random_baseline_n=10000`)
- `optimize/tables/random_baseline_hits.parquet` (baseline best-hit/core metadata under the same baseline contract)
- `provenance/lockfile.json` (snapshot of the pinned lockfile for reproducibility)

Analysis writes:

- `analysis/reports/summary.json`
- `analysis/reports/report.md` and `analysis/reports/report.json`
- `analysis/manifests/plot_manifest.json` and `analysis/manifests/table_manifest.json`
- `analysis/tables/table__*` (curated tabular artifacts)
- `plots/*` (curated figures)

### Config mapping

Each config block maps directly to a lifecycle phase or runtime contract:

- `workspace` -> run layout and regulator set expansion
- `catalog` -> which cached sources are eligible
- `ingest` / `discover` -> how inputs are sourced or derived
- `sample` -> optimization objective, optimizer/cooling settings, moves, elites
- `analysis` -> report generation and curated plot settings

See the full key reference at `docs/reference/config.md`.

Crosswalk (behavior -> config -> modules -> artifacts):

| Behavior / phase | Config keys (v3) | Primary layers | Writes |
|---|---|---|---|
| Fetch motifs/sites (cache) | `catalog.*`, `ingest.*`, `discover.*` | `ingest/`, `store/` | `<catalog.root>/normalized/...` + `catalog.json` |
| Pin exact inputs (lock) | `workspace.regulator_sets`, `catalog.*` | `store/`, `app/` | `<workspace>/.cruncher/locks/<config>.lock.json` |
| Validate locked PWMs (parse) | (lockfile-driven) | `app/` | `<workspace>/.cruncher/parse/provenance/` |
| Gibbs annealing optimization (sample) | `sample.*` | `core/`, `app/` | `<run_dir>/optimize/` + manifest/status updates |
| Elite filter + TFBS-core MMR | `sample.elites.*` | `core/`, `app/` | `optimize/elites*.parquet` |
| Artifact-only reporting (analyze) | `analysis.*` | `analysis/`, `app/` | `<run_dir>/analysis/` + `<run_dir>/plots/` |
| Study sweep + summarize | `configs/studies/*.study.yaml` | `study/`, `app/` | `outputs/studies/<study_name>/<study_id>/tables` + `outputs/plots/study__<study_name>__<study_id>__plot__*` |

### Architecture mapping

Cruncher is organized by responsibility:

- `core/` -> scoring, move operators, optimizers (pure compute)
- `ingest/` -> source adapters and normalization
- `store/` -> catalog, locks, and run index
- `app/` -> orchestration of fetch/lock/parse/sample/analyze
- `analysis/` -> artifact-only plotting and reporting
- `cli/` -> UX and command wiring

For the full module layout and on-disk schema, see
`docs/reference/architecture.md`.

### Related docs

- [Sampling + analysis](sampling_and_analysis.md)
- [Config reference](../reference/config.md)
- [Architecture and artifacts](../reference/architecture.md)
- [Two-TF demo (end-to-end)](../demos/demo_pairwise.md)
