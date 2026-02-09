# Cruncher intent and lifecycle

## Contents
- [Intent](#intent)
- [Intended use cases](#intended-use-cases)
- [Non-goals](#non-goals)
- [Core behavior](#core-behavior)
- [Math and scoring](#math-and-scoring)
- [Diversity and elites](#diversity-and-elites)
- [Run lifecycle](#run-lifecycle)
- [Artifacts and outputs](#artifacts-and-outputs)
- [Config mapping](#config-mapping)
- [Architecture mapping](#architecture-mapping)
- [Related docs](#related-docs)

## Intent

Cruncher designs **short, fixed-length DNA sequences** that jointly satisfy one or
more transcription factor PWMs and returns a **diverse elite set**. The system is
an optimization engine, not a posterior inference engine. It favors:

- deterministic, auditable runs
- strict input validation (no silent fallbacks)
- reproducible artifacts and analysis

**Practical mental model:** deterministic data prep + strict optimization + artifact-native analytics.

## Intended use cases

- Multi-TF promoter/operator design under tight length constraints.
- Tradeoff exploration when multiple TF motifs must co-exist in a constrained length.
- Producing a small, diverse, high-quality candidate set for downstream assays.
- Campaign sweeps across many regulator sets, then aggregate comparison.

## Non-goals

- **Posterior inference:** PT/MCMC here is used as an optimization engine + diagnostics, not Bayesian inference.
- **Variable-length design:** sequences are fixed length by contract.
- **Motif discovery as the primary workflow:** discovery is supported (MEME/STREME integration), but Cruncher's core is sequence design under pinned PWMs.

## Core behavior

- **Fixed length** is a hard invariant: every candidate is exactly
  `sample.sequence_length`, and must be at least the widest PWM length.
- **PT-only optimization**: parallel tempering MCMC explores sequence space.
- **Best-hit per TF**: each TF is scored by the best-scoring window in the
  sequence; bidirectional scans include both strands.
- **Optimization objective**: the default objective prioritizes the weakest TF
  (`objective.combine=min`) so all TFs must improve together.
- **Artifact-only analysis**: analysis runs do not rescan sequences; they read
  sample artifacts and hit metadata only.

## Math and scoring

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

## Diversity and elites

Elites are selected with **TFBS-core MMR**:

- For each TF, Cruncher extracts the best-hit core sequence oriented to the PWM.
- Distances are computed **per TF**, then averaged across TFs.
- Weights are **tolerant**: low-information PWM positions contribute more to
  distance, preserving consensus-critical bases while encouraging diversity in
  flexible positions.

When bidirectional scoring is enabled, Cruncher uses **canonical identity**
(lexicographic min of sequence and reverse complement) for uniqueness, elite
dedupe, and success counters.

## Run lifecycle

1. **fetch** -> cache motifs/sites in the local catalog
2. **lock** -> resolve TF names to exact cached artifacts and hashes
3. **parse** -> validate locked PWMs (no logo rendering)
4. **sample** -> PT optimization + elite selection + artifacts
5. **analyze** -> curated `plot__*`/`table__*` artifacts + report from artifacts only

The demo workflows in `docs/demos/` follow this lifecycle end-to-end.

## Artifacts and outputs

Sampling writes (under `optimize/` unless noted):

- `optimize/sequences.parquet` (per-draw scores + metadata)
- `optimize/elites.parquet` (elite sequences + summaries)
- `optimize/elites_hits.parquet` (per-elite x per-TF best-hit/core metadata)
- `optimize/random_baseline.parquet` (baseline score cloud)
- `optimize/random_baseline_hits.parquet` (baseline best-hit/core metadata)
- `input/lockfile.json` (snapshot of the pinned lockfile for reproducibility)

Analysis writes:

- `analysis/summary.json`
- `analysis/report.md` and `analysis/report.json`
- `analysis/plot_manifest.json` and `analysis/table_manifest.json`
- `analysis/table__*` (curated tabular artifacts)
- `plots/plot__*` (curated figures)

## Config mapping

Each config block maps directly to a lifecycle phase or runtime contract:

- `workspace` -> run layout and regulator set expansion
- `catalog` -> which cached sources are eligible
- `ingest` / `discover` -> how inputs are sourced or derived
- `sample` -> optimization objective, PT settings, moves, elites
- `analysis` -> report generation and curated plot settings

See the full key reference at `docs/reference/config.md`.

Crosswalk (behavior -> config -> modules -> artifacts):

| Behavior / phase | Config keys (v3) | Primary layers | Writes |
|---|---|---|---|
| Fetch motifs/sites (cache) | `catalog.*`, `ingest.*`, `discover.*` | `ingest/`, `store/` | `<catalog.root>/normalized/...` + `catalog.json` |
| Pin exact inputs (lock) | `workspace.regulator_sets` or `campaigns[]` + `--campaign`, `catalog.*` | `store/`, `app/` | `<workspace>/.cruncher/locks/<config>.lock.json` |
| Validate locked PWMs (parse) | (lockfile-driven) | `app/` | `<workspace>/.cruncher/parse/input/` |
| PT optimization (sample) | `sample.*` | `core/`, `app/` | `<run_dir>/optimize/` + manifest/status updates |
| Elite filter + TFBS-core MMR | `sample.elites.*` | `core/`, `app/` | `optimize/elites*.parquet` |
| Artifact-only reporting (analyze) | `analysis.*` | `analysis/`, `app/` | `<run_dir>/analysis/` + `<run_dir>/plots/` |
| Campaign expand + summarize | `campaigns[]`, `campaign` | `app/` | `outputs/campaign/<name>/{analysis,plots}` |

## Architecture mapping

Cruncher is organized by responsibility:

- `core/` -> scoring, move operators, optimizers (pure compute)
- `ingest/` -> source adapters and normalization
- `store/` -> catalog, locks, and run index
- `app/` -> orchestration of fetch/lock/parse/sample/analyze
- `analysis/` -> artifact-only plotting and reporting
- `cli/` -> UX and command wiring

For the full module layout and on-disk schema, see
`docs/reference/architecture.md`.

## Related docs

- [Sampling + analysis](sampling_and_analysis.md)
- [Config reference](../reference/config.md)
- [Architecture and artifacts](../reference/architecture.md)
- [Two-TF demo (end-to-end)](../demos/demo_basics_two_tf.md)
