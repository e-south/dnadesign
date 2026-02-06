# Cruncher intent and lifecycle

## Contents
- [Intent](#intent)
- [Intended use cases](#intended-use-cases)
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

## Intended use cases

- Generate a small set of high-scoring, diverse sequences for downstream testing.
- Explore tradeoffs when multiple TF motifs must co-exist in a constrained length.
- Compare motif sets by their achievable score/overlap profiles under a shared
  optimization regime.

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
5. **analyze** -> plots/tables + report from artifacts only

The demo workflows in `docs/demos/` follow this lifecycle end-to-end.

## Artifacts and outputs

Sampling writes:

- `artifacts/sequences.parquet` (per-draw scores and metadata)
- `artifacts/elites.parquet` (elite sequences + summaries)
- `artifacts/elites_hits.parquet` (per-elite x per-TF best-hit/core metadata)
- `artifacts/random_baseline.parquet` (baseline score cloud)
- `artifacts/random_baseline_hits.parquet` (baseline best-hit/core metadata)

Analysis writes:

- `analysis/summary.json`
- `analysis/report.md` and `analysis/report.json`
- `analysis/plot_manifest.json` and `analysis/table_manifest.json`

## Config mapping

Each config block maps directly to a lifecycle phase or runtime contract:

- `workspace` -> run layout and regulator set expansion
- `catalog` -> which cached sources are eligible
- `ingest` / `discover` -> how inputs are sourced or derived
- `sample` -> optimization objective, PT settings, moves, elites
- `analysis` -> report generation and curated plot settings

See the full key reference at `docs/reference/config.md`.

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
