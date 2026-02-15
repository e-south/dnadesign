# Workspace And Demo Guide

This guide describes the workspace contract and the curated demos shipped with `baserender`.

## Workspace Contract

Each workspace contains:
- `job.yaml`
- `inputs/`
- `outputs/`

Operational behavior:
- `job.yaml` relative paths resolve from the workspace root.
- If `results_root` is omitted, runtime defaults to `outputs/`.
- For `images` output with no explicit `dir`, workspace jobs default to `outputs/plots/`.
- `run_report.json` is optional and emitted only when `run.emit_report: true`.

## Workspace Commands

```bash
uv run baserender workspace init demo_run
uv run baserender workspace list
uv run baserender job validate --workspace demo_run
uv run baserender job run --workspace demo_run

# if workspaces are outside the default root:
uv run baserender job run --workspace demo_run --workspace-root /path/to/workspaces
```

## Curated Demos

### `demo_densegen_render`
- input: `inputs/input.parquet` (DenseGen-style TFBS rows)
- output: PNG files under `outputs/plots/`

### `demo_cruncher_render`
- input: `inputs/elites_showcase_records.parquet` (normalized Record-shape hotpath)
- input: `inputs/motif_library.json` (canonical motif primitives)
- output: PDF files under `outputs/plots/`

Input contract intent:
- keep only runtime-essential primitives in `inputs/` (YAGNI)
- no deep Cruncher catalog/path coupling inside baserender demos
- motif logos come from `motif_library.json` (optimization motifs), not inferred from elite windows

`elites_showcase_records.parquet` includes normalized columns consumed by `generic_features`:
- `id`
- `sequence`
- `features`
- `effects`
- `display`

#### `demo_cruncher_render/job.yaml` visual contract

Sequence text tone in this demo is controlled by:

- `render.style.overrides.sequence.bold_consensus_bases: true`
- `render.style.overrides.sequence.non_consensus_color`
- `render.style.overrides.sequence.tone_quantile_low`
- `render.style.overrides.sequence.tone_quantile_high`

When enabled, baserender computes a per-position score (forward and reverse rows independently):

- For each covering motif at position `i`:
  - `R = 2 - H`, where `H = -Σ_b p(b) log2 p(b)` over `b ∈ {A,C,G,T}`
  - `r = R / 2` (normalized information content in `[0,1]`)
  - `s = r * p(b_obs)` where `b_obs` is the displayed row base at `i`
- Aggregate across covering motifs with information weighting:
  - `score_i = Σ s / Σ r`
  - This avoids artificial dilution from stacked low-information motifs.
- Normalize across covered positions using quantile min-max:
  - low = `Q(tone_quantile_low)`, high = `Q(tone_quantile_high)`
  - `norm_i = clamp((score_i - low)/(high - low), 0, 1)`

Multi-motif behavior:

- Contributions are combined by `Σ s / Σ r` across covering motifs.
- If one motif strongly supports the displayed base and another weakly supports it, the final raw score sits between them.
- A low-information motif contributes little because both numerator and denominator are weighted by `r = R/2`.

Sense vs antisense behavior:

- The forward row uses `b_obs = sequence[i]`.
- The reverse row uses `b_obs = reverse_complement(sequence)[i]`.
- Forward-row tone uses only motifs whose target window strand is `fwd`.
- Reverse-row tone uses only motifs whose target window strand is `rev`.
- Each row therefore gets its own strand-scoped score/gray value at the same genomic position.
- Scores are normalized per row over covered positions only; uncovered positions stay at the light endpoint.

Important interpretation note:

- Gray darkness is relative within the panel/row after quantile normalization.
- It is strongest for ranking positions in a panel, not for absolute cross-panel calibration.

Visual mapping:

- `norm_i = 0` -> `non_consensus_color` (light)
- `norm_i = 1` -> `style.color_sequence` (dark)
- intermediate values are linearly interpolated

Interpretation:
- darker sequence glyph = stronger motif-supported alignment at that position
- lighter sequence glyph = weaker support / lower information-weighted match

Run both demos:

```bash
uv run baserender job validate --workspace demo_densegen_render --workspace-root src/dnadesign/baserender/workspaces
uv run baserender job run --workspace demo_densegen_render --workspace-root src/dnadesign/baserender/workspaces

uv run baserender job validate --workspace demo_cruncher_render --workspace-root src/dnadesign/baserender/workspaces
uv run baserender job run --workspace demo_cruncher_render --workspace-root src/dnadesign/baserender/workspaces
```

Keep ad-hoc workspaces out of git. Track only curated demos in this package.
