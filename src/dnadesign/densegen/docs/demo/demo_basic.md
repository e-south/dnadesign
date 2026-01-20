# DenseGen demo (end-to-end)

This is the canonical DenseGen demo. It stages a workspace, validates config, plans
constraints, generates sequences, inspects outputs, and renders plots. The demo is Parquet-only
and uses the dense-arrays CBC backend. All paths are explicit; missing files fail fast.

## Contents
- [0) Prereqs](#0-prereqs) - sync deps and set a run root.
- [1) Inspect demo inputs](#1-inspect-demo-inputs) - confirm the input files used by the demo.
- [1b) (Optional) Rebuild inputs from Cruncher](#1b-optional-rebuild-inputs-from-cruncher) - cross-tool flow.
- [2) Stage a workspace](#2-stage-a-workspace) - copy inputs and rewrite paths.
- [3) Validate config](#3-validate-config) - schema and sanity checks.
- [4) Plan constraints](#4-plan-constraints) - see resolved quotas and constraint buckets.
- [5) Inspect the resolved run config](#5-inspect-the-resolved-run-config) - verify inputs, outputs, solver.
- [6) (Optional) Stage‑A + Stage‑B previews](#6-optional-stagea--stageb-previews) - preview pools and libraries.
- [7) Run generation](#7-run-generation) - produce sequences and metadata.
- [8) Inspect run summary](#8-inspect-run-summary) - review run-level counts.
- [9) Audit report](#9-audit-report) - build offered-vs-used tables.
- [10) Inspect outputs](#10-inspect-outputs) - list Parquet artifacts.
- [11) Plot analysis](#11-plot-analysis) - render tf_usage and tf_coverage.
- [Appendix (optional)](#appendix-optional) - PWM sampling + USR output.

## 0) Prereqs

If you have not synced dependencies yet:

```bash
uv sync --locked
```

This demo uses **FIMO** (MEME Suite) to adjudicate strong motif matches. Ensure `fimo` is on PATH
or set `MEME_BIN` to the MEME bin directory. If you use pixi, run commands via
`pixi run dense ...` so MEME tools are available (recommended for validation + run steps).

All commands below assume you are at the repo root. We will write the demo run to a scratch
directory; set a run root:

```bash
RUN_ROOT=/private/tmp/densegen-demo-20260115-1405
mkdir -p "$RUN_ROOT"
```

Pick any writable scratch path; the example outputs below match this path.

## 1) Inspect demo inputs

The canonical demo inputs live in the DenseGen demo folder (copied from the Cruncher
basic demo so the run is self‑contained). They are merged into one TF pool via
`pwm_meme_set`:

```
src/dnadesign/densegen/workspaces/demo_meme_two_tf/inputs/lexA.txt
src/dnadesign/densegen/workspaces/demo_meme_two_tf/inputs/cpxR.txt
```

These are MEME files parsed with Cruncher’s MEME parser (DenseGen reuses the same parsing
logic for DRY). The demo uses LexA + CpxR motifs and exercises PWM sampling bounds. Sampling
uses FIMO p-values to define “strong” matches and `selection_policy: stratified` to balance
across canonical p‑value bins (see the input-stage sampling table in `dense inspect inputs`).

Inspect the resolved inputs + Stage‑A sampling table:

```bash
pixi run dense inspect inputs -c src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml
```

### 1b) (Optional) Rebuild inputs from Cruncher

If you want to see the cross‑tool flow (DAP‑seq/RegulonDB → MEME → DenseGen), regenerate
inputs from the Cruncher demo workspace:

```bash
CRUNCHER_CFG=src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml

# Fetch cached sites + motifs (local MEME demo source)
uv run cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c "$CRUNCHER_CFG"
uv run cruncher fetch motifs --source demo_local_meme --tf lexA --tf cpxR --update -c "$CRUNCHER_CFG"
uv run cruncher lock -c "$CRUNCHER_CFG"

# Export DenseGen inputs
uv run cruncher catalog export-sites --set 1 --out /tmp/densegen_sites.csv -c "$CRUNCHER_CFG"
uv run cruncher catalog export-densegen --set 1 --out /tmp/densegen_pwms -c "$CRUNCHER_CFG"
```

Then point DenseGen `inputs` to `/tmp/densegen_sites.csv` (binding‑sites mode) or to the
artifact directory `/tmp/densegen_pwms` (PWM artifact mode).

## 2) Stage a workspace

Stage a self-contained workspace from the demo template (this copies inputs and rewrites
paths):

```bash
uv run dense workspace init --id demo_press --root "$RUN_ROOT" \
  --template src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml \
  --copy-inputs
```

Example output:

```text
✨ Workspace staged: /private/tmp/densegen-demo-20260115-1405/demo_press/config.yaml
```

If you re-run the demo in the same run root and DenseGen’s schema has changed, you may see a
Parquet schema mismatch. Either delete `outputs/dense_arrays.parquet` +
`outputs/_densegen_ids.sqlite` or stage a fresh workspace.

## 3) Validate config

```bash
pixi run dense validate-config -c /private/tmp/densegen-demo-20260115-1405/demo_press/config.yaml
```

Example output:

```text
✅ Config is valid.
```

## 4) Plan constraints

```bash
uv run dense inspect plan -c /private/tmp/densegen-demo-20260115-1405/demo_press/config.yaml
```

Example output:

```text
┏━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ name ┃ quota ┃ has promoter_constraints ┃
┡━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ meme_demo │ 50 │ no                      │
└──────┴───────┴──────────────────────────┘
```

## 5) Inspect the resolved run config

This step shows the resolved inputs, outputs, solver selection, and the two-stage sampling knobs.

```bash
uv run dense inspect config -c /private/tmp/densegen-demo-20260115-1405/demo_press/config.yaml
```

Example output (abridged):

```text
Config: /private/tmp/densegen-demo-20260115-1405/demo_press/config.yaml
Run: id=demo_press root=/private/tmp/densegen-demo-20260115-1405/demo_press
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ name           ┃ type          ┃ source                                                       ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ lexA_cpxR_meme │ pwm_meme_set  │ 2 files (/private/tmp/densegen-demo-20260115-1405/demo_press… │
└────────────────┴───────────────┴──────────────────────────────────────────────────────────────┘
┏━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ backend ┃ strategy ┃ options ┃ strands ┃
┡━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ CBC     │ iterate  │ 0       │ double  │
└─────────┴──────────┴─────────┴─────────┘
Input-stage PWM sampling
... (PWM sampling settings + candidate caps shown here)
Solver-stage library sampling
...
```

## 6) (Optional) Stage‑A + Stage‑B previews

Stage‑A: materialize the TFBS pool (FIMO mining + stratified selection). This is useful when
you want to inspect mining yields per p‑value bin before running the solver:

```bash
pixi run dense stage-a build-pool -c /private/tmp/densegen-demo-20260115-1405/demo_press/config.yaml
```

Stage‑B: build a solver library from the pool without running the solver:

```bash
pixi run dense stage-b build-libraries -c /private/tmp/densegen-demo-20260115-1405/demo_press/config.yaml
```

## 7) Run generation

```bash
pixi run dense run -c /private/tmp/densegen-demo-20260115-1405/demo_press/config.yaml --no-plot
```

The demo config sets `logging.progress_style: screen`, so in a TTY you will see a
refreshing dashboard (progress, leaderboards, last sequence). To see per‑sequence
logs, set `progress_style: stream` (and optionally tune `progress_every`).

Example output (abridged):

```text
2026-01-15 14:02:02 | INFO | dnadesign.densegen.src.utils.logging_utils | Logging initialized (level=INFO)
Quota plan: meme_demo=50
2026-01-15 14:02:02 | INFO | dnadesign.densegen.src.adapters.optimizer.dense_arrays | Solver selected: CBC
2026-01-15 14:02:05 | INFO | dnadesign.densegen.src.adapters.sources.pwm_sampling | FIMO yield for motif lexA: hits=120 accepted=120 selected=80 bins=(0e+00,1e-10]:40 (1e-10,1e-08]:35 ... selected_bins=(0e+00,1e-10]:26 ...
2026-01-15 14:02:06 | INFO | dnadesign.densegen.src.core.pipeline | [demo/demo] 2/50 (4.00%) (local 2/2) CR=1.050 | seq ATTGACAGTAAACCTGCGGGAAATATAATTTACTCCGTATTTGCACATGGTTATCCACAG
2026-01-15 14:02:05 | INFO | dnadesign.densegen.src.core.pipeline | Inputs manifest written: /private/tmp/densegen-demo-20260115-1405/demo_press/outputs/meta/inputs_manifest.json
🎉 Run complete.
```

DenseGen suppresses noisy pyarrow sysctl warnings to keep stdout clean during long runs.

## 8) Inspect run summary

DenseGen writes `outputs/meta/run_manifest.json`, `outputs/meta/inputs_manifest.json`, and
`outputs/meta/effective_config.json`. Summarize the run manifest:

```bash
uv run dense inspect run --run /private/tmp/densegen-demo-20260115-1405/demo_press
```

Example output:

```text
Run: demo_press  Root: /private/tmp/densegen-demo-20260115-1405/demo_press  Schema: 2.4  dense-arrays: <version> (<source>)
┏━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃ input        ┃ plan ┃ generated ┃ duplica… ┃ failed ┃ resamples ┃ librari… ┃ stalls ┃
┡━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│ lexA_cpxR_meme │ meme_demo │ 50  │ 0        │ 0      │ 0         │ 3        │ 0      │
└──────────────┴──────┴───────────┴──────────┴────────┴───────────┴──────────┴────────┘
```

Use `--verbose` for constraint-failure breakdowns and duplicate-solution counts.
Use `--library` to print offered-vs-used summaries for quick debugging:

```bash
uv run dense inspect run --run /private/tmp/densegen-demo-20260115-1405/demo_press --library --top-per-tf 5
```

This library summary is the quickest way to audit which TFBS were offered vs
used in the solver stage (Stage‑B sampling).

If any solutions are rejected, DenseGen writes them to
`outputs/attempts.parquet` in the run root.

## 9) Audit report

Generate an audit-grade summary of the run:

```bash
uv run dense report -c /private/tmp/densegen-demo-20260115-1405/demo_press/config.yaml --format all
```

This writes `outputs/report.json`, `outputs/report.md`, `outputs/report.html`, and `outputs/report_assets/`.

## 10) Inspect outputs

List the generated Parquet artifacts and manifests:

```bash
ls /private/tmp/densegen-demo-20260115-1405/demo_press/outputs
```

Example output:

```text
attempts.parquet
composition.parquet
dense_arrays.parquet
libraries
pools
report.html
report.json
report.md
report_assets
```

Inspect Stage‑A pools and Stage‑B libraries:

```bash
ls /private/tmp/densegen-demo-20260115-1405/demo_press/outputs/pools
ls /private/tmp/densegen-demo-20260115-1405/demo_press/outputs/libraries
```

## 11) Plot analysis

First, list the available plots:

```bash
uv run dense ls-plots
```

Example output:

```text
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ plot                ┃ description                                          ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ compression_ratio   │ Histogram of compression ratios across sequences.    │
│ tf_usage            │ TF usage summary (stacked by length/TFBS or totals). │
│ gap_fill_gc         │ GC content target vs actual for gap-fill pads.       │
│ plan_counts         │ Plan counts over time by promoter constraint bucket. │
│ tf_coverage         │ Per-base TFBS coverage across sequences.             │
│ tfbs_positional_frequency │ TFBS positional frequency (line plot).        │
│ tfbs_positional_histogram │ Positional TFBS histogram (overlaid, per-nt).  │
│ diversity_health    │ Diversity health over time (coverage + entropy).      │
│ tfbs_length_density │ TFBS length distribution (histogram/KDE).            │
│ tfbs_usage          │ TFBS usage by TF, ranked by occurrences.             │
└─────────────────────┴──────────────────────────────────────────────────────┘
```

Then render four plots:

```bash
uv run dense plot -c /private/tmp/densegen-demo-20260115-1405/demo_press/config.yaml --only tf_usage,tf_coverage,tfbs_positional_histogram,diversity_health
```

Example output (abridged):

```text
DenseGen plotting • source: parquet:/private/tmp/densegen-demo-20260115-1405/demo_press/outputs/dense_arrays.parquet • rows: 5
Output: /private/tmp/densegen-demo-20260115-1405/demo_press/outputs
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ plot        ┃ saved to                                                                  ┃ status ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ tf_usage    │ /private/tmp/densegen-demo-20260115-1405/demo_press/outputs/tf_usage.png    │ ok     │
│ tf_coverage │ /private/tmp/densegen-demo-20260115-1405/demo_press/outputs/tf_coverage.png │ ok     │
│ tfbs_positional_histogram │ /private/tmp/densegen-demo-20260115-1405/demo_press/outputs/tfbs_positional_histogram.png │ ok │
│ diversity_health │ /private/tmp/densegen-demo-20260115-1405/demo_press/outputs/diversity_health.png │ ok │
└─────────────┴───────────────────────────────────────────────────────────────────────────┴────────┘
📊 Plots written.
```

If Matplotlib complains about cache permissions, set a writable cache directory:

```bash
export MPLCONFIGDIR=/tmp/matplotlib
```

List the generated plots:

```bash
ls /private/tmp/densegen-demo-20260115-1405/demo_press/outputs
```

Example output:

```text
tf_coverage.png
tf_usage.png
```

## Appendix (optional)

### PWM sampling input

DenseGen can sample binding sites directly from PWM files. The example below uses the
LexA MEME motif (copied from the Cruncher demo so it is self-contained) and a
low-percentile (background-like) sampling strategy:

```yaml
inputs:
  - name: lexA_meme
    type: pwm_meme
    path: inputs/lexA.txt
    motif_ids: [lexA]
    sampling:
      strategy: background
      scoring_backend: densegen
      n_sites: 200
      oversample_factor: 5
      score_percentile: 10
```

Swap `type` and `path` to `pwm_jaspar` or `pwm_matrix_csv` with the same `sampling` block.

For **strong match** sampling with FIMO p-values:

```yaml
inputs:
  - name: lexA_meme
    type: pwm_meme
    path: inputs/lexA.txt
    motif_ids: [lexA]
    sampling:
      strategy: stochastic
      scoring_backend: fimo
      pvalue_threshold: 1e-4
      selection_policy: top_n
      n_sites: 80
      oversample_factor: 10
```

To mine specific affinity strata, add canonical p‑value bins and select bins by index:

```yaml
    sampling:
      scoring_backend: fimo
      pvalue_threshold: 1e-3
      selection_policy: stratified
      pvalue_bins: [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
      mining:
        batch_size: 5000
        max_batches: 4
        retain_bin_ids: [1, 2]  # (1e-6..1e-4] and (1e-4..1e-3]
```

### Add USR output

USR is an optional I/O adapter. To write both Parquet and USR:

```yaml
output:
  targets: [usr, parquet]
  schema:
    bio_type: dna
    alphabet: dna_4
  usr:
    dataset: demo_densegen
    root: /path/to/usr/datasets
    allow_overwrite: false
  parquet:
    path: outputs/dense_arrays.parquet
    deduplicate: true
```

When multiple outputs are configured, DenseGen requires them to be in sync before writing.
