# DenseGen demo (end-to-end)

This is the canonical DenseGen demo. It stages a run directory, validates config, plans
constraints, generates sequences, inspects outputs, and renders plots. The demo is Parquet-only
and uses the dense-arrays CBC backend. All paths are explicit; missing files fail fast.

## Contents
- [0) Prereqs](#0-prereqs) - sync deps and set a run root.
- [1) Inspect demo inputs](#1-inspect-demo-inputs) - confirm the input files used by the demo.
- [2) Stage a run directory](#2-stage-a-run-directory) - copy inputs and rewrite paths.
- [3) Validate config](#3-validate-config) - schema and sanity checks.
- [4) Plan constraints](#4-plan-constraints) - see resolved quotas and constraint buckets.
- [5) Describe the resolved run](#5-describe-the-resolved-run) - verify inputs, outputs, solver.
- [6) Run generation](#6-run-generation) - produce sequences and metadata.
- [7) Summarize the run](#7-summarize-the-run) - review run-level counts.
- [8) Inspect outputs](#8-inspect-outputs) - list Parquet artifacts.
- [9) Plot analysis](#9-plot-analysis) - render tf_usage and tf_coverage.
- [Appendix (optional)](#appendix-optional) - PWM sampling + USR output.

## 0) Prereqs

If you have not synced dependencies yet:

```bash
uv sync --locked
```

All commands below assume you are at the repo root. We will write the demo run to a scratch
directory; set a run root:

```bash
RUN_ROOT=/private/tmp/densegen-demo-20260115-1405
mkdir -p "$RUN_ROOT"
```

Pick any writable scratch path; the example outputs below match this path.

## 1) Inspect demo inputs

The canonical demo inputs live here:

```
src/dnadesign/densegen/runs/demo/inputs/tf2tfbs_mapping_cpxR_LexA.csv
src/dnadesign/densegen/runs/demo/inputs/pwm_demo.jaspar
src/dnadesign/densegen/runs/demo/inputs/pwm_demo.meme
src/dnadesign/densegen/runs/demo/inputs/pwm_matrix_demo.csv
```

The demo uses the binding sites CSV by default.

## 2) Stage a run directory

Stage a self-contained run directory from the demo template (this copies inputs and rewrites
paths):

```bash
uv run dense stage --id demo_press --root "$RUN_ROOT" \
  --template src/dnadesign/densegen/runs/demo/config.yaml \
  --copy-inputs
```

Example output:

```text
✨ Run staged: /private/tmp/densegen-demo-20260115-1405/demo_press/config.yaml
```

## 3) Validate config

```bash
uv run dense validate -c /private/tmp/densegen-demo-20260115-1405/demo_press/config.yaml
```

Example output:

```text
✅ Config is valid.
```

## 4) Plan constraints

```bash
uv run dense plan -c /private/tmp/densegen-demo-20260115-1405/demo_press/config.yaml
```

Example output:

```text
┏━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ name ┃ quota ┃ has promoter_constraints ┃
┡━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ demo │ 5     │ yes                      │
└──────┴───────┴──────────────────────────┘
```

## 5) Describe the resolved run

This step shows the resolved inputs, outputs, and the dense-arrays solver selection.

```bash
uv run dense describe -c /private/tmp/densegen-demo-20260115-1405/demo_press/config.yaml
```

Example output (abridged):

```text
Config: /private/tmp/densegen-demo-20260115-1405/demo_press/config.yaml
Run: id=demo_press root=/private/tmp/densegen-demo-20260115-1405/demo_press
┏━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ name ┃ type          ┃ source                                                                                   ┃
┡━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ demo │ binding_sites │ /private/tmp/densegen-demo-20260115-1405/demo_press/inputs/tf2tfbs_mapping_cpxR_LexA.csv │
└──────┴───────────────┴──────────────────────────────────────────────────────────────────────────────────────────┘
┏━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ backend ┃ strategy ┃ options ┃ strands ┃
┡━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ CBC     │ iterate  │ 0       │ double  │
└─────────┴──────────┴─────────┴─────────┘
```

## 6) Run generation

```bash
uv run dense run -c /private/tmp/densegen-demo-20260115-1405/demo_press/config.yaml --no-plot
```

Example output (abridged):

```text
2026-01-15 14:02:02 | INFO | dnadesign.densegen.src.utils.logging_utils | Logging initialized (level=INFO)
Quota plan: demo=5
2026-01-15 14:02:02 | INFO | dnadesign.densegen.src.adapters.optimizer.dense_arrays | Solver selected: CBC
2026-01-15 14:02:05 | INFO | dnadesign.densegen.src.core.pipeline | [demo/demo] 5/5 (100.00%) (local 5/5) CR=1.050 | seq ATTGACAGTAAACCTGCGGGAAATATAATTTACTCCGTATTTGCACATGGTTATCCACAG
🎉 Run complete.
```

On macOS you may see Arrow sysctl warnings after generation; they are emitted by pyarrow and do
not indicate a DenseGen failure.

## 7) Summarize the run

DenseGen writes a `run_manifest.json` in the run root. Summarize it:

```bash
uv run dense summarize --run /private/tmp/densegen-demo-20260115-1405/demo_press
```

Example output:

```text
Run: demo_press  Root: /private/tmp/densegen-demo-20260115-1405/demo_press  Schema: 2.1  dense-arrays: <version> (<source>)
┏━━━━━━━┳━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃ input ┃ plan ┃ generated ┃ duplica… ┃ failed ┃ resamples ┃ librari… ┃ stalls ┃
┡━━━━━━━╇━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│ demo  │ demo │ 5         │ 0        │ 0      │ 0         │ 1        │ 0      │
└───────┴──────┴───────────┴──────────┴────────┴───────────┴──────────┴────────┘
```

Use `--verbose` for constraint-failure breakdowns and duplicate-solution counts.

If any solutions are rejected, DenseGen writes them to `rejections/part-*.parquet` in the run
root.

## 8) Inspect outputs

List the generated Parquet artifacts:

```bash
ls /private/tmp/densegen-demo-20260115-1405/demo_press/outputs/parquet
```

Example output:

```text
_densegen_ids.sqlite
part-10ca57ae0c1d410d8b88206d194a2ff1.parquet
```

## 9) Plot analysis

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
│ tfbs_length_density │ TFBS length distribution (histogram/KDE).            │
│ tfbs_usage          │ TFBS usage by TF, ranked by occurrences.             │
└─────────────────────┴──────────────────────────────────────────────────────┘
```

Then render two plots:

```bash
uv run dense plot -c /private/tmp/densegen-demo-20260115-1405/demo_press/config.yaml --only tf_usage,tf_coverage
```

Example output (abridged):

```text
DenseGen plotting • source: parquet:/private/tmp/densegen-demo-20260115-1405/demo_press/outputs/parquet • rows: 5
Output: /private/tmp/densegen-demo-20260115-1405/demo_press/plots
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ plot        ┃ saved to                                                                  ┃ status ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ tf_usage    │ /private/tmp/densegen-demo-20260115-1405/demo_press/plots/tf_usage.pdf    │ ok     │
│ tf_coverage │ /private/tmp/densegen-demo-20260115-1405/demo_press/plots/tf_coverage.pdf │ ok     │
└─────────────┴───────────────────────────────────────────────────────────────────────────┴────────┘
📊 Plots written.
```

If Matplotlib complains about cache permissions, set a writable cache directory:

```bash
export MPLCONFIGDIR=/tmp/matplotlib
```

List the generated plots:

```bash
ls /private/tmp/densegen-demo-20260115-1405/demo_press/plots
```

Example output:

```text
tf_coverage.pdf
tf_usage.pdf
```

## Appendix (optional)

### PWM sampling input

DenseGen can sample binding sites directly from PWM files. The example below uses a
low-percentile (background-like) sampling strategy:

```yaml
inputs:
  - name: pwm_demo
    type: pwm_jaspar
    path: inputs/pwm_demo.jaspar
    motif_ids: [DemoTF]
    sampling:
      strategy: background
      n_sites: 200
      oversample_factor: 5
      score_percentile: 10
```

Swap `type` and `path` to `pwm_meme` or `pwm_matrix_csv` with the same `sampling` block.

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
    path: outputs/parquet
    deduplicate: true
```

When multiple outputs are configured, DenseGen requires them to be in sync before writing.
