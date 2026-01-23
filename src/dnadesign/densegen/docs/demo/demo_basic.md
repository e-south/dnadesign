## DenseGen demo (workspace‑first)

This walkthrough uses the packaged demo template. The staged workspace contains MEME `.txt` motifs
in `inputs/` (lexA + cpxR), and Stage‑A sampling uses those files directly.

### Contents
- [0) Prereqs](#0-prereqs) - sync deps and ensure solver tools.
- [1) Stage a workspace](#1-stage-a-workspace) - scaffold a self‑contained workspace.
- [2) Validate config](#2-validate-config) - schema + solver probe.
- [3) Inspect inputs](#3-inspect-inputs) - Stage‑A inputs + sampling summary.
- [3b) (Optional) Build inputs via Cruncher (external workspace)](#3b-optional-build-inputs-via-cruncher-external-workspace)
- [4) Inspect config](#4-inspect-config) - resolved outputs + Stage‑A/Stage‑B settings.
- [5) Stage‑A build‑pool](#5-stage-a-build-pool) - materialize TFBS pools.
- [6) Stage‑B build‑libraries](#6-stage-b-build-libraries) - materialize solver libraries.
- [7) Run generation](#7-run-generation) - execute Stage‑A + Stage‑B + optimization.
- [8) Inspect run summary](#8-inspect-run-summary) - library + events.
- [9) List plots](#9-list-plots) - available plot names.
- [10) Plot](#10-plot) - render plots.
- [11) Report](#11-report) - write audit report.
- [12) Reset the demo](#12-reset-the-demo) - wipe outputs for a clean rerun.

---

### 0) Prereqs

If you have not synced dependencies yet:

```bash
uv sync --locked
```

Stage‑A FIMO sampling requires MEME Suite (`fimo` on PATH). If you use pixi, run commands via
`pixi run dense ...` so MEME tools are available. If running from source, prefix commands with
`uv run`.

---

### 1) Stage a workspace

Why: create a self‑contained workspace with `config.yaml`, `inputs/`, and `outputs/`.

```bash
dense workspace init --id demo --template-id demo_meme_two_tf --copy-inputs
cd demo
```

---

### 2) Validate config

Why: fail fast on schema issues and confirm solver availability.

```bash
dense validate-config --probe-solver
```

---

### 3) Inspect inputs

Why: confirm Stage‑A inputs and sampling settings.

```bash
dense inspect inputs
```

The demo uses MEME `.txt` motifs already in `inputs/` (`lexA.txt`, `cpxR.txt`).

---

### 3b) (Optional) Build inputs via Cruncher (external workspace)

Why: generate Stage‑A motif artifacts and binding‑site tables in **Cruncher’s** workspace, then
copy the exports into this DenseGen workspace.

Follow the Cruncher demo (see `cruncher/docs/demos/demo_basics_two_tf.md`) in its own workspace.
From the Cruncher workspace directory, export DenseGen inputs (no `-c` flag needed when you run in CWD):

```bash
cd <cruncher_workspace>
cruncher catalog export-sites --set 1 --out outputs/exports/densegen_sites.csv
cruncher catalog export-densegen --set 1 --out outputs/exports/densegen_pwms
```

Copy those exports into this DenseGen workspace:

```bash
cp outputs/exports/densegen_sites.csv <densegen_workspace>/inputs/
cp -R outputs/exports/densegen_pwms <densegen_workspace>/inputs/motif_artifacts
```

To use these exports, update `config.yaml` inputs to `type: binding_sites` (CSV/Parquet) or
`type: pwm_artifact_set` (JSON artifacts). The DenseGen workspace remains config‑centric (one
runtime config), while Cruncher keeps its own workspace + config.

---

### 4) Inspect config

Why: confirm resolved outputs, Stage‑A sampling knobs, fixed elements, and Stage‑B sampling policy.

Rationale for the demo settings: we want **dozens of binding sites per motif**, so we set Stage‑A
`n_sites` and oversampling/mining caps to reach that target; Stage‑B sampling then builds fixed‑size
libraries before running the solver.
This demo also pins a strong σ70 promoter pair (`TTGACA`/`TATAAT`) as fixed elements; the default
`tf_coverage` plot overlays these sites when `plots.options.tf_coverage.include_promoter_sites: true`.
To keep the 60‑bp budget feasible with ~21–22 bp TFBS lengths, the plan sets
`min_required_regulators: 1` while listing both LexA and CpxR, so each sequence must include at
least one of the two regulators.

```bash
dense inspect config
```

---

### 5) Stage‑A build‑pool

Why: materialize TFBS pools for inspection and for deterministic Stage‑B previews.

```bash
dense stage-a build-pool
```

---

### 6) Stage‑B build‑libraries

Why: preview solver libraries without running the optimizer.

```bash
dense stage-b build-libraries
```

---

### 7) Run generation

Why: execute Stage‑A sampling (if needed), Stage‑B sampling, and solver optimization.

```bash
dense run
```

This demo config also enables plot generation from the run (`plots.default`) and saves plots in
`outputs/plots/` using `plots.format` (switch to `pdf` or `svg` in `config.yaml` if desired).
Reports do not generate plots; they can optionally link the existing plot manifest.
The demo quota is intentionally small (`generation.quota: 12` with `runtime.max_seconds_per_plan: 60`)
to keep the end‑to‑end run fast; scale these up for production runs.
The demo uses `solver.strategy: iterate` for full solver runs; switch to `diverse` or `optimal`
as needed for exploration.
If run outputs already exist (e.g., `outputs/tables/*.parquet` or `outputs/meta/run_state.json`),
choose `--resume` to continue or `--fresh` to clear outputs. Use `dense run --no-plot` to skip
auto‑plots when re‑running.

---

### 8) Inspect run summary

Why: inspect Stage‑B library usage and runtime events.

```bash
dense inspect run --library --events --library-limit 5
```

---

### 9) List plots

Why: see available plot names before selecting a subset.

```bash
dense ls-plots
```

---

### 10) Plot

Why: render selected plots from existing outputs.

```bash
dense plot --only tf_usage,tf_coverage
```

If Matplotlib complains about cache permissions, set a workspace‑scoped cache:

```bash
export MPLCONFIGDIR=outputs/.mpl-cache
```

---

### 11) Report

Why: generate a human‑readable audit summary.

```bash
dense report --format md --plots include
```

If you skipped plots during the run, generate them first:

```bash
dense plot
```

---

### 12) Reset the demo

Why: wipe run outputs and state so you can re-run the demo cleanly.

```bash
dense campaign-reset
```

This removes the workspace `outputs/` directory but leaves `config.yaml` and `inputs/` intact.

---

@e-south
