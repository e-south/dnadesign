
## DenseGen demo

This demo runs inside a packaged workspace that already contains inputs and a `config.yaml`. It exercises the following pipeline:

- **Stage‑A**: mine + score PWM‑sampled TFBS candidates with **FIMO**, then retain a TFBS pool.
- **Stage‑B**: call the dense-arrays backend to assemble sequences subject to your TFBS pools and defined constraints.
- **Outputs**: write tables, manifests, plots, and an audit report under `outputs/`.

This workspace uses **three PWM artifacts** (LexA, BaeR, and CpxR).

---

### Contents

- [Prereqs](#prereqs)
- [1) Enter the demo workspace](#1-enter-the-demo-workspace)
- [2) Validate config + solver](#2-validate-config--solver)
- [3) Build Stage‑A pools](#3-build-stage-a-pools)
- [4) Optional: Build Stage‑B libraries](#4-optional-build-stage-b-libraries)
- [5) Run generation](#5-run-generation)
- [6) Plot](#6-plot)
- [7) Report (optional)](#7-report-optional)
- [8) Reset the demo (optional)](#8-reset-the-demo-optional)
- [Where outputs go](#where-outputs-go)
- [Common troubleshooting](#common-troubleshooting)
- [Next steps](#next-steps)

---

## Prereqs

You need:

1) **Python deps** (DenseGen + dependencies)
2) **MEME Suite** available so `fimo` is on PATH
3) A **solver backend** (e.g., CBC, GUROBI) that your dense-arrays install can see

From the repo root, sync Python dependencies and system tools:

```bash
uv sync --locked
pixi install
pixi run fimo --version
```

Workspace + runner setup:

```bash
# Option A: cd into the workspace
cd src/dnadesign/densegen/workspaces/demo_meme_two_tf  # enter demo workspace
CONFIG="$PWD/config.yaml"  # point to workspace config

# Option B: run from anywhere in the repo
CONFIG=src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml  # config path from repo root

# Choose a runner (pixi is the default in this repo; uv is optional).
dense() { pixi run dense -- "$@"; }  # convenience wrapper

# Optional: uv-only wrapper
# dense() { uv run dense "$@"; }

# From here on, commands use $CONFIG for clarity; if you're in the workspace, you can omit -c.
```

If you want DenseGen to verify your solver is reachable, `dense validate-config --probe-solver -c "$CONFIG"`
does that (step 2).

---

## 1) Enter the demo workspace (if needed)

```bash
cd src/dnadesign/densegen/workspaces/demo_meme_two_tf
CONFIG="$PWD/config.yaml"
```

Why this matters:

* This folder is a **self-contained workspace**: it has a `config.yaml` plus `inputs/` and an `outputs/` root.
* DenseGen’s CLI can **auto-discover** config when you are in (or under) a directory containing `config.yaml`.

Optional, but useful: print what the workspace thinks it will use.

```bash
dense inspect inputs -c "$CONFIG"
dense inspect plan -c "$CONFIG"
dense inspect config -c "$CONFIG"
```

Those inspection commands are “read-only” and are the fastest way to confirm:

* which inputs are wired (and whether pools/libraries already exist),
* what plan items will run (quota/fraction resolution),
* which solver backend + strategy is configured.

---

## 2) Validate config + solver

```bash
dense validate-config --probe-solver -c "$CONFIG"
```

What this does:

* **Schema validation**: unknown keys and invalid values are errors (no silent fallbacks).
* **Path sanity**: ensures paths resolve correctly (relative to `config.yaml`).
* **Solver probe (optional)**: verifies your configured backend is available *before* you do any work.

If you hit an error here, fix it now—everything downstream assumes config is valid.

---

## 3) Build Stage‑A pools

```bash
dense stage-a build-pool --fresh -c "$CONFIG"
```

What Stage‑A is doing (high-level):

* For PWM-backed inputs, Stage‑A generates candidate sites, scores them with **FIMO log‑odds**
  (forward strand only), applies eligibility rules, deduplicates by your configured uniqueness key
  (commonly `core` for PWM inputs), and then **retains** `n_sites` per regulator according to the
  selection policy (e.g., top-score or MMR).
* Stage‑A writes the **retained pool**—the exact sites Stage‑B will draw from later. Diagnostics may
  compare “top-score” vs “diversified,” but the pool parquet is the single source of truth.

What you should see:

* A **Stage‑A plan** table (per input × TF) showing retain counts, mining budget, eligibility rule,
  selection policy, uniqueness mode, and length policy.
* A mining progress section (per motif) showing how many candidates were scored and how many unique,
  eligible sites were found.
* A recap summary with score and diversity diagnostics plus a legend.

A real Stage‑A run in this demo looks like (abridged):

```text
Stage-A plan
│ input               │ TF   │ retain │ budget       │ eligibility      │ selection   │ uniqueness │ length        │
│ lexA_cpxR_artifacts │ cpxR │ 250    │ fixed=500000 │ best_hit_score>0 │ mmr(a=0.50) │ core       │ range(15..20) │
│ lexA_cpxR_artifacts │ lexA │ 250    │ fixed=500000 │ best_hit_score>0 │ mmr(a=0.50) │ core       │ range(15..20) │

Stage-A sampling recap
│ TF   │ generated │ eligible_unique │ pool │ retained │ tier fill │ selection   │ overlap │ ... │
│ cpxR │ 500,000   │ 122,064 (25%)   │ ...  │ 250      │ 1.000%    │ mmr(a=0.50) │ 36.0%   │ ... │
│ lexA │ 500,000   │ 86,514 (17%)    │ ...  │ 250      │ 1.000%    │ mmr(a=0.50) │ 32.0%   │ ... │

✨ Pool manifest written: outputs/pools/pool_manifest.json
```

How to interpret common recap fields:

* **generated**: how many PWM candidates were scored (your mining budget).
* **eligible_unique**: how many unique sites survive eligibility + dedupe.
* **retained**: the final pool size for that TF (what Stage‑B will sample from).
* **tier fill**: the deepest diagnostic rung needed to collect enough candidates for the selection pool.
* **selection**: the Stage‑A retention policy (e.g., `top_score` or `mmr(alpha=…)`).
* **overlap**: how much the diversified selection overlaps the pure top-score set (MMR diagnostics).

Where Stage‑A writes:

* `outputs/pools/pool_manifest.json` — the audit-friendly Stage‑A summary (config hash, fingerprints,
  mining yields, score summaries, diversity summaries, warnings/shortfalls).
* `outputs/pools/<input>__pool.parquet` — the **retained TFBS pool** actually used downstream.

Stage‑A caching semantics:

* Without `--fresh`, `dense stage-a build-pool` appends *new unique* TFBS into an existing pool by default.
* With `--fresh`, it rebuilds from scratch (recommended for demos / pressure testing).

> Important: the regulator labels that show up in Stage‑A pools are the labels you must use in
> `generation.plan[].required_regulators`. For PWM artifact inputs, those labels are typically motif IDs.
> If you’re unsure what they are, run `dense inspect inputs --show-motif-ids -c "$CONFIG"`.

---

## 4) Optional: Build Stage‑B libraries

```bash
dense stage-b build-libraries --overwrite -c "$CONFIG"
```

You do **not** need this step for a normal run when `library_source: build` (the default);
`dense run` will build libraries automatically.

This helper is useful when you want to:

* **Inspect feasibility** before running the solver (are there enough bp/sites to satisfy constraints?).
* **Materialize library artifacts** for replay workflows (`library_source: artifact`).
* Compare library builds across config tweaks without running the full generation.

Where Stage‑B writes:

* `outputs/libraries/library_builds.parquet`
* `outputs/libraries/library_members.parquet`
* `outputs/libraries/library_manifest.json`

---

## 5) Run generation

```bash
dense run -c "$CONFIG"
```

What happens here:

* DenseGen consumes Stage‑A pools (and Stage‑B libraries, either prebuilt or built on demand),
  then calls the solver to generate sequences under your plan and constraints.
* Stage‑B is the only stage that typically resamples during a run. When runtime guards trigger
  (stalls, duplicates, exhaustion), DenseGen rebuilds libraries and tries again.

Run guards you should be aware of:

* If `outputs/` already contains run-state or output tables, DenseGen will require you to choose:

  * `dense run --resume -c "$CONFIG"` (continue), or
  * `dense run --fresh -c "$CONFIG"` (delete `outputs/` then start over)

Debugging tips (optional flags):

* `dense run --show-tfbs -c "$CONFIG"` — include TFBS strings in progress output
* `dense run --show-solutions -c "$CONFIG"` — include full solution sequences in progress output

After the run completes, inspect the run summary:

```bash
dense inspect run --events --library -c "$CONFIG"
```

Why `dense inspect run` is worth doing:

* It summarizes generated counts, duplicates, failures, resamples, and library rebuilds.
* `--library` gives an aggregated offered-vs-used view across Stage‑B libraries (high signal when diagnosing “why am I stalling?”).
* `--events` summarizes runtime events from `outputs/meta/events.jsonl`.

---

## 6) Plot

The canonical “small but high-signal” plot set for this demo is:

```bash
dense plot --only stage_a_summary,placement_map -c "$CONFIG"
```

What these plots tell you:

* **stage_a_summary**: pool quality (yield, score distribution, diversity diagnostics)
* **placement_map**: the run fingerprint (per-position occupancy, TF allocation, and fixed-element overlays)

If you want to discover what plots exist:

```bash
dense ls-plots -c "$CONFIG"
```

Operational note: run health is usually faster to inspect than to plot:

```bash
# dense inspect run --events --library -c "$CONFIG"
```

---

## 7) Report (optional)

```bash
dense report --plots include -c "$CONFIG"
```

Notes:

* `dense report` generates audit-grade summaries from the canonical outputs.
* `--plots include` links plots from `outputs/plots/plot_manifest.json` (so run `dense plot` first).

---

## 8) Reset the demo (optional)

If you want to rerun from scratch (keeping config + inputs intact):

```bash
dense campaign-reset -c "$CONFIG"
```

This removes the entire `outputs/` directory under the configured run root.

---

## Where outputs go

In this packaged workspace, everything is written under:

```
src/dnadesign/densegen/workspaces/demo_meme_two_tf/outputs/
```

Common layout:

```
outputs/
  tables/
    dense_arrays.parquet
    attempts.parquet
    solutions.parquet
    composition.parquet
    run_metrics.parquet
  pools/
  libraries/
  plots/
  report/
  meta/
  logs/
```

If you’re learning the artifact model, start here:

* `outputs/tables/dense_arrays.parquet` — final sequences (canonical dataset)
* `outputs/meta/run_manifest.json` — run summary (what happened)
* `outputs/pools/pool_manifest.json` — Stage‑A summary (what got retained, and why)
* `outputs/meta/events.jsonl` — structured timeline of resamples, stalls, library builds

For full schemas and join keys, see: `../guide/outputs-metadata.md` and `../reference/outputs.md`.

---

## Common troubleshooting

### `fimo: command not found`

DenseGen’s PWM-backed Stage‑A requires MEME Suite.
Ensure `fimo` is on PATH (or run via your environment manager that provides it).

```bash
pixi run fimo --version
```

### Solver backend not available

Run:

```bash
dense validate-config --probe-solver -c "$CONFIG"
```

If the probe fails, either install/configure your backend (CBC / GUROBI) or adjust `densegen.solver`
in `config.yaml`.

### “required_regulators not found” / regulator label mismatches

Your `required_regulators` must match the regulator labels used in Stage‑A pools.

Run:

```bash
dense inspect inputs --show-motif-ids -c "$CONFIG"
```

Then copy the exact labels into `generation.plan[].required_regulators`.

### Plotting issues / Matplotlib cache permissions

If Matplotlib complains about cache directories, set a writable cache path:

```bash
export MPLCONFIGDIR=/tmp/matplotlib
```

---

## Next steps

If you want the deeper “why” behind each stage:

* `../guide/sampling.md` — Stage‑A + Stage‑B sampling semantics and how to read plots
* `../guide/generation.md` — plan items, fixed elements, and solver strategy
* `../guide/inputs.md` — input types and Stage‑A PWM behavior
* `../workflows/cruncher_pwm_pipeline.md` — Cruncher → DenseGen artifact handoff
* `../reference/cli.md` — full CLI operator manual
* `../reference/config.md` — exact config schema (strict)

---

@e-south

````
