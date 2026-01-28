## DenseGen demo

This walkthrough runs DenseGen end‑to‑end in a packaged workspace. The workspace already includes two **DenseGen PWM artifact** JSONs (LexA + CpxR) under `inputs/motif_artifacts/`, so you can run Stage‑A immediately without building inputs elsewhere.

What you will do:

- **Stage‑A**: mine and score TFBS per motif → write a cached TFBS pool under `outputs/pools/`.
- **Stage‑B**: sample small libraries from those pools → offer each library to the solver.
- **Solver**: assemble 60‑bp sequences (with fixed promoter constraints) and write Parquet outputs.
- **Inspect + plot + report**: verify what happened using canonical artifacts.

If you want deeper detail on the sampling semantics while reading the demo, jump to:
- [Stage‑A sampling (PWM mining → TFBS pools)](../guide/sampling.md#stage-a-sampling)
- [Stage‑B sampling (TFBS pools → solver libraries)](../guide/sampling.md#stage-b-sampling)

### Contents
0. [Prereqs](#0-prereqs) — sync deps and ensure solver + MEME tools.
1. [Enter the demo workspace](#1-enter-the-demo-workspace) — run from the workspace so paths resolve.
2. [Validate the config](#2-validate-the-config) — schema + solver probe (fail fast).
3. [Inspect inputs](#3-inspect-inputs) — confirm Stage‑A sources and knobs.
3b. [(Optional) Refresh inputs via Cruncher](#3b-optional-refresh-inputs-via-cruncher-external-workspace)
4. [Inspect config](#4-inspect-config) — quick read of Stage‑A/Stage‑B + fixed elements.
5. [Stage‑A build‑pool](#5-stage-a-build-pool) — materialize TFBS pools (cached).
6. [Stage‑B build‑libraries](#6-stage-b-build-libraries) — preview solver libraries (optional but useful).
7. [Run generation](#7-run-generation) — Stage‑B sampling + optimization + outputs.
8. [Inspect run summary](#8-inspect-run-summary) — library utilization + events.
9. [List plots](#9-list-plots) — discover plot names.
10. [Plot](#10-plot) — render canonical diagnostics plots.
11. [Report](#11-report) — write an audit summary (optionally linking plots).
12. [Reset the demo](#12-reset-the-demo) — wipe outputs for a clean rerun.

---

### 0. Prereqs

Install Python deps:

```bash
uv sync --locked
```

Stage‑A PWM sampling uses **MEME Suite FIMO**, so `fimo` must be available on PATH.

This demo assumes `pixi` for a reproducible environment:

```bash
pixi install
pixi run fimo --version
pixi run dense --help
```

Optional convenience alias (lets DenseGen find `fimo` reliably under pixi):

```bash
alias fimo="pixi run fimo"
```

If you are not using pixi, confirm MEME is visible:

```bash
fimo --version
```

If you run DenseGen from source without a `dense` wrapper, prefix with `uv run`:

```bash
uv run dense --help
```

---

### 1. Enter the demo workspace

Run commands from the demo workspace directory so relative paths in `config.yaml` resolve correctly.

```bash
cd src/dnadesign/densegen/workspaces/demo_meme_two_tf
```

If you use pixi to run DenseGen, pin the config path via an alias. (This matters because pixi tasks run from the repo root, not from the workspace.)

```bash
alias dense="pixi run dense -c $PWD/config.yaml"
```

If this workspace has existing outputs and you want a clean start:

```bash
dense campaign-reset
```

---

### 2. Validate the config

Fail fast on schema issues and confirm the solver backend is available.

```bash
dense validate-config --probe-solver
```

If this fails, fix it before moving on—everything else depends on a valid config and a working solver.

---

### 3. Inspect inputs

Confirm what Stage‑A will read and how it will sample.

```bash
dense inspect inputs
```

You should see a PWM input pointing at the packaged artifacts under `inputs/motif_artifacts/`, for example:

* `lexA__meme_suite_meme__lexA_CTGTATAWAWWHACA.json`
* `cpxR__meme_suite_meme__cpxR_MANWWHTTTAM.json`

At this point you’ve verified: inputs resolve, paths exist, and Stage‑A sampling is configured.

---

### 3b. (Optional) Refresh inputs via Cruncher (external workspace)

The demo ships with PWM artifacts, so you can skip this. Do this only if you want to regenerate the LexA/CpxR artifacts from Cruncher.

In a Cruncher workspace (follow `cruncher/docs/demos/demo_basics_two_tf.md`), export DenseGen artifacts directly into this DenseGen workspace:

```bash
cd <cruncher_workspace>
cruncher catalog export-densegen --set 1 --densegen-workspace demo_meme_two_tf
```

Tip: `--densegen-workspace` accepts either:

* a workspace name (resolved under `src/dnadesign/densegen/workspaces`), or
* an absolute path.

Cruncher fails fast if it can’t find `config.yaml` and `inputs/` in the target workspace.

If you also want to drive Stage‑A from explicit binding sites (instead of PWM mining), export them too:

```bash
cruncher catalog export-sites --set 1 --densegen-workspace demo_meme_two_tf
```

Then point a DenseGen input at the exported table (optional):

```yaml
inputs:
  - name: demo_sites
    type: binding_sites
    path: inputs/densegen_sites.parquet
```

---

### 4. Inspect config

`dense inspect config` is your “what will happen” checkpoint: resolved outputs, Stage‑A sampling knobs, Stage‑B sampling policy, and fixed elements.

DenseGen sampling is staged:

* **Stage‑A** materializes per‑TF pools from the PWM artifacts in `inputs/motif_artifacts/`.
* **Stage‑B** repeatedly samples **small solver libraries** from those pools during generation.

If you want the full semantics behind these terms (eligibility, tier targeting math, MMR diversity, coverage weighting), see:

* [Stage‑A sampling (PWM mining → TFBS pools)](../guide/sampling.md#stage-a-sampling)
* [Stage‑B sampling (TFBS pools → solver libraries)](../guide/sampling.md#stage-b-sampling)

What this demo config is doing (quick read):

* **Stage‑A (`densegen.inputs[].sampling`)**
  Mine `n_sites=200` per motif using FIMO score eligibility (`best_hit_score > 0`), collapse near‑duplicates
  by core (`uniqueness.key: core`), and retain a score-first but diversity-biased set via MMR
  (`selection.policy: mmr`, `alpha=0.9`). Sites are length-normalized with `length.policy: range` (`[15, 20]`).

* **Stage‑B (`densegen.generation.sampling`)**
  Build `library_size=20` subsampled libraries using `library_sampling_strategy: coverage_weighted`, requiring
  regulator coverage (`cover_all_regulators: true`) and enforcing core uniqueness inside each library
  (`unique_binding_cores: true`).

Also helpful for interpreting outputs later:

* Fixed promoter constraints: see [Promoter constraints](../guide/generation.md#promoter-constraints).
* `placement_map` overlay semantics: see [Plots](../reference/outputs.md#plots).

Now run:

```bash
dense inspect config
```

---

### 5. Stage‑A build‑pool

Materialize the TFBS pools. This is the canonical Stage‑A artifact step and is typically what you want for reproducible runs and debugging.

```bash
dense stage-a build-pool --fresh
```

Notes:

* By default, `stage-a build-pool` appends new unique TFBS into existing pools; `--fresh` rebuilds from scratch.
* The CLI recap includes per‑TF tier boundary scores, so you can sanity-check tier cutoffs without scripting.
* The resulting pools are cached under `outputs/pools/` and reused by default in subsequent runs.

Optional: immediately visualize Stage‑A yield/tiering/length effects:

```bash
dense plot --only stage_a_summary
```

---

### 6. Stage‑B build‑libraries

This step previews solver libraries without running optimization. It’s optional, but it’s the fastest way to validate that Stage‑B can build feasible libraries from your pools.

```bash
dense stage-b build-libraries
```

What to look for:

* Libraries should contain both regulators when `cover_all_regulators: true`.
* Library feasibility fields (bp accounting and slack) help you catch “can’t fit” problems early.
* The CLI output summarizes libraries per input/plan (min/median/max); details live in `outputs/libraries/`.

If library artifacts already exist and you want to rebuild them:

```bash
dense stage-b build-libraries --overwrite
```

---

### 7. Run generation

Run Stage‑B sampling + solver optimization using the existing Stage‑A pools.

```bash
dense run
```

Important behavior:

* If pools are missing or stale, `dense run` fails fast.
  Fix by rebuilding: `dense stage-a build-pool --fresh`, or run once with: `dense run --rebuild-stage-a`.

This demo config also auto-generates plots during the run (`plots.default`) and writes them to `outputs/plots/` using `plots.format`. Plots are generated from canonical artifacts (Parquet tables + manifests), not from debug logs.

Core outputs to know about:

* `outputs/tables/dense_arrays.parquet` — final sequences (+ metadata)
* `outputs/tables/attempts.parquet` — solver attempt audit log
* `outputs/tables/run_metrics.parquet` — aggregated diagnostics that power run plots
* `outputs/meta/run_manifest.json` + `outputs/meta/events.jsonl` — run summary + structured events

Rerun behavior:

* If run outputs already exist (e.g., `outputs/tables/*.parquet` or `outputs/meta/run_state.json`), choose:

  * `dense run --resume` to continue, or
  * `dense run --fresh` to clear outputs and restart.
* Use `dense run --no-plot` to skip auto-plots on a rerun.

The demo quota is intentionally small (`generation.quota: 12`) with a short per-plan time cap (`runtime.max_seconds_per_plan: 60`) so the end‑to‑end flow stays quick. Scale these up for real runs.

---

### 8. Inspect run summary

Inspect aggregated Stage‑B library usage and runtime events (stalls/resamples/library rebuilds).

```bash
dense inspect run --library --events
```

This is the best single command to answer: “did Stage‑B sample what we expected, and did the run stall?”

---

### 9. List plots

See the available plot names (useful before choosing subsets).

```bash
dense ls-plots
```

---

### 10. Plot

Render selected diagnostics plots from existing outputs.

Stage summaries only:

```bash
dense plot --only stage_a_summary,stage_b_summary
```

Or include the default run plots plus stage summaries:

```bash
dense plot --only placement_map,tfbs_usage,run_health,stage_a_summary,stage_b_summary
```

If Matplotlib complains about cache permissions, set a workspace-scoped cache:

```bash
export MPLCONFIGDIR=outputs/.mpl-cache
```

---

### 11. Report

Generate a human-readable audit summary. Reports don’t generate plots; they can link existing ones.

```bash
dense report --format md --plots include
```

If you skipped plots during the run, generate them first:

```bash
dense plot
```

---

### 12. Reset the demo

Wipe outputs and state so you can rerun cleanly.

```bash
dense campaign-reset
```

This removes the workspace `outputs/` directory but leaves `config.yaml` and `inputs/` intact.

---

@e-south
