## cruncher demo 1

**Design short dsDNA sequences that satisfy two PWMs at once.**

**cruncher** scores each TF by the best PWM match anywhere in the candidate sequence on either strand, then optimizes the min/soft‑min across TFs so the weakest TF improves. It explores sequence space with Gibbs + parallel tempering (MCMC) and returns a diverse elite set (unique up to reverse‑complement) plus diagnostics for stability/mixing. Motif overlap is allowed and treated as informative structure in analysis.

**Terminology:**

- **sites** = training binding sequences
- **PWMs/matrices** = scoring models
- **lock** pins the exact matrices used for scoring.

### Contents

- [Demo setup](#demo-setup)
- Core workflow
  - [Preview sources and inventories](#preview-sources-and-inventories)
  - [Fetch local DAP-seq motifs + binding sites](#fetch-local-dap-seq-motifs--binding-sites)
  - [Fetch binding sites (RegulonDB, network)](#fetch-binding-sites-regulondb-network)
    - [Optional: HT-only or combined site modes (RegulonDB)](#optional-ht-only-or-combined-site-modes-regulondb)
  - [Verify cache (pre-lock)](#verify-cache-pre-lock)
  - [Combine curated + DAP-seq sites for discovery](#combine-curated--dap-seq-sites-for-discovery)
  - [Discover motifs from merged sites (MEME/STREME)](#discover-motifs-from-merged-sites-memestreme)
    - [Optional: trim aligned PWMs to a fixed window](#optional-trim-aligned-pwms-to-a-fixed-window)
  - [Inspect candidate PWMs + logos (pre-lock)](#inspect-candidate-pwms--logos-pre-lock)
    - [Optional: local motifs or alignment matrices](#optional-local-motifs-or-alignment-matrices)
  - [Select motif source + lock](#select-motif-source--lock)
  - [Inspect cached entries and targets (post-lock)](#inspect-cached-entries-and-targets-post-lock)
  - [Parse workflow (validate locked motifs)](#parse-workflow-validate-locked-motifs)
- [Sample (MCMC optimization)](#sample-mcmc-optimization)
    - [Run artifacts + performance snapshot](#run-artifacts--performance-snapshot)
    - [Analyze + report](#analyze--report)
    - [Optional: live analysis notebook](#optional-live-analysis-notebook)

---

> **Fast path (local-only, no discovery):** run end‑to‑end using the packaged demo motifs (no network and no MEME Suite required).
>
> ```bash
> cruncher fetch motifs --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
> cruncher fetch sites  --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
> cruncher lock   -c "$CONFIG"
> cruncher parse  -c "$CONFIG"
> cruncher sample -c "$CONFIG"
> cruncher analyze -c "$CONFIG"
> cruncher report --latest -c "$CONFIG"
> ```
>
> Continue below for a more in depth guide.

### Demo setup

- **Workspace**: `src/dnadesign/cruncher/workspaces/demo_basics_two_tf/`
- **Config**: `config.yaml`
- **Output root**: `runs/` (relative to the workspace)
- **Catalog cache**: `src/dnadesign/cruncher/.cruncher/` (shared across workspaces by default)
- **Motif flow**: fetch sites + local motifs → inspect/select → lock/sample using chosen matrices
- **Path placeholders**: example outputs use `<workspace>` for the demo workspace root

```bash
# Option A: cd into the workspace
cd src/dnadesign/cruncher/workspaces/demo_basics_two_tf  # enter demo workspace
CONFIG="$PWD/config.yaml"  # point to workspace config

# Option B: run from anywhere in the repo
CONFIG=src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml  # config path from repo root

# Choose a runner (pixi is the default in this repo; uv is optional).
cruncher() { pixi run cruncher -- "$@"; }  # convenience wrapper

# Optional: uv-only wrapper
# cruncher() { uv run cruncher "$@"; }

# From here on, commands use $CONFIG for clarity; if you're in the workspace, you can omit --config.
```

If you plan to run **motif discovery** (`discover`), verify MEME Suite and external tools:

```bash
cruncher doctor -c "$CONFIG"
```

Sampling + analysis do **not** require MEME Suite, so you can still run the fast path above without it. If it reports missing tools, install MEME Suite (pixi or system install) or set `motif_discovery.tool_path`; see the [MEME Suite guide](../guides/meme_suite.md). Local demo motifs live at `data/local_motifs/` (DAP‑seq MEME files from [this](https://www.nature.com/articles/s41592-021-01312-2) study).

---

### Preview sources and inventories

Optional: run this if you want to confirm which sources the config will use (and which steps will require network access).

```bash
# List configured sources
cruncher sources list -c "$CONFIG"  # list configured sources
```

```bash
# Inspect local demo source metadata
cruncher sources info demo_local_meme -c "$CONFIG"  # inspect demo source metadata

# Inspect RegulonDB source metadata
cruncher sources info regulondb -c "$CONFIG"  # inspect RegulonDB metadata
```
If you’re debugging RegulonDB inventory size, `cruncher sources summary --source regulondb --scope remote --remote-limit 20` is a quick sample. Use `sources datasets` or `fetch sites --dry-run` when you need HT dataset coverage.

---

### Fetch local DAP-seq motifs + binding sites

Local DAP-seq MEME files live under `data/local_motifs/` in the workspace. Fetch the motif matrices and the MEME BLOCKS training sites:

```bash
# Cache local motif matrices
cruncher fetch motifs --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"  # fetch local PWM matrices

# Cache local MEME BLOCKS sites
cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"  # fetch local training sites
```

---

### Fetch binding sites (RegulonDB, network)

Optional: fetch curated sites from RegulonDB if you want extra site evidence for discovery or to compare against curated annotations. If you’re offline, skip this — the demo still runs end‑to‑end with local motifs.

```bash
# Fetch curated sites from RegulonDB
cruncher fetch sites --tf lexA --tf cpxR --update -c "$CONFIG"  # fetch curated RegulonDB sites
```

Note: curated site sets aren’t partitioned by dataset/method; those fields are populated for HT datasets.

---

#### Optional: HT-only or combined site modes (RegulonDB)

You can restrict `pwm_source: sites` to curated or HT sites, or combine them. Make these adjustments before discovery/lock if you plan to build site-derived PWMs.

```yaml
motif_store:
  pwm_source: sites
  site_kinds: ["curated"]          # curated-only
  # site_kinds: ["meme_blocks"]    # local DAP-seq training sites only
  # site_kinds: ["ht_tfbinding"]   # HT-only (RegulonDB)
  # combine_sites: true            # combine across sources when site_kinds is null
```

Tip: if HT datasets return peaks without sequences, hydration uses NCBI by default (`ingest.genome_source=ncbi`). To run offline, provide a local FASTA via `--genome-fasta` or `ingest.genome_fasta`.

---

### Verify cache (pre-lock)

Optional sanity check. If you just ran `fetch`, you can skip this; `lock` / `targets status` will fail loudly if required inputs are missing.

```bash
cruncher catalog list -c "$CONFIG"  # quick view of cached motifs + sites
```

If you need per‑source cache counts, use `cruncher sources summary --scope cache`.

---

### Combine curated + DAP-seq sites for discovery

This only affects **motif discovery** (how site sets are merged before running MEME/STREME). With `combine_sites: true` (the demo default), all selected site sources for a TF are merged into one discovery input. Set `combine_sites: false` if you want separate per-source discoveries.

```yaml
motif_store:
  combine_sites: false
```

After changing `combine_sites` or `site_kinds`, re-run `cruncher lock` so parse/sample use the updated site sets. Tip: to use only local DAP‑seq training sites in discovery, set `site_kinds: ["meme_blocks"]`.

---

### Discover motifs from merged sites (MEME/STREME)

Optional: run discovery if you want to **rebuild a single aligned PWM per TF from cached sites** (useful when sites have mixed lengths or come from multiple sources). If you already trust the packaged matrices, skip to **Inspect candidate PWMs + logos** (or go straight to `lock`).

If you fetched binding sites from multiple sources and want a single aligned PWM per TF (e.g., to reconcile different site lengths), run MEME Suite on the combined site sets. This creates new motif matrices under `motif_discovery.source_id` (default `meme_suite_streme`, or override with `--source-id`).

Key points:

- Reads cached binding sites (variable lengths are OK).
- Outputs new PWM entries under the discovery `source_id`; `source_preference` selects which to use and `lock` pins IDs/hashes.
- Runs write under `src/dnadesign/cruncher/.cruncher/discoveries/` (shared catalog root) and ingest motifs into the catalog.

```bash
# Verify discovery prerequisites
cruncher discover check -c "$CONFIG"  # verify MEME Suite availability + inputs

# Run STREME discovery
cruncher discover motifs --tf lexA --tf cpxR --tool streme --source-id meme_suite_streme -c "$CONFIG"  # discover PWMs from sites

# Confirm discovered motifs in catalog
cruncher catalog list --source meme_suite_streme -c "$CONFIG"  # verify new motif entries
```

Discovery prints a short table (TF, tool, motif ID, length) and stores full outputs under the shared catalog root (e.g., `.../.cruncher/discoveries/`). Tip: if each sequence represents one site, prefer MEME and set `--meme-mod oops`.

```bash
# Run MEME discovery (OOPS)
cruncher discover motifs --tf lexA --tf cpxR --tool meme --meme-mod oops --source-id meme_suite_meme -c "$CONFIG"  # discover PWMs with MEME
```

This demo config already prefers MEME motifs for optimization: `pwm_source: matrix` with `source_preference: [meme_suite_meme, meme_suite_streme, ...]`. After discovery, continue to the optional trimming/inspection steps below, then lock so parse/sample use the new motifs.

---

#### Optional: trim aligned PWMs to a fixed window

If optimization needs a shorter PWM, set `pwm_window_lengths` to select the highest-information contiguous window before sampling:

```yaml
motif_store:
  pwm_window_lengths:
    lexA: 15
    cpxR: 15
```

Then inspect candidate PWMs/logos (next section) and proceed to `lock` and `sample` below.

---

### Inspect candidate PWMs + logos (pre-lock)

Use `catalog pwms/logos` **before** locking to compare candidate matrices (e.g., MEME vs STREME). `catalog pwms` reports information content in bits; site‑derived PWMs use Biopython with configurable pseudocounts (`motif_store.pseudocounts`):

```bash
# Summarize candidate PWMs
cruncher catalog pwms --set 1 -c "$CONFIG"  # summarize candidate matrices

# Render candidate logos
cruncher catalog logos --set 1 -c "$CONFIG"  # render logos for set 1

# Optional: compare MEME vs STREME sources directly
cruncher catalog pwms --source meme_suite_streme --set 1 -c "$CONFIG"  # inspect STREME matrices
cruncher catalog pwms --source meme_suite_meme --set 1 -c "$CONFIG"  # inspect MEME matrices
```

Example output:

```text
                                                           PWM summary
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ TF   ┃ Source          ┃ Motif ID             ┃ PWM source ┃ Length ┃ Window ┃ Bits  ┃ Sites (cached seq/total) ┃ Sites (matrix n) ┃ Site sets ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ lexA │ meme_suite_meme │ lexA_CTGTATAWAWWHACA │ matrix     │ 15     │ -      │ 16.13 │ 99/99                │ 99              │ -         │
│ cpxR │ meme_suite_meme │ cpxR_MANWWHTTTAM     │ matrix     │ 11     │ -      │ 5.47  │ 204/204              │ 204             │ -         │
└──────┴─────────────────┴──────────────────────┴────────────┴────────┴────────┴───────┴──────────────────────┴─────────────────┴───────────┘
```

Tip: add `--matrix` to print full PWM matrices or `--log-odds` for log‑odds matrices.

How to read the summary:

- **Bits**: quick specificity screen (very low bits often means a weak/degenerate PWM).
- **Length/Window**: confirm the width you intend to optimize against (use `pwm_window_lengths` if you want a shorter high‑information window).
- **Sites (cached seq/total)** vs **Sites (matrix n)**: cached coverage vs the count embedded in the matrix metadata (e.g., discovery output).

Logos are written under `runs/logos/catalog/<run_name>/` and include site counts (`n=...`) when available.

---

#### Optional: local motifs or alignment matrices

This demo already uses matrix mode. If you want to **prefer local motif matrices** or RegulonDB alignments over MEME/STREME discoveries, reorder `source_preference` like this:

```yaml
motif_store:
  pwm_source: matrix
  source_preference: [demo_local_meme, regulondb]
```

Then fetch local MEME motifs (if you have not already):

```bash
# Fetch local matrices for preference order
cruncher fetch motifs --source demo_local_meme --tf lexA --tf cpxR -c "$CONFIG"  # refresh local matrices
```

Note: not all RegulonDB transcription factors have alignment matrices. If `fetch motifs` fails with “alignment matrix is missing”, switch to `pwm_source: sites` and `fetch sites` instead.

---

### Select motif source + lock

Lockfiles resolve TF names to exact motif IDs and hashes for reproducibility.

```bash
# Write lockfile for reproducibility
cruncher lock -c "$CONFIG"  # write config.lock.json in workspace state
```

Example output:

```text
<workspace>/.cruncher/locks/config.lock.json
```

Think of `lock` as the “commit”: it freezes the exact motif IDs + hashes used for scoring. Re‑run it after discovery, after changing `source_preference`, or after changing any PWM windowing/trimming settings.

Lockfiles are required for `parse`, `sample`, and `targets status`. If you add new motifs (e.g., after `discover motifs`) or change `motif_store` preferences, re-run `lock` to refresh the pinned motif IDs and hashes.

---

### Inspect cached entries and targets (post-lock)

Quick checks (target status requires a lockfile):

```bash
cruncher targets list -c "$CONFIG"  # list targets per regulator set

# Readiness check (lock + cache)
cruncher targets status -c "$CONFIG"  # validate lock + cache alignment
```

Optional debugging / deeper visibility:

```bash
# Show one catalog entry
cruncher catalog show regulondb:RDBECOLITFC00214 -c "$CONFIG"  # inspect a specific entry

# Stats across candidates
cruncher targets stats -c "$CONFIG"  # show stats across candidates

# Fuzzy TF lookup
cruncher targets candidates --fuzzy -c "$CONFIG"  # fuzzy match TF names

# Dashboard: cache + runs
cruncher status -c "$CONFIG"  # show cache + run summaries
```

Note: with `combine_sites: true`, `targets status` reports merged site counts per TF (Sites shows merged seq/total) even when `pwm_source=matrix`; **Source** and **Motif ID** still reflect the locked matrix choice.

---

### Parse workflow (validate locked motifs)

Parse is a cheap sanity check that validates the locked PWMs load correctly and writes a manifest used by sampling.

```bash
# Validate lock + cached motifs
cruncher parse -c "$CONFIG"  # validate locked PWMs and write a parse manifest
```

Re-running parse with the same config + lock fingerprint reuses existing outputs and reports the location. Parse outputs live under `runs/parse/<run_name>/meta/`.

---

### Sample (MCMC optimization)

Each candidate dsDNA sequence is scored by scanning both strands for each PWM, taking the best match per TF, then optimizing the min/soft‑min across TFs (raise the weakest TF). The goal is not a single winner: **cruncher** returns a diverse elite set.

**Auto‑opt** runs short Gibbs + parallel tempering pilots to pick a robust optimizer/temperature ladder/budget so the final run is more performant. The chosen pilot is recorded in `runs/auto_opt/best_<run_group>.json`, and the final effective config is written to `runs/sample/<run_name>/meta/config_used.yaml`. For diagnostics and tuning guidance, see the [sampling + analysis guide](../guides/sampling_and_analysis.md).

This demo config sets `auto_opt.policy.allow_warn: true` so auto-opt will always pick a winner by the end of the configured budget levels, even if confidence is low (warnings are recorded). Set `allow_warn: false` to require a confidence-separated winner; if none emerges at the maximum configured budgets/replicates, auto-opt fails fast with guidance to increase `auto_opt.budget_levels` and/or `auto_opt.replicates`.

```bash
# Run sampling with auto-opt
cruncher sample -c "$CONFIG"  # run sampling with auto-opt pilots

# Run sampling without pilots
cruncher sample --no-auto-opt -c "$CONFIG"  # run sampling with configured optimizer only

# Run sampling with verbose logs
cruncher sample --verbose -c "$CONFIG"  # stream periodic status logs

# List runs for this workspace
cruncher runs list -c "$CONFIG"  # list run history

# Show latest run for set 1
cruncher runs latest --set-index 1 -c "$CONFIG"  # show most recent run

# Show best run for set 1
cruncher runs best --set-index 1 -c "$CONFIG"  # show best-scoring run
```

Example output (runs list, abridged):

```text
                               Runs
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Name                     ┃ Stage    ┃ Status    ┃ Created    ┃ Motifs ┃ Regulator set  ┃ PWM source ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ lexA-cpxR_20260114_155026_a5622e   │ sample   │ completed │ 2026-01-14 │ 2      │ lexA,cpxR       │ matrix     │
│ *lexA-cpxR_20260114_155007_9a3c1f  │ auto_opt │ completed │ 2026-01-14 │ 2      │ lexA,cpxR       │ matrix     │
│ lexA-cpxR_20260114_144521_29fb6f   │ parse    │ completed │ 2026-01-14 │ 2      │ lexA,cpxR       │ matrix     │
└──────────────────────────┴──────────┴───────────┴────────────┴────────┴────────────────┴────────────┘
```

Auto‑opt pilot runs also appear here under the `auto_opt` stage. If a run was interrupted and still shows `running`, mark it stale:

```bash
# Mark stale runs as aborted
cruncher runs clean --stale --older-than-hours 0 -c "$CONFIG"  # mark stale runs
```

For live progress, you can watch the run status in another terminal:

```bash
# Stream live run status
cruncher runs watch <run_name> -c "$CONFIG"  # watch status updates

# Stream status with plots
cruncher runs watch <run_name> --plot -c "$CONFIG"  # watch status + plots
```

---

#### Run artifacts + performance snapshot

Use `runs show` to inspect what a run produced:

```bash
# Inspect artifacts for a run
cruncher runs show 20260114_131314_c2b4ce -c "$CONFIG"  # list artifacts for a run
```

Key artifacts include `meta/config_used.yaml`, `artifacts/sequences.parquet`,
`artifacts/trace.nc` (if enabled), and `artifacts/elites.*`.

Runtime scales with `budget.draws`, `budget.tune`, and `budget.restarts` in the config; adjust them to match your runtime/quality budget.

---

#### Analyze + report

Use analysis to answer three questions:

1. **Did we raise the weakest TF?** (look at per‑TF scores and the min/soft‑min objective distribution)
2. **Did we keep diversity?** (unique sequences *up to reverse‑complement*, avoid a single collapsed cluster)
3. **Did the sampler mix?** (trace/tempering diagnostics; poor mixing usually means you should rerun with auto‑opt or increase budgets)

Motif overlap/co‑localization is allowed; use the overlap/structure plots to see *how* the PWMs are being jointly satisfied.

```bash
# Analyze latest sample run (default when analysis.runs is empty)
cruncher analyze -c "$CONFIG"
# Or pin to a specific run:
# cruncher analyze --run <run_name|run_dir> -c "$CONFIG"

# Write report from latest run
cruncher report --latest -c "$CONFIG"  # write report from latest run
```

Outputs land under `<workspace>/runs/sample/<run_name>/analysis/` with a
summary in `analysis/summary.json`.
This demo enables a small Tier‑0 plot set by default; use
`cruncher analyze --plots all` to generate the full suite.

If you're running via `pixi`, prefix those next-step commands with `pixi run cruncher --`.

For a compact diagnostics checklist and tuning guidance, see the
[sampling + analysis guide](../guides/sampling_and_analysis.md).

---

#### Optional: live analysis notebook

Replace the run directory below with your own sample run (see `cruncher runs latest`).

```bash
cruncher notebook <workspace>/runs/sample/lexA-cpxR_20260113_115749_e72283 --latest  # generate analysis notebook
```

Notebook files are written under `<run_dir>/analysis/notebooks/`.

---

@e-south
