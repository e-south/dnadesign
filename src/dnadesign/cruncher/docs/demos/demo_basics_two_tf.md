## cruncher demo 1

**Jointly maximizing a sequence based on two TFs.**

This demo walks through generating DNA sequences that jointly resemble motifs for two TFs (e.g., LexA + CpxR). The process involves fetching binding sites, locking motif/PWM data (i.e., our "scorecards"), sampling sequence space, optimization, and analyzing results.

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

### Demo setup

- **Workspace**: `src/dnadesign/cruncher/workspaces/demo_basics_two_tf/`
- **Config**: `config.yaml`
- **Output root**: `runs/` (relative to the workspace; runs live under `runs/<stage>/<run_name>/`, where run_name includes the TF slug; `setN_` is added only when multiple regulator sets are configured)
- **Catalog cache**: `src/dnadesign/cruncher/.cruncher/` (shared across workspaces by default; override with `motif_store.catalog_root`)
- **Motif flow**: fetch sites + local motifs → (optional discovery) → inspect/select → lock/sample using chosen matrices
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

Smoke test for external dependencies:

```bash
# Verify MEME Suite and external tools
cruncher doctor -c "$CONFIG"  # check external dependencies
```

If it reports missing tools, install MEME Suite (pixi or system install) or set `motif_discovery.tool_path`; see the [MEME Suite guide](../guides/meme_suite.md). Local demo motifs live at `data/local_motifs/` (DAP‑seq MEME files from [this](https://www.nature.com/articles/s41592-021-01312-2) study).

---

### Preview sources and inventories

List the sources registered by the demo config:

```bash
# List configured sources
cruncher sources list -c "$CONFIG"  # list configured sources
```

Optional deep‑dive (inventory / datasets):

```bash
# Inspect local demo source metadata
cruncher sources info demo_local_meme -c "$CONFIG"  # inspect demo source metadata

# Inspect RegulonDB source metadata
cruncher sources info regulondb -c "$CONFIG"  # inspect RegulonDB metadata

# Sample remote inventory
cruncher sources summary --source regulondb --scope remote --remote-limit 20 -c "$CONFIG"  # sample remote inventory
```
Use `--remote-limit` for large inventories. `sources datasets` or `fetch sites --dry-run` are useful when you need HT dataset coverage.

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

We optimize with MEME/STREME‑discovered matrices (`pwm_source: matrix`), but discovery is site‑driven, so we fetch sites too. This step hits the network‑backed RegulonDB source; if you’re offline, skip it and continue with local motifs.

```bash
# Fetch curated sites from RegulonDB
cruncher fetch sites --tf lexA --tf cpxR --update -c "$CONFIG"  # fetch curated RegulonDB sites
```

Curated site sets do not carry a dataset or method (those fields are only populated for HT datasets). Repeat runs will report “No new sites cached” unless you pass `--update`.

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

Confirm what you have cached before discovery/lock:

```bash
# Verify local cache entries
cruncher sources summary --source demo_local_meme --scope cache -c "$CONFIG"  # check local cache

# Verify RegulonDB cache entries
cruncher sources summary --source regulondb --scope cache -c "$CONFIG"  # check RegulonDB cache

# List cached motifs/sites
cruncher catalog list -c "$CONFIG"  # list cached motifs and sites
```

If a source shows zero entries, you likely haven’t fetched from it yet (cache scope only shows what’s already cached).

---

### Combine curated + DAP-seq sites for discovery

This demo config already merges site sets per TF (`combine_sites: true`) so MEME/STREME sees all available sites. If you want to keep sources separate, set `combine_sites: false` and re-lock:

```yaml
motif_store:
  combine_sites: false
```

After changing `combine_sites` or `site_kinds`, re-run `cruncher lock` so parse/sample use the updated site sets. Tip: `catalog pwms --set 1` shows the preferred matrix source; to use only local DAP‑seq training sites in discovery, set `site_kinds: ["meme_blocks"]`.

---

### Discover motifs from merged sites (MEME/STREME)

If you fetched binding sites from multiple sources and want a single aligned PWM per TF (e.g., to reconcile different site lengths), run MEME Suite on the combined site sets. This creates new motif matrices under `motif_discovery.source_id` (default `meme_suite_streme`, or override with `--source-id`).

Discovery notes:

- Discovery uses cached binding sites (variable lengths are OK), regardless of `motif_store.pwm_source`.
- Width bounds default to the per‑TF site length range; override with explicit `minw/maxw` if needed.
- Runs write under `src/dnadesign/cruncher/.cruncher/discoveries/` (the shared catalog root) and ingest motifs into the catalog, replacing prior entries for the same TF/source unless you pass `--keep-existing`.
- Discovery uses raw sites by default; set `motif_discovery.window_sites=true` to pre‑window.

```bash
# Verify discovery prerequisites
cruncher discover check -c "$CONFIG"  # verify MEME Suite availability + inputs

# Run STREME discovery
cruncher discover motifs --tf lexA --tf cpxR --tool streme --source-id meme_suite_streme -c "$CONFIG"  # discover PWMs from sites

# Confirm discovered motifs in catalog
cruncher catalog list --source meme_suite_streme -c "$CONFIG"  # verify new motif entries
```

Example output:

```text
INFO lexA: using site-length bounds for discovery (minw=15, maxw=22, site lengths 15-22).
INFO cpxR: using site-length bounds for discovery (minw=11, maxw=21, site lengths 11-21).
                                                                        Motif discovery
┏━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ TF   ┃ Tool   ┃ Motif ID                                      ┃ Length ┃ Output                                                                              ┃
┡━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ lexA │ streme │ meme_suite_streme:lexA_1-CTGTATAWAWWHACAGT    │ 17     │ <repo>/src/dnadesign/cruncher/.cruncher/discoveries/discover_lexA_20260113_114748_ │
│ cpxR │ streme │ meme_suite_streme:cpxR_1-MTTTACAYWWMTTTACAWWW │ 20     │ <repo>/src/dnadesign/cruncher/.cruncher/discoveries/discover_cpxR_20260113_114748_ │
└──────┴────────┴───────────────────────────────────────────────┴────────┴─────────────────────────────────────────────────────────────────────────────────────┘
```

The “Motif discovery” table reports the tool used, the motif ID, and the discovered width. **Tip:** if each sequence represents one site, prefer MEME and set `--meme-mod oops`:

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

Use **Bits** as a quick quality screen. **Sites (cached seq/total)** reflects cached site coverage for the source (if any), while **Sites (matrix n)** is the site count embedded in the matrix metadata (e.g., discovery runs, site‑derived PWMs). If either is low, consider switching sources, combining sites, or raising `motif_store.min_sites_for_pwm`. To constrain PWM length for optimization, set `motif_store.pwm_window_lengths`; the chosen window appears in **Window**.

Logos are written under `runs/logos/catalog/<run_name>/` and include site counts (`n=...`) when available. Re-running with unchanged inputs reuses existing outputs. Use `--source` on `catalog pwms/logos` to compare tools directly.

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

Lockfiles are required for `parse`, `sample`, and `targets status`. If you add new motifs (e.g., after `discover motifs`) or change `motif_store` preferences, re-run `lock` to refresh the pinned motif IDs and hashes.

---

### Inspect cached entries and targets (post-lock)

Optional deeper visibility into what’s cached and what will be used (target status requires a lockfile):

```bash
# List cached motifs/sites
cruncher catalog list -c "$CONFIG"  # list catalog entries

# Show one catalog entry
cruncher catalog show regulondb:RDBECOLITFC00214 -c "$CONFIG"  # inspect a specific entry

# List TF targets by set
cruncher targets list -c "$CONFIG"  # list targets per regulator set

# Readiness check (lock + cache)
cruncher targets status -c "$CONFIG"  # validate lock + cache alignment

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

```bash
# Validate lock + cached motifs
cruncher parse -c "$CONFIG"  # validate locked PWMs and write a parse manifest
```

Re-running parse with the same config + lock fingerprint reuses existing outputs and reports the location. Parse outputs live under `runs/parse/<run_name>/meta/` and include a run manifest for traceability. Use `cruncher catalog logos` if you want PWM logos.

---

### Sample (MCMC optimization)

Auto‑optimize is enabled by default (short Gibbs + PT pilots). The selected pilot is recorded in `runs/auto_opt/best_<run_group>.json` and marked with a leading `*` in `cruncher runs list`. The final chosen config is always written to `runs/sample/<run_name>/meta/config_used.yaml`. Use `--no-auto-opt` to skip pilots and force the configured optimizer. For diagnostics and tuning guidance, see the [sampling + analysis guide](../guides/sampling_and_analysis.md).

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

```bash
# Analyze latest sample run
cruncher analyze --latest -c "$CONFIG"  # analyze latest sample run

# Write report from latest run
cruncher report --latest -c "$CONFIG"  # write report from latest run
```

Outputs land under `<workspace>/runs/sample/<run_name>/analysis/` with a
summary in `analysis/meta/summary.json`.

If you're running via `pixi`, prefix those next-step commands with `pixi run cruncher --`.

For a compact diagnostics checklist and tuning guidance, see the
[sampling + analysis guide](../guides/sampling_and_analysis.md).

---

#### Optional: live analysis notebook

```bash
cruncher notebook <workspace>/runs/sample/lexA-cpxR_20260113_115749_e72283 --latest  # generate analysis notebook
```

Notebook files are written under `<run_dir>/analysis/notebooks/`.

---

@e-south
