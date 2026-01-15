## cruncher demo 1

**Jointly maximizing a sequence based on two TFs.**

This demo walks through generating DNA sequences that jointly resemble motifs for two TFs (e.g., LexA + CpxR). The process involves fetching binding sites, locking motif data (i.e., our "scorecards"), sampling sequence space, optimization, and analyzing results with the bundled `demo_basics_two_tf` workspace.

### Contents

- [Demo setup](#demo-setup)
- Core workflow
  - [Preview sources and inventories](#preview-sources-and-inventories)
  - [Fetch binding sites](#fetch-binding-sites)
  - [Fetch local DAP-seq motifs + binding sites](#fetch-local-dap-seq-motifs--binding-sites)
  - [Combine curated + DAP-seq sites for discovery](#combine-curated--dap-seq-sites-for-discovery)
- [Discover motifs from merged sites (MEME/STREME)](#discover-motifs-from-merged-sites-memestreme)
- [Optional: trim aligned PWMs to a fixed window](#optional-trim-aligned-pwms-to-a-fixed-window)
- [Inspect candidate PWMs + logos (pre-lock)](#inspect-candidate-pwms--logos-pre-lock)
- [Select motif source + lock](#select-motif-source--lock)
- [Inspect cached entries and targets (post-lock)](#inspect-cached-entries-and-targets-post-lock)
- [Parse workflow (validate locked motifs + render logos)](#parse-workflow-validate-locked-motifs--render-logos)
- [Sample (MCMC optimization)](#sample-mcmc-optimization)
  - [Run artifacts + performance snapshot](#run-artifacts--performance-snapshot)
  - [Analyze + report](#analyze--report)
- Optional workflows
  - [Optional: live analysis notebook](#optional-live-analysis-notebook)
  - [Optional: local motifs or alignment matrices](#optional-local-motifs-or-alignment-matrices)
  - [Optional: HT-only or combined site modes (RegulonDB)](#optional-ht-only-or-combined-site-modes-regulondb)

---

### Demo setup

- **Workspace**: `src/dnadesign/cruncher/workspaces/demo_basics_two_tf/`
- **Config**: `config.yaml`
- **Output root**: `runs/` (relative to the workspace; runs live under `runs/<stage>/<run_name>/`, where run_name includes the TF slug; `setN_` is added only when multiple regulator sets are configured)
- **Motif flow**: fetch sites → discover MEME/STREME motifs → lock/sample using those matrices
- **Path placeholders**: example outputs use `<workspace>` for the demo workspace root

```bash
# Option A: cd into the workspace
cd src/dnadesign/cruncher/workspaces/demo_basics_two_tf
CONFIG="$PWD/config.yaml"

# Option B: run from anywhere in the repo
CONFIG=src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml

# Choose a runner (uv is the default in this repo; pixi is optional).
cruncher() { uv run cruncher "$@"; }
# cruncher() { pixi run cruncher -- "$@"; }
# From here on, commands use $CONFIG for clarity; if you're in the workspace, you can omit --config.
```

Smoke test for external dependencies (MEME Suite):

```bash
cruncher doctor -c "$CONFIG"
```

If it reports missing tools, install MEME Suite (pixi or system install) or set `motif_discovery.tool_path`; see the [MEME Suite guide](../guides/meme_suite.md). Local demo motifs live at `data/local_motifs/` (DAP‑seq MEME files from [this](https://www.nature.com/articles/s41592-021-01312-2) study).

---

### Preview sources and inventories

List the sources registered by the demo config:

```bash
cruncher sources list -c "$CONFIG"
```

Optional deep‑dive (inventory / datasets):

```bash
cruncher sources info demo_local_meme -c "$CONFIG"
cruncher sources info regulondb -c "$CONFIG"
cruncher sources summary --source regulondb --scope remote --remote-limit 20 -c "$CONFIG"
cruncher sources summary --source regulondb --scope cache -c "$CONFIG"
cruncher sources summary --source demo_local_meme --scope cache -c "$CONFIG"
```
Use `--remote-limit` for large inventories. `sources datasets` or `fetch sites --dry-run` are useful when you need HT dataset coverage.

---

### Fetch binding sites

This demo prefers MEME/STREME‑discovered motifs for optimization (`pwm_source: matrix`) but still fetches sites because discovery runs on sites (and because you may want to compare site‑derived PWMs). Use `--dataset-id` to pin HT datasets; if a dataset returns zero TF‑binding records, **cruncher** fails fast so you can choose a different dataset. `motif_store.site_window_lengths` only affects site‑derived PWMs unless you set `motif_discovery.window_sites=true`.

```bash
cruncher fetch sites --tf lexA --tf cpxR --update -c "$CONFIG"
```

Curated site sets do not carry a dataset or method (those fields are only populated for HT datasets). Repeat runs will report “No new sites cached” unless you pass `--update`.

---

### Fetch local DAP-seq motifs + binding sites

Local DAP-seq MEME files live under `data/local_motifs/` in the workspace. Fetch the motif matrices and the MEME BLOCKS training sites:

```bash
cruncher fetch motifs --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
```

---

---

### Combine curated + DAP-seq sites for discovery

This demo config already merges site sets per TF (`combine_sites: true`) so MEME/STREME sees all available sites. If you want to keep sources separate, set `combine_sites: false` and re-lock:

```yaml
motif_store:
  combine_sites: false
```

```bash
cruncher lock -c "$CONFIG"
cruncher targets status -c "$CONFIG"
```

Note: with `combine_sites: true`, `targets status` reports merged site counts per TF
(Sites shows merged seq/total) even when `pwm_source=matrix`; **Source** and **Motif ID**
still reflect the locked matrix choice. Tip: `catalog pwms --set 1` shows the preferred
matrix source; to use only local DAP‑seq training sites in discovery, set
`site_kinds: ["meme_blocks"]`.

---

### Discover motifs from merged sites (MEME/STREME)

If you fetched binding sites from multiple sources and want a single aligned PWM per TF (e.g., to reconcile different site lengths), run MEME Suite on the combined site sets. This creates new motif matrices under `motif_discovery.source_id` (default `meme_suite_streme`, or override with `--source-id`).

Discovery notes:

- Discovery uses cached binding sites (variable lengths are OK), regardless of `motif_store.pwm_source`.
- Width bounds default to the per‑TF site length range; override with explicit `minw/maxw` if needed.
- Runs write under `.cruncher/<workspace>/discoveries/` and ingest motifs into the catalog, replacing prior entries for the same TF/source unless you pass `--keep-existing`.
- Discovery uses raw sites by default; set `motif_discovery.window_sites=true` to pre‑window.

```bash
cruncher discover check -c "$CONFIG"
cruncher discover motifs --tf lexA --tf cpxR --tool streme --source-id meme_suite_streme -c "$CONFIG"
cruncher catalog list --source meme_suite_streme -c "$CONFIG"
```

Example output:

```bash
INFO lexA: using site-length bounds for discovery (minw=15, maxw=22, site lengths 15-22).
INFO cpxR: using site-length bounds for discovery (minw=11, maxw=21, site lengths 11-21).
                                                                        Motif discovery
┏━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ TF   ┃ Tool   ┃ Motif ID                                      ┃ Length ┃ Output                                                                              ┃
┡━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ lexA │ streme │ meme_suite_streme:lexA_1-CTGTATAWAWWHACAGT    │ 17     │ <workspace>/.cruncher/demo_basics_two_tf/discoveries/discover_lexA_20260113_114748_ │
│ cpxR │ streme │ meme_suite_streme:cpxR_1-MTTTACAYWWMTTTACAWWW │ 20     │ <workspace>/.cruncher/demo_basics_two_tf/discoveries/discover_cpxR_20260113_114748_ │
└──────┴────────┴───────────────────────────────────────────────┴────────┴─────────────────────────────────────────────────────────────────────────────────────┘
```

The “Motif discovery” table reports the tool used, the motif ID, and the discovered width. **Tip:** if each sequence represents one site, prefer MEME and set `--meme-mod oops`:

```bash
cruncher discover motifs --tf lexA --tf cpxR --tool meme --meme-mod oops --source-id meme_suite_meme -c "$CONFIG"
```

This demo config already prefers MEME motifs for optimization: `pwm_source: matrix` with `source_preference: [meme_suite_meme, meme_suite_streme, ...]`. After discovery, continue to “Select motif source + lock” below so parse/sample use the new motifs.

---

### Optional: trim aligned PWMs to a fixed window

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
cruncher catalog pwms --set 1 -c "$CONFIG"
cruncher catalog logos --set 1 -c "$CONFIG"

# Optional: compare MEME vs STREME sources directly
cruncher catalog pwms --source meme_suite_streme --set 1 -c "$CONFIG"
cruncher catalog pwms --source meme_suite_meme --set 1 -c "$CONFIG"
```

Example output:

```bash
                                                 PWM summary
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┓
┃ TF   ┃ Source          ┃ Motif ID             ┃ PWM source ┃ Length ┃ Window ┃ Bits  ┃ n sites ┃ Site sets ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━┩
│ lexA │ meme_suite_meme │ lexA_CTGTATAWAWWHACA │ matrix     │ 15     │ -      │ 16.13 │ 99      │ -         │
│ cpxR │ meme_suite_meme │ cpxR_MANWWHTTTAM     │ matrix     │ 11     │ -      │ 5.47  │ 204     │ -         │
└──────┴─────────────────┴──────────────────────┴────────────┴────────┴────────┴───────┴─────────┴───────────┘
```

Tip: add `--matrix` to print full PWM matrices or `--log-odds` for log‑odds matrices.

Use **Bits** as a quick quality screen. **n sites** is populated when the source tracks
site counts (e.g., discovery runs, site‑derived PWMs); otherwise it is `-`. Low counts or
very low information content are signals to switch sources, combine sites, or raise
`motif_store.min_sites_for_pwm`. To constrain PWM length for optimization, set
`motif_store.pwm_window_lengths`; the chosen window appears in **Window**.

Logos are written under `runs/logos/catalog/<run_name>/` and include site counts (`n=...`)
when available. Re-running with unchanged inputs reuses existing outputs. Use `--source` on
`catalog pwms/logos` to compare tools directly.

---

### Select motif source + lock

Lockfiles resolve TF names to exact motif IDs and hashes for reproducibility.

```bash
cruncher lock -c "$CONFIG"
```

Example output:

```bash
<workspace>/.cruncher/demo_basics_two_tf/locks/config.lock.json
```

Lockfiles are required for `parse`, `sample`, and `targets status`. If you add new motifs (e.g., after `discover motifs`) or change `motif_store` preferences, re-run `lock` to refresh the pinned motif IDs and hashes.

---

### Inspect cached entries and targets (post-lock)

Optional deeper visibility into what’s cached and what will be used (target status requires a lockfile):

```bash
cruncher catalog list -c "$CONFIG"                             # list cached motifs/sites
cruncher catalog show regulondb:RDBECOLITFC00214 -c "$CONFIG"  # show one catalog entry
cruncher targets list -c "$CONFIG"                             # list TF targets by set
cruncher targets status -c "$CONFIG"                           # readiness check (lock + cache)
cruncher targets stats -c "$CONFIG"                            # stats across candidates
cruncher targets candidates --fuzzy -c "$CONFIG"               # fuzzy TF lookup
cruncher status -c "$CONFIG"                                   # dashboard: cache + runs
```

---

### Parse workflow (validate locked motifs + render logos)

```bash
cruncher parse -c "$CONFIG"
```

Re-running parse with the same config + lock fingerprint reuses existing outputs and reports the location. Logos are written under `runs/logos/parse/<run_id>/`.

---

### Sample (MCMC optimization)

Auto‑optimize is enabled by default (short Gibbs + PT pilots). The selected pilot is recorded in `runs/auto_opt/best_<run_group>.json` and marked with a leading `*` in `cruncher runs list`. The final chosen config is always written to `runs/sample/<run_name>/meta/config_used.yaml`. Use `--no-auto-opt` to skip pilots and force the configured optimizer. For diagnostics and tuning guidance, see the [sampling + analysis guide](../guides/sampling_and_analysis.md).

```bash
cruncher sample -c "$CONFIG"
cruncher sample --no-auto-opt -c "$CONFIG"
cruncher sample --verbose -c "$CONFIG"
cruncher runs list -c "$CONFIG"
cruncher runs latest --set-index 1 -c "$CONFIG"
cruncher runs best --set-index 1 -c "$CONFIG"
```

Example output (runs list, abridged):

```bash
                               Runs
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Name                     ┃ Stage    ┃ Status    ┃ Created    ┃ Motifs ┃ Regulator set  ┃ PWM source ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ lexA-cpxR_20260114_155026_a5622e   │ sample   │ completed │ 2026-01-14 │ 2      │ lexA,cpxR       │ matrix     │
│ *lexA-cpxR_20260114_155007_9a3c1f  │ auto_opt │ completed │ 2026-01-14 │ 2      │ lexA,cpxR       │ matrix     │
│ lexA-cpxR_20260114_144521_29fb6f   │ parse    │ completed │ 2026-01-14 │ 2      │ lexA,cpxR       │ matrix     │
└──────────────────────────┴──────────┴───────────┴────────────┴────────┴────────────────┴────────────┘
```

Auto‑opt pilot runs also appear here under the `auto_opt` stage.
If a run was interrupted and still shows `running`, mark it stale:

```bash
cruncher runs clean --stale --older-than-hours 0 -c "$CONFIG"
```

For live progress, you can watch the run status in another terminal:

```bash
cruncher runs watch <run_name> -c "$CONFIG"
cruncher runs watch <run_name> --plot -c "$CONFIG"
```

---

### Run artifacts + performance snapshot

Use `runs show` to inspect what a run produced:

```bash
cruncher runs show 20260114_131314_c2b4ce -c "$CONFIG"
```

Key artifacts include `meta/config_used.yaml`, `artifacts/sequences.parquet`,
`artifacts/trace.nc` (if enabled), and `artifacts/elites.*`.

Runtime scales with `budget.draws`, `budget.tune`, and `budget.restarts` in the config; adjust them to match your runtime/quality budget.

---

### Analyze + report

```bash
cruncher analyze --latest -c "$CONFIG"
cruncher report --latest -c "$CONFIG"
```

Outputs land under `<workspace>/runs/sample/<run_name>/analysis/` with a
summary in `analysis/meta/summary.json`.

If you're running via `pixi`, prefix those next-step commands with `pixi run cruncher --`.

For a compact diagnostics checklist and tuning guidance, see the
[sampling + analysis guide](../guides/sampling_and_analysis.md).

---

### Optional: live analysis notebook

```bash
cruncher notebook <workspace>/runs/sample/lexA-cpxR_20260113_115749_e72283 --latest
```

Notebook files are written under `<run_dir>/analysis/notebooks/`.

---

### Optional: local motifs or alignment matrices

This demo already uses matrix mode. If you want to **prefer local motif matrices**
or RegulonDB alignments over MEME/STREME discoveries, reorder `source_preference`
like this:

```yaml
motif_store:
  pwm_source: matrix
  source_preference: [demo_local_meme, regulondb]
```

Then fetch local MEME motifs (if you have not already):

```bash
cruncher fetch motifs --source demo_local_meme --tf lexA --tf cpxR -c "$CONFIG"
```

Note: not all RegulonDB regulons ship alignment matrices. If `fetch motifs` fails
with “alignment matrix is missing”, temporarily switch to `pwm_source: sites` and
`fetch sites` instead.

---

### Optional: HT-only or combined site modes (RegulonDB)

You can restrict `pwm_source: sites` to curated or HT sites, or combine them:

```yaml
motif_store:
  pwm_source: sites
  site_kinds: ["curated"]          # curated-only
  # site_kinds: ["meme_blocks"]    # local DAP-seq training sites only
  # site_kinds: ["ht_tfbinding"]   # HT-only (RegulonDB)
  # combine_sites: true            # combine across sources when site_kinds is null
```

Tip: if HT datasets return peaks without sequences, hydration uses NCBI by default
(`ingest.genome_source=ncbi`). To run offline, provide a local FASTA via
`--genome-fasta` or `ingest.genome_fasta`.

If HT site lengths vary, use `cruncher targets stats` and set -c "$CONFIG"
`motif_store.site_window_lengths` per TF or dataset before building site-derived PWMs (sites mode).
Discovery uses raw sites unless `motif_discovery.window_sites=true`.

---

@e-south
