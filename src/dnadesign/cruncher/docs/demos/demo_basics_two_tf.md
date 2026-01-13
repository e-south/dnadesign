## cruncher demo 1

**Jointly maximizing a sequence based on two TFs.**

This demo walks through generating DNA sequences that jointly resemble motifs for two TFs (e.g., LexA + CpxR). The process involves fetching binding sites, locking motif data (i.e., our "scorecards"), sampling sequence space, optimization, and analyzing results with the bundled `demo_basics_two_tf` workspace.

### Contents

- [Demo setup](#demo-setup)
- Core workflow
  - [Preview sources and inventories](#preview-sources-and-inventories)
  - [Fetch binding sites](#fetch-binding-sites)
  - [Fetch local DAP-seq motifs + binding sites](#fetch-local-dap-seq-motifs--binding-sites)
  - [Summarize cached regulators by source](#summarize-cached-regulators-by-source)
  - [Combine curated + DAP-seq sites for discovery](#combine-curated--dap-seq-sites-for-discovery)
  - [Discover motifs from merged sites (MEME/STREME)](#discover-motifs-from-merged-sites-memestreme)
  - [Compare MEME vs STREME outputs (logos)](#compare-meme-vs-streme-outputs-logos)
  - [Optional: trim aligned PWMs to a fixed window](#optional-trim-aligned-pwms-to-a-fixed-window)
  - [Select motif source + lock](#select-motif-source--lock)
  - [Inspect cached entries and targets](#inspect-cached-entries-and-targets)
  - [Compute PWMs + inspect information content](#compute-pwms--inspect-information-content)
  - [Render PWM logos (catalog)](#render-pwm-logos-catalog)
  - [Parse workflow (validate motifs + render logos)](#parse-workflow-validate-motifs--render-logos)
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
- **Output root**: `runs/` (relative to the workspace; runs are grouped by regulator set under `runs/<stage>/setN_<tfs>/...`, plus shared `runs/logos/`)
- **Motif flow**: fetch sites → discover MEME/STREME motifs → lock/sample using those matrices
- **Path placeholders**: example outputs use `<workspace>` for the demo workspace root

```bash
# Option A: cd into the workspace
cd src/dnadesign/cruncher/workspaces/demo_basics_two_tf
CONFIG="$PWD/config.yaml"

# Option B: run from anywhere in the repo
CONFIG=src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml

# Choose a runner (pixi recommended when using MEME Suite).
cruncher() { pixi run cruncher -- "$@"; }
# cruncher() { uv run cruncher "$@"; }

# Optional: widen tables to avoid truncation in rich output.
export COLUMNS=160

# If you haven't installed system tools yet (from repo root):
# pixi install

# From here on, commands use $CONFIG for clarity; if you're in the workspace, you can omit --config.
```

Smoke test for external dependencies (MEME Suite):

```bash
cruncher doctor -c "$CONFIG"
```

If it reports missing tools, install MEME Suite (pixi recommended) or set `motif_discovery.tool_path`; see the [MEME Suite guide](../guides/meme_suite.md). Local demo motifs live at `data/local_motifs/` (DAP‑seq MEME files from [this](https://www.nature.com/articles/s41592-021-01312-2) study).

---

### Preview sources and inventories

List the sources registered by the demo config:

```bash
cruncher sources list -c "$CONFIG"
```

Example output:

```bash
                            Sources
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Source          ┃ Description                                ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ demo_local_meme │ Demo DAP-seq motifs + sites (local MEME)   │
│ regulondb       │ RegulonDB datamarts GraphQL (curated + HT) │
└─────────────────┴────────────────────────────────────────────┘
```

Inspect capabilities or inventory as needed:

```bash
cruncher sources info demo_local_meme -c "$CONFIG"
cruncher sources info regulondb -c "$CONFIG"
cruncher sources summary --source regulondb --scope remote --remote-limit 20 -c "$CONFIG"
```

Notes:

- `motifs:*` are motif matrix inventories; `sites:list` means binding-site sequences are available.
- If remote inventories are large use `--remote-limit`.
- Use `sources datasets` or `fetch sites --dry-run` when you need HT dataset coverage.

---

### Fetch binding sites

This demo prefers MEME/STREME‑discovered motifs for optimization (`pwm_source: matrix`) but still fetches sites because discovery runs on sites (and because you may want to compare site‑derived PWMs). Use `--dataset-id` to pin HT datasets; if a dataset returns zero TF‑binding records, **cruncher** fails fast so you can choose a different dataset.

Note: `motif_store.site_window_lengths` only affects site‑derived PWMs (or discovery if you enable `motif_discovery.window_sites=true`).

```bash
cruncher fetch sites --tf lexA --tf cpxR --update -c "$CONFIG"
```

Example output (abridged, INFO log level):

```bash
16:04:05 INFO     Fetching binding sites from regulondb for TFs=['lexA', 'cpxR'] motif_ids=[]
         INFO     Fetching binding sites for TF 'lexA'
16:04:06 INFO     Fetching binding sites for TF 'cpxR'
         WARNING  Skipping curated site RDBECOLIBSC04560: invalid curated binding-site coordinates
                                         Fetched binding-site sets
┏━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ TF   ┃ Source    ┃ Motif ID         ┃ Kind    ┃ Dataset ┃ Method ┃ Sites ┃ Total ┃ Mean len ┃ Updated    ┃
┡━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ cpxR │ regulondb │ RDBECOLITFC00170 │ curated │ -       │ -      │ 154   │ 154   │ 15.3     │ 2026-01-13 │
│ lexA │ regulondb │ RDBECOLITFC00214 │ curated │ -       │ -      │ 49    │ 49    │ 19.5     │ 2026-01-13 │
└──────┴───────────┴──────────────────┴─────────┴─────────┴────────┴───────┴───────┴──────────┴────────────┘
```

Curated site sets do not carry a dataset or method (those fields are only populated for HT datasets). Repeat runs will report “No new sites cached” unless you pass `--update`.

---

### Fetch local DAP-seq motifs + binding sites

Local DAP-seq MEME files live under `data/local_motifs/` in the workspace. Fetch the motif matrices and the MEME BLOCKS training sites:

```bash
cruncher fetch motifs --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
```

Example output:

```bash
                             Fetched motifs
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ TF   ┃ Source          ┃ Motif ID ┃ Length ┃ Matrix     ┃ Updated    ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ cpxR │ demo_local_meme │ cpxR     │ 21     │ yes (file) │ 2026-01-13 │
│ lexA │ demo_local_meme │ lexA     │ 22     │ yes (file) │ 2026-01-13 │
└──────┴─────────────────┴──────────┴────────┴────────────┴────────────┘
```

Example output (local sites):

```bash
                                         Fetched binding-site sets
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ TF   ┃ Source          ┃ Motif ID ┃ Kind        ┃ Dataset ┃ Method ┃ Sites ┃ Total ┃ Mean len ┃ Updated    ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ cpxR │ demo_local_meme │ cpxR     │ meme_blocks │ -       │ -      │ 50    │ 50    │ 21.0     │ 2026-01-13 │
│ lexA │ demo_local_meme │ lexA     │ meme_blocks │ -       │ -      │ 50    │ 50    │ 22.0     │ 2026-01-13 │
└──────┴─────────────────┴──────────┴──────────━──┴─────────┴────────┴───────┴───────┴──────────┴────────────┘
```

---

### Summarize cached regulators by source

Summarize cached regulators:

```bash
cruncher sources summary --source demo_local_meme --scope cache -c "$CONFIG"
cruncher sources summary --source regulondb --scope cache -c "$CONFIG"
```

This is a quick sanity check on cached motifs/sites per source before discovery.

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
Tip: `catalog pwms --set 1` will show the currently preferred matrix source. To use only
the local DAP-seq training sites in discovery, set `site_kinds: ["meme_blocks"]`.

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

### Compare MEME vs STREME outputs (logos)

If you want to compare both tools, run discovery into distinct sources and render both logos.

```bash
cruncher discover motifs --tf lexA --tf cpxR --tool streme --source-id meme_suite_streme -c "$CONFIG"
cruncher discover motifs --tf lexA --tf cpxR --tool meme --meme-mod oops --source-id meme_suite_meme -c "$CONFIG"

# Compare matrices and logos per tool
cruncher catalog pwms --source meme_suite_streme --set 1 -c "$CONFIG"
cruncher catalog pwms --source meme_suite_meme --set 1 -c "$CONFIG"
cruncher catalog logos --source meme_suite_streme --set 1 -c "$CONFIG"
cruncher catalog logos --source meme_suite_meme --set 1 -c "$CONFIG"
```
After comparing, re-run `cruncher lock -c "$CONFIG"`; this demo prefers MEME-derived motifs first so sampling/analysis uses MEME when both are available. To switch, reorder `motif_store.source_preference` in `config.yaml`.

---

### Optional: trim aligned PWMs to a fixed window

If optimization needs a shorter PWM, set `pwm_window_lengths` to select the highest-information contiguous window before sampling:

```yaml
motif_store:
  pwm_window_lengths:
    lexA: 15
    cpxR: 15
```

```bash
cruncher catalog pwms --set 1 -c "$CONFIG"
```

Then proceed to `lock` and `sample` below to run optimization with the aligned (and optionally trimmed) PWMs.

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

### Inspect cached entries and targets

Use these when you want deeper visibility into what’s cached and what will be used:

```bash
cruncher catalog list -c "$CONFIG"
cruncher catalog show regulondb:RDBECOLITFC00214 -c "$CONFIG"
cruncher targets list -c "$CONFIG"
cruncher targets status -c "$CONFIG"
cruncher targets stats -c "$CONFIG"
cruncher targets candidates --fuzzy -c "$CONFIG"
cruncher status -c "$CONFIG"
```

---

### Compute PWMs + inspect information content

`catalog pwms` builds PWMs from cached sites (or matrices) and reports information content in bits. Site-derived PWMs use Biopython with configurable pseudocounts (`motif_store.pseudocounts`):

```bash
cruncher catalog pwms --set 1 -c "$CONFIG"
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

---

### Render PWM logos (catalog)

```bash
cruncher catalog logos --set 1 -c "$CONFIG"
```

Example output (paths shortened):

```bash
Rendered PWM logos
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ TF   ┃ Source          ┃ Motif ID             ┃ Length ┃ Bits  ┃ Output                                                               ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ lexA │ meme_suite_meme │ lexA_CTGTATAWAWWHACA │ 15     │ 16.13 │ <workspace>/runs/logos/catalog/set1_lexA-cpxR_20260113_114842_d986f5 │
│ cpxR │ meme_suite_meme │ cpxR_MANWWHTTTAM     │ 11     │ 5.47  │ <workspace>/runs/logos/catalog/set1_lexA-cpxR_20260113_114842_d986f5 │
└──────┴─────────────────┴──────────────────────┴────────┴───────┴──────────────────────────────────────────────────────────────────────┘
Logos saved to <workspace>/runs/logos/catalog/set1_lexA-cpxR_20260113_114842_d986f5
```
Logos include the site count (`n=...`) in the subtitle when available (e.g., MEME/STREME discoveries or site-derived PWMs).

---

### Parse workflow (validate motifs + render logos)

```bash
cruncher parse -c "$CONFIG"
```

Logos are written under `runs/logos/parse/<run_id>/`. This demo uses `pwm_source=matrix`, so the subtitle shows the adapter source and matrix origin (alignment/meme/streme/file). If you switch to `pwm_source=sites`, the subtitle will instead show how many site sets were merged, the contributing sources, and the site-kind mix.

---

### Sample (MCMC optimization)

Auto‑optimize is enabled by default (short Gibbs + PT pilots). Use `--no-auto-opt` to skip pilots and force the configured optimizer. For diagnostics and tuning guidance, see the [sampling + analysis guide](../guides/sampling_and_analysis.md).

```bash
cruncher sample -c "$CONFIG"
cruncher sample --no-auto-opt -c "$CONFIG"
cruncher runs list -c "$CONFIG"
cruncher runs latest --set-index 1 -c "$CONFIG"
cruncher runs best --set-index 1 -c "$CONFIG"
```

Example output (runs list, abridged):

```bash
                                                                     Runs
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Name                                         ┃ Stage  ┃ Status    ┃ Created                          ┃ Motifs ┃ Regulator set  ┃ PWM source ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ set1_lexA-cpxR_20260113_115749_e72283 │ sample │ completed │ 2026-01-13T16:57:49.511391+00:00 │ 2      │ set1:lexA,cpxR │ matrix     │
│ set1_lexA-cpxR_20260113_114853_a44d99 │ parse  │ completed │ 2026-01-13T16:48:53.807009+00:00 │ 2      │ set1:lexA,cpxR │ matrix     │
└──────────────────────────────────────────────┴────────┴───────────┴──────────────────────────────────┴────────┴────────────────┴────────────┘
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
cruncher runs show set1_lexA-cpxR_20260113_115749_e72283 -c "$CONFIG"
```

Example output (abridged):

```bash
run: set1_lexA-cpxR_20260113_115749_e72283
stage: sample
status: completed
created_at: 2026-01-13T16:57:49.511391+00:00
motif_count: 2
regulator_set: {'index': 1, 'tfs': ['lexA', 'cpxR']}
pwm_source: matrix
run_dir: <workspace>/runs/sample/set1_lexA-cpxR/set1_lexA-cpxR_20260113_115749_e72283
artifacts:
┏━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Stage  ┃ Type     ┃ Label                                  ┃ Path              ┃
┡━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ sample │ config   │ Resolved config (config_used.yaml)     │ meta/config_used.yaml        │
│ sample │ trace    │ Trace (NetCDF)                         │ artifacts/trace.nc            │
│ sample │ table    │ Sequences with per-TF scores (Parquet) │ artifacts/sequences.parquet   │
│ sample │ table    │ Elite sequences (Parquet)              │ artifacts/elites.parquet      │
│ sample │ json     │ Elite sequences (JSON)                 │ artifacts/elites.json         │
│ sample │ metadata │ Elite metadata (YAML)                  │ artifacts/elites.yaml         │
└────────┴──────────┴────────────────────────────────────────┴───────────────────┘
```

Runtime scales with `draws`, `tune`, and `chains` in the config; adjust them to match your runtime/quality budget.

---

### Analyze + report

```bash
cruncher analyze --latest -c "$CONFIG"
cruncher report --latest -c "$CONFIG"
```

Example output (analyze):

```bash
WARNING  Balance index undefined for 4 rows with joint_mean=0; writing NaN.
Analysis outputs → <workspace>/runs/sample/set1_lexA-cpxR/set1_lexA-cpxR_20260113_115749_e72283/analysis
  summary: <workspace>/runs/sample/set1_lexA-cpxR/set1_lexA-cpxR_20260113_115749_e72283/analysis/meta/summary.json
  diagnostics: <workspace>/runs/sample/set1_lexA-cpxR/set1_lexA-cpxR_20260113_115749_e72283/analysis/tables/diagnostics.json
  analysis_id: 20260113T165803Z_e758d7
Next steps:
  cruncher runs show <workspace>/config.yaml set1_lexA-cpxR_20260113_115749_e72283
  cruncher notebook --latest <workspace>/runs/sample/set1_lexA-cpxR/set1_lexA-cpxR_20260113_115749_e72283
  cruncher report --latest <workspace>/config.yaml
```

If you're running via `pixi`, prefix those next-step commands with `pixi run cruncher --`.

For a compact diagnostics checklist and tuning guidance, see the
[sampling + analysis guide](../guides/sampling_and_analysis.md).

---

### Optional: live analysis notebook

```bash
cruncher notebook <workspace>/runs/sample/set1_lexA-cpxR/set1_lexA-cpxR_20260113_115749_e72283 --latest
```

Example output:

```bash
Notebook created →
<workspace>/runs/sample/set1_lexA-cpxR/set1_lexA-cpxR_20260113_115749_e72283/analysis/notebooks/run_overview.py
Open with: marimo edit
<workspace>/runs/sample/set1_lexA-cpxR/set1_lexA-cpxR_20260113_115749_e72283/analysis/notebooks/run_overview.py
Read-only app: marimo run
<workspace>/runs/sample/set1_lexA-cpxR/set1_lexA-cpxR_20260113_115749_e72283/analysis/notebooks/run_overview.py
```

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
