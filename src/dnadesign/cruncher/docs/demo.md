## cruncher demo (local motifs + RegulonDB)

This walkthrough shows an end-to-end process for discovering, fetching, locking, sampling, and analyzing two TFs (LexA + CpxR) with the bundled demo workspace. The demo includes:

- a local MEME motif source (`demo_local_meme`) for fast, offline matrix ingestion
- RegulonDB curated binding-site access for real-world inventory and site-based PWMs (HT optional)

Captured outputs below were generated on **2026-01-06** using `CRUNCHER_LOG_LEVEL=WARNING` and `COLUMNS=200`
to avoid truncated tables (unless noted otherwise). Expect timestamps and counts to differ in your environment.

### Enter the demo workspace

The demo workspace lives here:

- `src/dnadesign/cruncher/workspaces/demo/`

You can either `cd` into the workspace (auto-detects `config.yaml`), or run from
anywhere using the workspace-aware flags:

```bash
# Option A: cd into the workspace
cd src/dnadesign/cruncher/workspaces/demo

# Option B: run from anywhere
cruncher --workspace demo sources list
cruncher --config src/dnadesign/cruncher/workspaces/demo/config.yaml sources list
```

The demo config is tuned for fast local runs. Increase `draws` and `tune` for
real experiments.

Local demo motifs live at `data/local_motifs/` relative to the workspace root
(two small MEME files). From the repo root, the full path is
`src/dnadesign/cruncher/workspaces/demo/data/local_motifs/`.
To point to a real dataset, update `ingest.local_sources[].root` in the config.

## Preview what is available (optional)

List the sources registered by the demo config:

```bash
cruncher sources list config.yaml
```

Example output:

```bash
                            Sources
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Source          ┃ Description                                ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ demo_local_meme │ Demo MEME motifs (local files)             │
│ regulondb       │ RegulonDB datamarts GraphQL (curated + HT) │
└─────────────────┴────────────────────────────────────────────┘
```

Summarize available regulators for a specific source (remote inventory):

```bash
cruncher sources summary --source demo_local_meme --scope remote config.yaml
cruncher sources summary --source regulondb --scope remote --remote-limit 20 config.yaml
```

Example output (local inventory):

```bash
         Remote inventory by source
┏━━━━━━━━━━━━━━━━━┳━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Source          ┃ TFs ┃ Motifs ┃ Datasets ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ demo_local_meme │   2 │      2 │        0 │
└─────────────────┴─────┴────────┴──────────┘
      Remote regulators: demo_local_meme
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ TF   ┃ Sources         ┃ Motifs ┃ Datasets ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ cpxR │ demo_local_meme │      1 │        0 │
│ lexA │ demo_local_meme │      1 │        0 │
└──────┴─────────────────┴────────┴──────────┘
```

Example output (RegulonDB inventory, remote-limit=20):

```bash
 Remote inventory by source (limit=20)
┏━━━━━━━━━━━┳━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Source    ┃ TFs ┃ Motifs ┃ Datasets ┃
┡━━━━━━━━━━━╇━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ regulondb │  20 │     20 │      533 │
└───────────┴─────┴────────┴──────────┘
 Remote regulators: regulondb (limit=20)
┏━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ TF     ┃ Sources   ┃ Motifs ┃ Datasets ┃
┡━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ AgrB   │ regulondb │      1 │        0 │
│ ArrS   │ regulondb │      1 │        0 │
│ ChiX   │ regulondb │      1 │        0 │
│ DicF   │ regulondb │      1 │        0 │
│ DsrA   │ regulondb │      1 │        0 │
│ FnrS   │ regulondb │      1 │        0 │
│ GadY   │ regulondb │      1 │        0 │
│ GcvB   │ regulondb │      1 │        0 │
│ IstR-1 │ regulondb │      1 │        0 │
│ McaS   │ regulondb │      1 │        0 │
│ MgrR   │ regulondb │      1 │        0 │
│ MicF   │ regulondb │      1 │        0 │
│ OhsC   │ regulondb │      1 │        0 │
│ OxyS   │ regulondb │      1 │        0 │
│ ppGpp  │ regulondb │      1 │        0 │
│ RseX   │ regulondb │      1 │        0 │
│ RydC   │ regulondb │      1 │        0 │
│ SdsN   │ regulondb │      1 │        0 │
│ SgrS   │ regulondb │      1 │        0 │
│ SymR   │ regulondb │      1 │        0 │
└────────┴───────────┴────────┴──────────┘
```

Preview HT datasets (note: `fetch sites --dry-run` lists HT datasets):

```bash
cruncher fetch sites --tf lexA --tf cpxR --dry-run
```

Example output:

```bash
                         HT datasets
┏━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃ TF   ┃ Dataset ID       ┃ Source   ┃ Method    ┃ Genome   ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│ lexA │ RHTECOLIBSD02444 │ BAUMGART │ TFBINDING │ U00096.3 │
│ lexA │ RHTECOLIBSD03022 │ GALAGAN  │ TFBINDING │ -        │
│ cpxR │ RHTECOLIBSD02736 │ PALSSON  │ TFBINDING │ U00096.3 │
│ cpxR │ RHTECOLIBSD02409 │ BAUMGART │ TFBINDING │ U00096.3 │
│ cpxR │ RHTECOLIBSD02988 │ GALAGAN  │ TFBINDING │ -        │
└──────┴──────────────────┴──────────┴───────────┴──────────┘
```

### Fetch binding sites (curated; HT optional)

This demo uses `motif_store.pwm_source: sites`, so we cache curated sites and
build PWMs at runtime. If you set `ingest.regulondb.ht_sites: true`, HT datasets
are also available (use `--dataset-id` to pin a specific HT dataset). Some TFs
may not have HT data; cruncher will warn and continue with curated sites.

```bash
cruncher fetch sites --tf lexA --tf cpxR
```

Example output (abridged, default INFO log level):

```bash
10:42:47 INFO     Using config from CWD: ./config.yaml
         INFO     Fetching binding sites from regulondb for TFs=['lexA', 'cpxR']
                  motif_ids=[]
         INFO     Skipping TF 'lexA' (cached sites exist). Use --update to
                  refresh.
         INFO     Skipping TF 'cpxR' (cached sites exist). Use --update to
                  refresh.
No new sites cached (all matches already present). Use --update to refresh.
```

### Summarize cached regulators (per source)

After fetching, you can summarize cached regulators for each source:

```bash
cruncher sources summary --source demo_local_meme --scope cache
cruncher sources summary --source regulondb --scope cache
```

Example output (local cache):

```bash
       Cache overview
  (source=demo_local_meme)
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric            ┃ Value ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ entries           │ 2     │
│ sources           │ 1     │
│ TFs               │ 2     │
│ motifs            │ 2     │
│ site sets         │ 0     │
│ sites (seq/total) │ 0/0   │
│ datasets          │ 0     │
└───────────────────┴───────┘
                  Cache regulators (source=demo_local_meme)
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ TF   ┃ Sources         ┃ Motifs ┃ Site sets ┃ Sites (seq/total) ┃ Datasets ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ cpxR │ demo_local_meme │      1 │         0 │ 0/0               │        0 │
│ lexA │ demo_local_meme │      1 │         0 │ 0/0               │        0 │
└──────┴─────────────────┴────────┴───────────┴───────────────────┴──────────┘
```

Example output (RegulonDB cache, abridged):

```bash
        Cache overview
      (source=regulondb)
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric            ┃ Value   ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ entries           │ 2       │
│ sources           │ 1       │
│ TFs               │ 2       │
│ motifs            │ 0       │
│ site sets         │ 2       │
│ sites (seq/total) │ 203/203 │
│ datasets          │ 0       │
└───────────────────┴─────────┘
                  Cache regulators (source=regulondb)
┏━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ TF   ┃ Sources   ┃ Motifs ┃ Site sets ┃ Sites (seq/total) ┃ Datasets ┃
┡━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ cpxR │ regulondb │      0 │         1 │ 154/154           │        0 │
│ lexA │ regulondb │      0 │         1 │ 49/49             │        0 │
└──────┴───────────┴────────┴───────────┴───────────────────┴──────────┘
```

Tip: to include remote inventory, add `--scope remote --remote-limit 200`.

### Inspect what is cached (optional)

```bash
cruncher catalog list
cruncher targets status
cruncher targets stats
cruncher targets candidates --fuzzy
```

If you cached local motifs in matrix mode, `catalog list` will also show
`demo_local_meme` entries with `Matrix=yes` and no site sets.

Example output (targets list):

```bash
$ cruncher targets list
  Configured
   targets
┏━━━━━┳━━━━━━┓
┃ Set ┃ TF   ┃
┡━━━━━╇━━━━━━┩
│   1 │ lexA │
│   1 │ cpxR │
└─────┴──────┘
```

### Categories & campaigns (optional)

The demo config includes both a small pairwise campaign and an expanded
multi-category campaign:

- `demo_pair` (Stress + Envelope)
- `demo_categories` (Category1/2/3, no selectors)
- `demo_categories_best` (Category1/2/3 with quality selectors)

Category definitions (from the demo config):

- Category1: CpxR, BaeR
- Category2: LexA, RcdA, Lrp, Fur
- Category3: Fnr, Fur, AcrR, SoxR, SoxS, Lrp

List targets by category or campaign:

```bash
cruncher targets list --category Stress
cruncher targets list --category Category1
cruncher targets list --campaign demo_pair
cruncher targets list --campaign demo_categories
```

Fetch sites for the expanded categories (RegulonDB curated sites):

```bash
cruncher fetch sites --campaign demo_categories --no-selectors
```

Optional DAP-seq local source:

- If you have the O'Malley DAP-seq MEME files locally, add a `local_sources`
  entry (see `docs/config.md`) with `extract_sites: true`.
- Then fetch from that source instead of RegulonDB:

```bash
cruncher fetch sites --source omalley_ecoli_meme --campaign demo_categories --no-selectors
```

Summarize what is available:

```bash
cruncher sources summary --source regulondb --scope cache
cruncher targets stats --campaign demo_categories
```

Apply selectors to keep the strongest candidates and generate a derived config:

```bash
cruncher campaign generate --campaign demo_categories_best --out config.demo_categories_best.yaml
```

The companion manifest (`config.demo_categories_best.campaign_manifest.json`)
records per-TF metrics (site counts, plus info bits if you enable that selector).

Run a multi-dimensional optimization and plot the facet grid:

```bash
cruncher lock config.demo_categories_best.yaml
cruncher parse config.demo_categories_best.yaml
cruncher sample config.demo_categories_best.yaml
cruncher analyze --latest --plots pairgrid config.demo_categories_best.yaml
```

Optional campaign-level summary (pairs + facets across runs):

```bash
cruncher campaign summarize --campaign demo_categories_best --skip-missing
```

Notes:

- Large campaigns can generate many regulator sets. For a quick demo, trim
  `regulator_sets` in the generated config to a smaller subset.
- If you want pairwise plots for a specific TF pair, update `analysis.tf_pair`
  in the generated config before running `cruncher analyze`.
- `selectors.min_info_bits` requires PWMs to be buildable. For site-based PWMs
  with variable site lengths, set `motif_store.site_window_lengths` per TF (or
  switch to matrix-based sources) before enabling that selector.
- The demo config pre-populates `site_window_lengths` for the expanded TF list
  so multi-TF parse/sample runs work without extra edits.

Example output (targets status):

```bash
                                                          Target status
┏━━━━━┳━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┓
┃ Set ┃ TF   ┃ Source    ┃ Motif ID         ┃ Organism ┃ Matrix ┃ Sites (seq/total) ┃ Site kind ┃ Dataset ┃ PWM source ┃ Status ┃
┡━━━━━╇━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━┩
│   1 │ lexA │ regulondb │ RDBECOLITFC00214 │ -        │ no     │ 49/49             │ curated   │ -       │ sites      │ ready  │
│   1 │ cpxR │ regulondb │ RDBECOLITFC00170 │ -        │ no     │ 154/154           │ curated   │ -       │ sites      │ ready  │
└─────┴──────┴───────────┴──────────────────┴──────────┴────────┴───────────────────┴───────────┴─────────┴────────────┴────────┘
```

Example output (catalog list):

```bash
                                                               Catalog
┏━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ TF   ┃ Source    ┃ Motif ID         ┃ Organism ┃ Matrix ┃ Sites (seq/total) ┃ Site kind ┃ Dataset ┃ Method ┃ Mean len ┃ Updated    ┃
┡━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ cpxR │ regulondb │ RDBECOLITFC00170 │ -        │ no     │ 154/154           │ curated   │ -       │ -      │ 15.3     │ 2026-01-05 │
│ lexA │ regulondb │ RDBECOLITFC00214 │ -        │ no     │ 49/49             │ curated   │ -       │ -      │ 19.5     │ 2026-01-04 │
└──────┴───────────┴──────────────────┴──────────┴────────┴───────────────────┴───────────┴─────────┴────────┴──────────┴────────────┘
```

Optional: a bird's-eye view of cache, targets, and recent runs (abridged; paths shortened for portability):

```bash
$ cruncher status
                               Configuration
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Setting      ┃ Value                                                     ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ config       │ /path/to/repo/src/dnadesign/cruncher/workspaces/demo/config.yaml             │
│ catalog_root │ /path/to/repo/src/dnadesign/cruncher/workspaces/demo/.cruncher               │
│ out_dir      │ /path/to/repo/src/dnadesign/cruncher/workspaces/demo/runs                    │
│ pwm_source   │ sites                                                     │
│ sources      │ regulondb                                                 │
│ lockfile     │ present                                                   │
└──────────────┴───────────────────────────────────────────────────────────┘
        Cache
┏━━━━━━━━━━━┳━━━━━━━┓
┃ Metric    ┃ Value ┃
┡━━━━━━━━━━━╇━━━━━━━┩
│ entries   │ 3     │
│ motifs    │ 0     │
│ site_sets │ 3     │
└───────────┴───────┘
               Targets
┏━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ Total ┃ Ready ┃ Warning ┃ Blocking ┃
┡━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│     2 │     2 │       0 │        0 │
└───────┴───────┴─────────┴──────────┘
Runs total: 8 (parse:4, sample:4)
                                  Recent runs
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Run                        ┃ Stage  ┃ Status    ┃ Created                    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ sample_set1_lexA-cpxR_20260105_162916_52e28b │ sample │ completed │ 20260105 │
│ parse_set1_lexA-cpxR_20260105_162915_906e79  │ parse  │ completed │ 20260105 │
└────────────────────────────┴────────┴───────────┴────────────────────────────┘
```

### Lock TFs to exact cached motifs

Lockfiles resolve TF names to exact motif IDs and hashes for reproducibility:

```bash
cruncher lock
```

Lockfiles live at `.cruncher/locks/<config>.lock.json` and are required for
`parse` and `sample`.

### Parse logos (optional)

```bash
cruncher parse
```

Logos are written under `runs/parse_set<index>_<tfset>_<timestamp>/` (inside the demo workspace).
When `motif_store.pwm_source=sites`, the logo subtitle shows whether binding sites were
curated, high-throughput, or combined.

### Sample (MCMC optimization)

```bash
cruncher sample
cruncher runs list
```

Use `cruncher runs list` to find the sample run name for reporting.
For live progress, you can watch the run status:

```bash
cruncher runs watch <run_name>
```

Example output (runs list, abridged):

```bash
                                                                     Runs
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Name                                         ┃ Stage  ┃ Status    ┃ Created                          ┃ Motifs ┃ Regulator set  ┃ PWM source ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ sample_set1_lexA-cpxR_20260105_162916_52e28b │ sample │ completed │ 2026-01-05T21:29:20.934494+00:00 │ 2      │ set1:lexA,cpxR │ sites      │
│ parse_set1_lexA-cpxR_20260105_162915_906e79  │ parse  │ completed │ 2026-01-05T21:29:15.480222+00:00 │ 2      │ set1:lexA,cpxR │ sites      │
│ sample_set1_lexA-cpxR_20260105_151840_e817e1 │ sample │ completed │ 2026-01-05T20:18:44.655691+00:00 │ 2      │ set1:lexA,cpxR │ sites      │
└──────────────────────────────────────────────┴────────┴───────────┴──────────────────────────────────┴────────┴────────────────┴────────────┘
```

### Analyze + report

```bash
cruncher analyze --latest
cruncher report <run_name>
```

Each analyze run writes the latest analysis into `analysis/` inside the sample run folder,
including `summary.json`, `analysis_used.yaml`, plots, and tables. When archiving is enabled,
the previous analysis is moved to `analysis/_archive/<analysis_id>/`.

Optional interactive exploration:

```bash
cruncher notebook <path/to/sample_run> --latest
```

Tip: `cruncher runs show <run_name>` prints the full run directory.

### Optional: use matrix mode (local motifs + alignment)

To use local motif matrices (or alignment matrices from RegulonDB), switch to
matrix mode:

```yaml
motif_store:
  pwm_source: matrix
  source_preference: [demo_local_meme, regulondb]
```

Then fetch the local MEME motifs:

```bash
cruncher fetch motifs --source demo_local_meme --tf lexA --tf cpxR
```

Example output (local motifs):

```bash
                             Fetched motifs
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ TF   ┃ Source          ┃ Motif ID ┃ Length ┃ Matrix     ┃ Updated    ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ cpxR │ demo_local_meme │ cpxR     │ 5      │ yes (file) │ 2026-01-06 │
│ lexA │ demo_local_meme │ lexA     │ 6      │ yes (file) │ 2026-01-06 │
└──────┴─────────────────┴──────────┴────────┴────────────┴────────────┘
```

Note: not all RegulonDB regulons ship alignment matrices. If `fetch motifs` fails
with “alignment matrix is missing”, use `pwm_source: sites` and `fetch sites`
instead.

### Optional: HT-only or combined site modes (RegulonDB)

You can restrict `pwm_source: sites` to curated or HT sites, or combine them:

```yaml
motif_store:
  pwm_source: sites
  site_kinds: ["curated"]          # curated-only
  # site_kinds: ["ht_tfbinding"]   # HT-only
  # combine_sites: true            # combine curated + HT when site_kinds is null
```

Tip: if HT datasets return peaks without sequences, hydration uses NCBI by default
(`ingest.genome_source=ncbi`). To run offline, provide a local FASTA via
`--genome-fasta` or `ingest.genome_fasta`.

If HT site lengths vary, use `cruncher targets stats` and set
`motif_store.site_window_lengths` per TF or dataset before building PWMs.

### Next steps

- Command reference: [cli.md](cli.md)
- Config knobs: [config.md](config.md)
- Ingestion details: [ingestion.md](ingestion.md)


### Where outputs live

- `.cruncher/` - project-local cache, lockfiles, and run index.
- `<out_dir>/` - parse/sample runs (set by `cruncher.out_dir`, for example `runs/`).

Sample run contents:

```
config_used.yaml
run_manifest.json
analysis/
report.md
```
