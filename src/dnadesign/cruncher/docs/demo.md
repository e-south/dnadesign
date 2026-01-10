## cruncher demo (two TFs: LexA + CpxR)

This walkthrough shows an end-to-end process for discovering, fetching, locking, sampling, and analyzing two TFs (LexA + CpxR) with the bundled demo workspace. For multi-category campaigns and N>2 TF demos, see [demo_campaigns.md](demo_campaigns.md). For live sampling and validation UX, see [demo_progressive.md](demo_progressive.md). The demo includes:

- a local MEME motif source (`demo_local_meme`) for fast, offline matrix ingestion
- RegulonDB curated binding-site access for real-world inventory and site-based PWMs (HT optional)

Captured outputs below were generated on **2026-01-10** using `CRUNCHER_LOG_LEVEL=WARNING` and `COLUMNS=200`
to avoid truncated tables (unless noted otherwise). Expect timestamps and counts to differ in your environment.

### Enter the demo workspace

The demo workspace lives here:

- `src/dnadesign/cruncher/workspaces/demo/`

You can either `cd` into the workspace (auto-detects `config.yaml`), or run from
anywhere by setting a workspace selector:

```bash
# Option A: cd into the workspace
cd src/dnadesign/cruncher/workspaces/demo

# Option B: run from anywhere
export CRUNCHER_WORKSPACE=demo
# (or: cruncher --workspace demo <command>)
# (or: cruncher --config src/dnadesign/cruncher/workspaces/demo/config.yaml <command>)

# From here on, commands assume the config is discoverable
# (via CWD or CRUNCHER_WORKSPACE).
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
cruncher sources list
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
cruncher sources summary --source demo_local_meme --scope remote
cruncher sources summary --source regulondb --scope remote --remote-limit 20
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
09:08:33 INFO     Using config from CWD: ./config.yaml
         INFO     Fetching binding sites from regulondb for TFs=['lexA', 'cpxR'] motif_ids=[]
         INFO     Skipping TF 'lexA' (cached sites exist). Use --update to refresh.
         INFO     Skipping TF 'cpxR' (cached sites exist). Use --update to refresh.
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
                  Cache by source (source=demo_local_meme)
┏━━━━━━━━━━━━━━━━━┳━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Source          ┃ TFs ┃ Motifs ┃ Site sets ┃ Sites (seq/total) ┃ Datasets ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ demo_local_meme │   2 │      2 │         0 │ 0/0               │        0 │
└─────────────────┴─────┴────────┴───────────┴───────────────────┴──────────┘
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
│ entries           │ 10      │
│ sources           │ 1       │
│ TFs               │ 10      │
│ motifs            │ 0       │
│ site sets         │ 10      │
│ sites (seq/total) │ 872/872 │
│ datasets          │ 0       │
└───────────────────┴─────────┘
                  Cache by source (source=regulondb)
┏━━━━━━━━━━━┳━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Source    ┃ TFs ┃ Motifs ┃ Site sets ┃ Sites (seq/total) ┃ Datasets ┃
┡━━━━━━━━━━━╇━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ regulondb │  10 │      0 │        10 │ 872/872           │        0 │
└───────────┴─────┴────────┴───────────┴───────────────────┴──────────┘
                  Cache regulators (source=regulondb)
┏━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ TF   ┃ Sources   ┃ Motifs ┃ Site sets ┃ Sites (seq/total) ┃ Datasets ┃
┡━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ lrp  │ regulondb │      0 │         1 │ 219/219           │        0 │
│ fur  │ regulondb │      0 │         1 │ 217/217           │        0 │
│ cpxR │ regulondb │      0 │         1 │ 154/154           │        0 │
│ fnr  │ regulondb │      0 │         1 │ 152/152           │        0 │
│ lexA │ regulondb │      0 │         1 │ 49/49             │        0 │
│ soxS │ regulondb │      0 │         1 │ 44/44             │        0 │
│ rcdA │ regulondb │      0 │         1 │ 15/15             │        0 │
│ acrR │ regulondb │      0 │         1 │ 11/11             │        0 │
│ soxR │ regulondb │      0 │         1 │ 7/7               │        0 │
│ baeR │ regulondb │      0 │         1 │ 4/4               │        0 │
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

Example output (targets status):

```bash
                                                       Configured targets
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
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ TF   ┃ Source          ┃ Motif ID         ┃ Organism         ┃ Matrix     ┃ Sites (seq/total) ┃ Site kind ┃ Dataset ┃ Method ┃ Mean len ┃ Updated    ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ acrR │ regulondb       │ RDBECOLITFC00065 │ -                │ no         │ 11/11             │ curated   │ -       │ -      │ 18.9     │ 2026-01-09 │
│ baeR │ regulondb       │ RDBECOLITFC00182 │ -                │ no         │ 4/4               │ curated   │ -       │ -      │ 20.0     │ 2026-01-09 │
│ cpxR │ demo_local_meme │ cpxR             │ Escherichia coli │ yes (file) │ no                │ -         │ -       │ -      │ -        │ 2026-01-10 │
│ cpxR │ regulondb       │ RDBECOLITFC00170 │ -                │ no         │ 154/154           │ curated   │ -       │ -      │ 15.3     │ 2026-01-09 │
│ fnr  │ regulondb       │ RDBECOLITFC00128 │ -                │ no         │ 152/152           │ curated   │ -       │ -      │ 14.3     │ 2026-01-09 │
│ fur  │ regulondb       │ RDBECOLITFC00093 │ -                │ no         │ 217/217           │ curated   │ -       │ -      │ 18.6     │ 2026-01-09 │
│ lexA │ demo_local_meme │ lexA             │ Escherichia coli │ yes (file) │ no                │ -         │ -       │ -      │ -        │ 2026-01-10 │
│ lexA │ regulondb       │ RDBECOLITFC00214 │ -                │ no         │ 49/49             │ curated   │ -       │ -      │ 19.5     │ 2026-01-09 │
│ lrp  │ regulondb       │ RDBECOLITFC00014 │ -                │ no         │ 219/219           │ curated   │ -       │ -      │ 14.7     │ 2026-01-09 │
│ rcdA │ regulondb       │ RDBECOLITFC00048 │ -                │ no         │ 15/15             │ curated   │ -       │ -      │ 10.0     │ 2026-01-09 │
│ soxR │ regulondb       │ RDBECOLITFC00071 │ -                │ no         │ 7/7               │ curated   │ -       │ -      │ 18.0     │ 2026-01-09 │
│ soxS │ regulondb       │ RDBECOLITFC00201 │ -                │ no         │ 44/44             │ curated   │ -       │ -      │ 20.0     │ 2026-01-09 │
└──────┴─────────────────┴──────────────────┴──────────────────┴────────────┴───────────────────┴───────────┴─────────┴────────┴──────────┴────────────┘
```

Optional: a bird's-eye view of cache, targets, and recent runs (abridged; paths shortened for portability):

```bash
$ cruncher status
                                                    Configuration
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Setting      ┃ Value                                                                                              ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ config       │ /path/to/repo/src/dnadesign/cruncher/workspaces/demo/config.yaml                                   │
│ catalog_root │ /path/to/repo/src/dnadesign/cruncher/workspaces/demo/.cruncher                                     │
│ out_dir      │ /path/to/repo/src/dnadesign/cruncher/workspaces/demo/runs                                          │
│ pwm_source   │ sites                                                                                              │
│ sources      │ demo_local_meme, regulondb                                                                         │
│ lockfile     │ present                                                                                            │
└──────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘
        Cache
┏━━━━━━━━━━━┳━━━━━━━┓
┃ Metric    ┃ Value ┃
┡━━━━━━━━━━━╇━━━━━━━┩
│ entries   │ 12    │
│ motifs    │ 2     │
│ site_sets │ 10    │
└───────────┴───────┘
               Targets
┏━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ Total ┃ Ready ┃ Warning ┃ Blocking ┃
┡━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│     2 │     2 │       0 │        0 │
└───────┴───────┴─────────┴──────────┘
Runs total: 11 (parse:6, sample:5)
                                              Recent runs
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Run                                          ┃ Stage  ┃ Status    ┃ Created                          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ sample_set1_lexA-cpxR_20260110_084954_637d14 │ sample │ completed │ 2026-01-10T13:49:57.965750+00:00 │
│ parse_set1_lexA-cpxR_20260110_084951_321d1a  │ parse  │ completed │ 2026-01-10T13:49:51.070655+00:00 │
│ sample_set1_lexA-cpxR_20260109_222246_6ef55c │ sample │ completed │ 2026-01-10T03:22:49.845352+00:00 │
│ parse_set1_lexA-cpxR_20260109_222242_7960d9  │ parse  │ completed │ 2026-01-10T03:22:42.297514+00:00 │
│ sample_set1_lexA-cpxR_20260109_152547_c7afe6 │ sample │ completed │ 2026-01-09T20:25:50.689524+00:00 │
└──────────────────────────────────────────────┴────────┴───────────┴──────────────────────────────────┘
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
│ sample_set1_lexA-cpxR_20260110_084954_637d14 │ sample │ completed │ 2026-01-10T13:49:57.965750+00:00 │ 2      │ set1:lexA,cpxR │ sites      │
│ parse_set1_lexA-cpxR_20260110_084951_321d1a  │ parse  │ completed │ 2026-01-10T13:49:51.070655+00:00 │ 2      │ set1:lexA,cpxR │ sites      │
│ sample_set1_lexA-cpxR_20260109_222246_6ef55c │ sample │ completed │ 2026-01-10T03:22:49.845352+00:00 │ 2      │ set1:lexA,cpxR │ sites      │
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
│ cpxR │ demo_local_meme │ cpxR     │ 5      │ yes (file) │ 2026-01-10 │
│ lexA │ demo_local_meme │ lexA     │ 6      │ yes (file) │ 2026-01-10 │
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
