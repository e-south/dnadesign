## cruncher demo 1

**Jointly maximizing a sequence based on two TFs.**

This demo walks through a process for discovering TFs (e.g., LexA + CpxR), fetching binding sites, locking motif data (i.e., our "scorecards"), sampling sequence space, optimization, and analyzing results with the bundled `demo_basics_two_tf` workspace.

The demo includes:

- a local DAP-seq MEME source (`demo_local_meme`) that provides motif matrices
  and MEME BLOCKS training-site sequences.
- RegulonDB curated binding-site access with optional high-throughput (HT) datasets.

Timestamps and run IDs in the example output will differ run-to-run.
This demo uses a dedicated cache root (`.cruncher/demo_basics_two_tf/`) so only LexA and CpxR appear in the cache summaries.

### Demo instance

- **Workspace**: `src/dnadesign/cruncher/workspaces/demo_basics_two_tf/`
- **Config**: `config.yaml`
- **Output root**: `runs/` (relative to the workspace)

```bash
# Option A: cd into the workspace
cd src/dnadesign/cruncher/workspaces/demo_basics_two_tf
CONFIG=config.yaml

# Option B: run from anywhere in the repo
CONFIG=src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml

# From here on, commands use $CONFIG for clarity; if you're in the workspace, you can omit --config.
```

Local demo motifs live at `data/local_motifs/`, residing in two MEME files derived from [this](https://www.nature.com/articles/s41592-021-01312-2) DAP-seq article).

### Preview sources and inventories (optional)

List the sources registered by the demo config:

```bash
uv run cruncher -c "$CONFIG" sources list
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

Inspect source capabilities (what each source can provide):

```bash
uv run cruncher -c "$CONFIG" sources info demo_local_meme
uv run cruncher -c "$CONFIG" sources info regulondb
```

Example output:

```bash
demo_local_meme: motifs:get, motifs:iter, motifs:list, sites:list
regulondb: datasets:list, motifs:get, motifs:iter, motifs:list, sites:list
```

`motifs:*` are motif matrix inventories (curated or local). `sites:list` means the source can return binding-site sequences (curated or MEME BLOCKS training sites). `datasets:list` is the HT dataset registry (RegulonDB only).

Summarize available regulators for a specific source (remote inventory):

```bash
uv run cruncher -c "$CONFIG" sources summary --source demo_local_meme --scope remote
uv run cruncher -c "$CONFIG" sources summary --source regulondb --scope remote --remote-limit 20
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
Legend: Motifs = source inventory entries; Datasets = HT dataset IDs mentioning the TF (not binding sites).
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
Legend: Motifs = source inventory entries; Datasets = HT dataset IDs mentioning the TF (not binding sites).
Note: --remote-limit samples the motif inventory; regulator rows are a partial view.
```

The remote regulators table is a **motif inventory** (not site counts). A TF can have curated motifs but no HT datasets, so `Datasets=0` is expected for many entries. Use `sources datasets` or `fetch sites --dry-run` to inspect HT availability for specific TFs.

List HT datasets for a TF (RegulonDB only):

```bash
uv run cruncher -c "$CONFIG" sources datasets regulondb --tf lexA --limit 5
```

Example output:

```bash
                                                 regulondb datasets
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Dataset ID       ┃ Source   ┃ Method    ┃ TFs                                                          ┃ Genome   ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ RHTECOLIBSD02444 │ BAUMGART │ TFBINDING │ DNA-binding transcriptional repressor LexA, ExrA, LexA, Spr… │ U00096.3 │
│ RHTECOLIBSD03022 │ GALAGAN  │ TFBINDING │ DNA-binding transcriptional repressor LexA, ExrA, LexA, Spr… │ -        │
└──────────────────┴──────────┴───────────┴──────────────────────────────────────────────────────────────┴──────────┘
```

Optional: use the fetch pipeline itself to preview HT datasets (TF-scoped and filter-aware):

```bash
uv run cruncher -c "$CONFIG" fetch sites --tf lexA --tf cpxR --dry-run
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

This demo uses `motif_store.pwm_source: sites`, so we cache curated sites and build PWMs at runtime. You can pin a specific HT dataset with `--dataset-id`, which enables HT access for that request. If a dataset returns zero TF-binding records, cruncher fails fast with a clear error so you can choose a different dataset.

```bash
uv run cruncher -c "$CONFIG" fetch sites --tf lexA --tf cpxR --update
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
│ cpxR │ regulondb │ RDBECOLITFC00170 │ curated │ -       │ -      │ 154   │ 154   │ 15.3     │ 2026-01-11 │
│ lexA │ regulondb │ RDBECOLITFC00214 │ curated │ -       │ -      │ 49    │ 49    │ 19.5     │ 2026-01-11 │
└──────┴───────────┴──────────────────┴─────────┴─────────┴────────┴───────┴───────┴──────────┴────────────┘
```

Curated site sets do not carry a dataset or method (those fields are only populated for HT datasets).

Repeat runs will report “No new sites cached” unless you pass `--update`.

### Fetch local DAP-seq motifs + training sites

Local DAP-seq MEME files live under `data/local_motifs/` in the workspace. Fetch the motif matrices and the MEME BLOCKS training sites:

```bash
uv run cruncher -c "$CONFIG" fetch motifs --source demo_local_meme --tf lexA --tf cpxR --update
uv run cruncher -c "$CONFIG" fetch sites --source demo_local_meme --tf lexA --tf cpxR --update
```

Example output:

```bash
                             Fetched motifs
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ TF   ┃ Source          ┃ Motif ID ┃ Length ┃ Matrix     ┃ Updated    ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ cpxR │ demo_local_meme │ cpxR     │ 21     │ yes (file) │ 2026-01-11 │
│ lexA │ demo_local_meme │ lexA     │ 22     │ yes (file) │ 2026-01-11 │
└──────┴─────────────────┴──────────┴────────┴────────────┴────────────┘
```

Example output (local sites):

```bash
                                         Fetched binding-site sets
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ TF   ┃ Source          ┃ Motif ID ┃ Kind        ┃ Dataset ┃ Method ┃ Sites ┃ Total ┃ Mean len ┃ Updated    ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ cpxR │ demo_local_meme │ cpxR     │ meme_blocks │ -       │ -      │ 50    │ 50    │ 21.0     │ 2026-01-11 │
│ lexA │ demo_local_meme │ lexA     │ meme_blocks │ -       │ -      │ 50    │ 50    │ 22.0     │ 2026-01-11 │
└──────┴─────────────────┴──────────┴──────────━──┴─────────┴────────┴───────┴───────┴──────────┴────────────┘
```

### Summarize cached regulators (per source)

After fetching, you can summarize cached regulators for each source:

```bash
uv run cruncher -c "$CONFIG" sources summary --source demo_local_meme --scope cache
uv run cruncher -c "$CONFIG" sources summary --source regulondb --scope cache
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
│ site sets         │ 2     │
│ sites (seq/total) │ 100/100 │
│ datasets          │ 0     │
└───────────────────┴───────┘
                  Cache by source (source=demo_local_meme)
┏━━━━━━━━━━━━━━━━━┳━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Source          ┃ TFs ┃ Motifs ┃ Site sets ┃ Sites (seq/total) ┃ Datasets ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ demo_local_meme │   2 │      2 │         2 │ 100/100           │        0 │
└─────────────────┴─────┴────────┴───────────┴───────────────────┴──────────┘
                  Cache regulators (source=demo_local_meme)
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ TF   ┃ Sources         ┃ Motifs ┃ Site sets ┃ Sites (seq/total) ┃ Datasets ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ cpxR │ demo_local_meme │      1 │         1 │ 50/50             │        0 │
│ lexA │ demo_local_meme │      1 │         1 │ 50/50             │        0 │
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
                  Cache by source (source=regulondb)
┏━━━━━━━━━━━┳━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Source    ┃ TFs ┃ Motifs ┃ Site sets ┃ Sites (seq/total) ┃ Datasets ┃
┡━━━━━━━━━━━╇━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ regulondb │   2 │      0 │         2 │ 203/203           │        0 │
└───────────┴─────┴────────┴───────────┴───────────────────┴──────────┘
                  Cache regulators (source=regulondb)
┏━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ TF   ┃ Sources   ┃ Motifs ┃ Site sets ┃ Sites (seq/total) ┃ Datasets ┃
┡━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ cpxR │ regulondb │      0 │         1 │ 154/154           │        0 │
│ lexA │ regulondb │      0 │         1 │ 49/49             │        0 │
└──────┴───────────┴────────┴───────────┴───────────────────┴──────────┘
```

## Optional: combine curated + DAP-seq sites for a single PWM

Now that both sources are cached, you can merge their site sets per TF and build
one PWM from multiple sources. Toggle `combine_sites` and re-lock:

```yaml
motif_store:
  combine_sites: true
```

```bash
uv run cruncher -c "$CONFIG" lock
uv run cruncher -c "$CONFIG" targets status
```

Example output (combined sites):

```bash
                                                       Configured targets
┏━━━━━┳━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┓
┃ Set ┃ TF   ┃ Source    ┃ Motif ID         ┃ Organism ┃ Matrix ┃ Sites (seq/total) ┃ Site kind ┃ Dataset ┃ PWM source ┃ Status ┃
┡━━━━━╇━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━┩
│   1 │ lexA │ regulondb │ RDBECOLITFC00214 │ -        │ no     │ 99/99             │ mixed     │ -       │ sites      │ ready  │
│   1 │ cpxR │ regulondb │ RDBECOLITFC00170 │ -        │ no     │ 204/204           │ mixed     │ -       │ sites      │ ready  │
└─────┴──────┴───────────┴──────────────────┴──────────┴────────┴───────────────────┴───────────┴─────────┴────────────┴────────┘
```

Tip: `catalog pwms --set 1` will now show `site sets=2` per TF. To use only the
local DAP-seq training sites, set `site_kinds: ["meme_blocks"]`.

## Lock TFs to exact cached motifs

Lockfiles resolve TF names to exact motif IDs and hashes for reproducibility:

```bash
uv run cruncher -c "$CONFIG" lock
```

Example output:

```bash
/Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_basics_two_tf/.cruncher/demo_basics_two_tf/locks/config.lock.json
```

Lockfiles are required for `parse`, `sample`, and `targets status`.

## Inspect cached entries and targets (optional)

```bash
uv run cruncher -c "$CONFIG" catalog list
uv run cruncher -c "$CONFIG" catalog show regulondb:RDBECOLITFC00214
uv run cruncher -c "$CONFIG" targets list
uv run cruncher -c "$CONFIG" targets status
uv run cruncher -c "$CONFIG" targets stats
uv run cruncher -c "$CONFIG" targets candidates --fuzzy
uv run cruncher -c "$CONFIG" status
```

Example output (catalog list):

```bash
                                                                        Catalog
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ TF   ┃ Source          ┃ Motif ID         ┃ Organism         ┃ Matrix     ┃ Sites (seq/total) ┃ Site kind ┃ Dataset ┃ Method ┃ Mean len ┃ Updated    ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ cpxR │ demo_local_meme │ cpxR             │ Escherichia coli │ yes (file) │ 50/50             │ meme_blocks │ -       │ -      │ 21.0     │ 2026-01-10 │
│ cpxR │ regulondb       │ RDBECOLITFC00170 │ -                │ no         │ 154/154           │ curated   │ -       │ -      │ 15.3     │ 2026-01-10 │
│ lexA │ demo_local_meme │ lexA             │ Escherichia coli │ yes (file) │ 50/50             │ meme_blocks │ -       │ -      │ 22.0     │ 2026-01-10 │
│ lexA │ regulondb       │ RDBECOLITFC00214 │ -                │ no         │ 49/49             │ curated   │ -       │ -      │ 19.5     │ 2026-01-10 │
└──────┴─────────────────┴──────────────────┴──────────────────┴────────────┴───────────────────┴───────────┴─────────┴────────┴──────────┴────────────┘
```

Example output (catalog show for a binding-site set):

```bash
source: regulondb
motif_id: RDBECOLITFC00214
tf_name: lexA
organism: -
kind: PFM
matrix_length: None
matrix_source: None
matrix_semantics: None
has_matrix: False
has_sites: True
site_count: 49
site_total: 49
site_kind: curated
site_length_mean: 19.49 (min=15, max=20, n=49)
site_length_source: sequence
dataset_id: -
dataset_source: -
dataset_method: -
reference_genome: -
updated_at: 2026-01-10T21:04:06.147428+00:00
synonyms: -
motif_path: -
sites_path: /Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_basics_two_tf/.cruncher/demo_basics_two_tf/normalized/sites/regulondb/RDBECOLITFC00214.jsonl
```

Example output (targets list):

```bash
$ uv run cruncher -c "$CONFIG" targets list
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

Optional: a bird's-eye view of cache, targets, and recent runs (abridged; paths shortened for portability):

```bash
$ uv run cruncher -c "$CONFIG" status
                                                           Configuration
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Setting      ┃ Value                                                                                                            ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ config       │ /Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml │
│ catalog_root │ /Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_basics_two_tf/.cruncher/demo_basics_two_tf   │
│ out_dir      │ /Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_basics_two_tf/runs        │
│ pwm_source   │ sites                                                                                                            │
│ sources      │ demo_local_meme, regulondb                                                                                       │
│ lockfile     │ present                                                                                                          │
└──────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
        Cache
┏━━━━━━━━━━━┳━━━━━━━┓
┃ Metric    ┃ Value ┃
┡━━━━━━━━━━━╇━━━━━━━┩
│ entries   │ 4     │
│ motifs    │ 2     │
│ site_sets │ 4     │
└───────────┴───────┘
               Targets
┏━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ Total ┃ Ready ┃ Warning ┃ Blocking ┃
┡━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│     2 │     2 │       0 │        0 │
└───────┴───────┴─────────┴──────────┘
Runs total: 5 (parse:2, sample:3)
                                              Recent runs
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Run                                          ┃ Stage  ┃ Status    ┃ Created                          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ sample_set1_lexA-cpxR_20260110_160900_58723f │ sample │ completed │ 2026-01-10T21:09:03.989466+00:00 │
│ parse_set1_lexA-cpxR_20260110_160854_41bbc7  │ parse  │ completed │ 2026-01-10T21:08:54.641651+00:00 │
│ sample_set1_lexA-cpxR_20260110_130820_e63208 │ sample │ completed │ 2026-01-10T18:08:20.007353+00:00 │
│ sample_set1_lexA-cpxR_20260110_125203_feea4f │ sample │ completed │ 2026-01-10T17:52:07.208085+00:00 │
│ parse_set1_lexA-cpxR_20260110_125126_67336d  │ parse  │ completed │ 2026-01-10T17:51:26.301780+00:00 │
└──────────────────────────────────────────────┴────────┴───────────┴──────────────────────────────────┘
```

## Compute PWMs + inspect information content

`catalog pwms` builds PWMs from cached sites (or matrices) and reports
information content in bits:

```bash
uv run cruncher -c "$CONFIG" catalog pwms --set 1
```

Example output:

```bash
                                        PWM summary
┏━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┓
┃ TF   ┃ Source    ┃ Motif ID         ┃ PWM source ┃ Length ┃ Bits  ┃ n sites ┃ Site sets ┃
┡━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━┩
│ lexA │ regulondb │ RDBECOLITFC00214 │ sites      │ 15     │ 10.36 │ 49      │ 1         │
│ cpxR │ regulondb │ RDBECOLITFC00170 │ sites      │ 11     │ 3.63  │ 154     │ 1         │
└──────┴───────────┴──────────────────┴────────────┴────────┴───────┴─────────┴───────────┘
```

Tip: add `--matrix` to print the full PWM matrices or `--log-odds` for log-odds matrices.

Use the **Bits** and **n sites** columns as a quick quality screen. Low site counts
or very low information content are signals to (a) prefer another source, (b) combine
curated + DAP-seq sites, or (c) raise `motif_store.min_sites_for_pwm` if you want to
enforce stricter minimums.

## Render PWM logos

```bash
uv run cruncher -c "$CONFIG" catalog logos --set 1
```

Example output (paths shortened):

```bash
Rendered PWM logos
┏━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ TF   ┃ Source    ┃ Motif ID         ┃ Length ┃ Bits  ┃ Output                                                       ┃
┡━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ lexA │ regulondb │ RDBECOLITFC00214 │ 15     │ 10.36 │ /path/to/.../logos_set1_lexA-cpxR_20260110_160826_7b196a │
│ cpxR │ regulondb │ RDBECOLITFC00170 │ 11     │ 3.63  │ /path/to/.../logos_set1_lexA-cpxR_20260110_160826_7b196a │
└──────┴───────────┴──────────────────┴────────┴───────┴──────────────────────────────────────────────────────────────────────┘
Logos saved to /path/to/.../logos_set1_lexA-cpxR_20260110_160826_7b196a
```

## Parse logos (optional)

```bash
uv run cruncher -c "$CONFIG" parse
```

Logos are written under `runs/parse_set<index>_<tfset>_<timestamp>/`.
When `motif_store.pwm_source=sites`, the logo subtitle shows whether binding sites were
curated, high-throughput, or combined.

## Sample (MCMC optimization)

```bash
uv run cruncher -c "$CONFIG" sample
uv run cruncher -c "$CONFIG" runs list
```

Example output (runs list, abridged):

```bash
                                                                     Runs
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Name                                         ┃ Stage  ┃ Status    ┃ Created                          ┃ Motifs ┃ Regulator set  ┃ PWM source ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ sample_set1_lexA-cpxR_20260110_160900_58723f │ sample │ completed │ 2026-01-10T21:09:03.989466+00:00 │ 2      │ set1:lexA,cpxR │ sites      │
│ parse_set1_lexA-cpxR_20260110_160854_41bbc7  │ parse  │ completed │ 2026-01-10T21:08:54.641651+00:00 │ 2      │ set1:lexA,cpxR │ sites      │
│ sample_set1_lexA-cpxR_20260110_130820_e63208 │ sample │ completed │ 2026-01-10T18:08:20.007353+00:00 │ 2      │ set1:lexA,cpxR │ sites      │
│ sample_set1_lexA-cpxR_20260110_125203_feea4f │ sample │ completed │ 2026-01-10T17:52:07.208085+00:00 │ 2      │ set1:lexA,cpxR │ sites      │
│ parse_set1_lexA-cpxR_20260110_125126_67336d  │ parse  │ completed │ 2026-01-10T17:51:26.301780+00:00 │ 2      │ set1:lexA,cpxR │ sites      │
└──────────────────────────────────────────────┴────────┴───────────┴──────────────────────────────────┴────────┴────────────────┴────────────┘
```

For live progress, you can watch the run status in another terminal:

```bash
uv run cruncher -c "$CONFIG" runs watch <run_name>
```

To plot live trends as PNGs while sampling:

```bash
uv run cruncher -c "$CONFIG" runs watch <run_name> --plot
```

To widen the live metrics window:

```bash
uv run cruncher -c "$CONFIG" runs watch <run_name> --metric-points 80 --metric-width 40
```

## Run artifacts and performance snapshot

Use `runs show` to inspect what a run produced:

```bash
uv run cruncher -c "$CONFIG" runs show sample_set1_lexA-cpxR_20260110_160900_58723f
```

Example output (abridged):

```bash
run: sample_set1_lexA-cpxR_20260110_160900_58723f
stage: sample
status: completed
created_at: 2026-01-10T21:09:00.008157+00:00
motif_count: 2
regulator_set: {'index': 1, 'tfs': ['lexA', 'cpxR']}
pwm_source: sites
run_dir: /Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_basics_two_tf/runs/sample_set1_lexA-cpxR_20260110_160900_58723f
artifacts:
┏━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Stage  ┃ Type     ┃ Label                                  ┃ Path              ┃
┡━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ sample │ config   │ Resolved config (config_used.yaml)     │ config_used.yaml  │
│ sample │ trace    │ Trace (NetCDF)                         │ trace.nc          │
│ sample │ table    │ Sequences with per-TF scores (Parquet) │ sequences.parquet │
│ sample │ table    │ Elite sequences (Parquet)              │ elites.parquet    │
│ sample │ json     │ Elite sequences (JSON)                 │ elites.json       │
│ sample │ metadata │ Elite metadata (YAML)                  │ elites.yaml       │
└────────┴──────────┴────────────────────────────────────────┴───────────────────┘
```

Runtime scales with `draws`, `tune`, and `chains` in the config; adjust them to match your runtime/quality budget.

## Analyze + report

```bash
uv run cruncher -c "$CONFIG" analyze --latest
uv run cruncher -c "$CONFIG" report sample_set1_lexA-cpxR_20260110_160900_58723f
```

Example output (analyze):

```bash
Random baseline: 100%|██████████| 12/12 [00:00<00:00, 11583.81it/s]
Analysis outputs → /Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_basics_two_tf/runs/sample_set1_lexA-cpxR_20260110_160900_58723f/analysis
  summary: /Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_basics_two_tf/runs/sample_set1_lexA-cpxR_20260110_160900_58723f/analysis/summary.json
  analysis_id: 20260110T210926Z_9eb3c2
Next steps:
  cruncher runs show /Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml sample_set1_lexA-cpxR_20260110_160900_58723f
  cruncher notebook --latest /Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_basics_two_tf/runs/sample_set1_lexA-cpxR_20260110_160900_58723f
  cruncher report /Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml sample_set1_lexA-cpxR_20260110_160900_58723f
```

If you're running via `uv`, prefix those next-step commands with `uv run`.

## Optional: open the analysis notebook (real time)

```bash
uv run cruncher notebook /Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_basics_two_tf/runs/sample_set1_lexA-cpxR_20260110_160900_58723f --latest
```

Example output:

```bash
Notebook created →
/Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_basics_two_tf/runs/sample_set1_lexA-cpxR_20260110_160900_58723f/analysis/notebooks/run_overview.py
Open with: marimo edit
/Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_basics_two_tf/runs/sample_set1_lexA-cpxR_20260110_160900_58723f/analysis/notebooks/run_overview.py
Read-only app: marimo run
/Users/Shockwing/Dropbox/projects/phd/dnadesign/src/dnadesign/cruncher/workspaces/demo_basics_two_tf/runs/sample_set1_lexA-cpxR_20260110_160900_58723f/analysis/notebooks/run_overview.py
```

## Optional: use matrix mode (local motifs + alignment)

To use local motif matrices (or alignment matrices from RegulonDB), switch to
matrix mode. This preserves the full MEME motif length for DAP-seq sources (no
site windowing):

```yaml
motif_store:
  pwm_source: matrix
  source_preference: [demo_local_meme, regulondb]
```

Then fetch local MEME motifs (if you have not already):

```bash
uv run cruncher -c "$CONFIG" fetch motifs --source demo_local_meme --tf lexA --tf cpxR
```

Note: not all RegulonDB regulons ship alignment matrices. If `fetch motifs` fails
with “alignment matrix is missing”, use `pwm_source: sites` and `fetch sites`
instead.

## Optional: HT-only or combined site modes (RegulonDB)

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

If HT site lengths vary, use `uv run cruncher -c "$CONFIG" targets stats` and set
`motif_store.site_window_lengths` per TF or dataset before building PWMs.

## See also

- Command reference: [CLI reference](../reference/cli.md)
- Config knobs: [Config reference](../reference/config.md)
- Ingestion details: [Ingestion guide](../guides/ingestion.md)

## Where outputs live

- `.cruncher/demo_basics_two_tf/` - demo-local cache, lockfiles, and run index.
- `<out_dir>/` - parse/sample runs (this demo writes to `runs/`).

Sample run contents:

```
config_used.yaml
run_manifest.json
analysis/
report.md
```
