## cruncher ingestion

Ingestion is how **cruncher** discovers and caches motif matrices and binding sites from external sources.

### How ingestion works

1. **Fetch** raw payloads from a source adapter (for example RegulonDB).
2. **Normalize** into `MotifRecord` and `SiteInstance`.
3. **Cache** normalized records under the project-local `.cruncher/` directory.
4. **Index** everything in `catalog.json` so **cruncher** can answer "what do we have?"

---

### Cache layout

```
<catalog_root>/
  catalog.json
  normalized/
    motifs/<source>/<motif_id>.json
    sites/<source>/<motif_id>.jsonl
  discoveries/      # MEME/STREME discovery runs (optional)
  .mplcache/         # Matplotlib cache (auto-managed)
```

Workspace state (per workspace `.cruncher/`):

```
<workspace>/.cruncher/
  locks/
    <config>.lock.json
  run_index.json
```

`catalog.json` is the local source of truth for cached motifs and sites. `discoveries/` is created only when you run
`cruncher discover motifs`. By default, the catalog cache is shared across workspaces (`src/dnadesign/cruncher/.cruncher`).
`run_index.json` tracks known runs for the workspace.

---

### General normalization rules

- **Sequences** must be A/C/G/T only (invalid sequences are rejected).
- **Coordinates** are stored as 0-based, half-open intervals.
- **Hydration** is required when HT peaks have coordinates but no sequences.
- **Site length variability**: if site lengths vary, set
  `motif_store.site_window_lengths` per TF or dataset before building PWMs.

---

### RegulonDB

**cruncher** queries the RegulonDB Datamarts GraphQL endpoint:

```
https://regulondb.ccg.unam.mx/graphql
```

Notes: **cruncher** uses the default trust store plus a bundled RegulonDB intermediate certificate. If the server rotates its chain, set `ingest.regulondb.ca_bundle`. If the API returns internal errors, **cruncher** fails fast; retry later or work from cache (`cruncher sources summary --scope cache`).

---

### Local motif directories

Local motif sources let you register on‑disk datasets as first‑class sources. Each file becomes a cached motif entry, with TF names derived from the filename stem by default. MEME files can also expose MEME BLOCKS sites.

Key behaviors:

- **Root validation**: missing roots or empty matches fail fast.
- **Explicit parsing**: you must provide `format_map` and/or `default_format`.
- **TF naming**: default is file stem (preserves case), configurable via `tf_name_strategy`.
- **Provenance**: dataset metadata (DOI, comments) lives in config tags/citation, not code.
- **Path resolution**: relative roots are resolved from the config file location.
- **Sites opt-in**: set `extract_sites=true` to parse MEME BLOCKS sites (training-set occurrences).
- **Motif selection**: use `meme_motif_selector` to disambiguate multi-motif MEME files.

Example config (DAP-seq MEME files on disk):

```yaml
ingest:
  local_sources:
    - source_id: omalley_ecoli_meme
      description: O'Malley et al. E. coli MEME motifs (Supplementary Data 2)
      root: /path/to/dnadesign-data/primary_literature/OMalley_et_al/escherichia_coli_motifs
      patterns: ["*.txt"]
      recursive: false
      format_map: {".txt": "MEME"}
      tf_name_strategy: stem
      matrix_semantics: probabilities
      extract_sites: false
      meme_motif_selector: null  # name_match | MEME-1 | 1 | "<MOTIF label>"
      citation: "O'Malley et al. 2021 (DOI: 10.1038/s41592-021-01312-2)"
      source_url: https://github.com/e-south/dnadesign-data
      tags:
        doi: 10.1038/s41592-021-01312-2
        title: "Persistence and plasticity in bacterial gene regulation"
        association: "TF-gene interactions"
        comments: "DAP-seq (DNA affinity purification sequencing) motifs across 354 TFs in 48 bacteria (~17,000 binding maps)."
```

Example CLI flow:

- `cruncher sources list <config>`
- `cruncher fetch motifs --source omalley_ecoli_meme --tf lexA <config>`
- `cruncher lock <config>`
- `cruncher parse <config>`

If you need custom parsing, register your parser module via `io.parsers.extra_modules` and map extensions to your format.

Use `source_url` plus `tags` to keep provenance, and swap in your own on‑disk dataset as needed.

---

### Local binding-site FASTA sources

Local site sources register binding-site sequences from a FASTA file as a first‑class source. Each header line
should start with a TF name, followed by `|`‑separated metadata. The sequence lines are used directly.

Example config (BaeR ChIP-exo FASTA on disk):

```yaml
ingest:
  site_sources:
    - source_id: baer_chip_exo
      description: Choudhary et al. BaeR ChIP-exo binding sites (processed FASTA)
      path: /path/to/dnadesign-data/primary_literature/Choudhary_et_al/processed/BaeR_binding_sites.fasta
      tf_name: BaeR
      record_kind: chip_exo
      citation: "Choudhary et al. 2020 (DOI: 10.1128/mSystems.00980-20)"
      source_url: https://doi.org/10.1128/mSystems.00980-20
      tags:
        assay: chip_exo
        doi: 10.1128/mSystems.00980-20
```

---

### Curated TF binding sites

Curated binding sites are fetched from the regulon datamart and cached as `SiteInstance` records. If you set `ingest.regulondb.motif_matrix_source: alignment`, **cruncher** uses the alignment payload when present; otherwise it fails fast.

---

### High-throughput datasets

HT datasets (ChIP‑seq, ChIP‑exo, DAP‑seq, gSELEX) are discovered and fetched via dataset queries:

- `cruncher sources datasets regulondb <config> [--tf <TF>]`
- `cruncher fetch sites --dry-run --tf <TF> <config>`

When multiple HT datasets exist for a TF, pin selection with:

- `motif_store.dataset_map` (TF -> dataset ID, strongest)
- `motif_store.dataset_preference` (ordered preference list)

Lockfiles record the chosen dataset ID for reproducibility.

`ingest.regulondb.ht_dataset_type` is validated against the remote
`listAllDatasetTypes` response and fails fast if the value is unknown.

---

### Hydration

Some HT datasets return coordinates without sequences. **cruncher** hydrates those coordinates using a reference genome.

Defaults:
- `ingest.genome_source=ncbi` (NCBI E-utilities)
- cached FASTA stored under `.cruncher/genomes/<accession>/`

Offline:
- set `ingest.genome_source=fasta` and provide `ingest.genome_fasta`
- or pass `--genome-fasta` to `cruncher fetch sites`

Hydration is strict:
- contig names must exist in the FASTA
- coordinates must be in bounds
- assembly/accession must be known or provided

Use `ingest.contig_aliases` if contig labels differ.

PWM generation from sites is also strict: cached site records must include sequences.
If any cached site lacks a sequence, `parse`/`sample` will error until you hydrate
those sites (for example via `cruncher fetch sites --hydrate <config>`).

---

### Fetching data

- `cruncher fetch motifs --tf <TF> <config>` -> caches matrices.
- `cruncher fetch sites --tf <TF> <config>` -> caches site sets.
- `cruncher fetch sites --hydrate <config>` -> hydrate missing sequences only (all cached site sets by default).
- `cruncher fetch sites --dataset-id <id> <config>` -> pin a specific HT dataset
  (also enables HT access for this request).
- `--offline` validates cache without network.
- `--update` forces refresh of cached artifacts.

---

### PWM creation strategy

- `motif_store.pwm_source=matrix` uses cached matrices (default).
- `motif_store.pwm_source=sites` builds PWMs from cached binding sites.
- `min_sites_for_pwm` is enforced unless `allow_low_sites=true`.

---

### Common issues

- Missing lockfile: run `cruncher lock <config>` before parse/sample.
- Target readiness: `cruncher targets status <config>`.
- Missing cache files: `cruncher cache verify <config>`.
- HT sites without sequences: set `ingest.genome_source` or use `--genome-fasta`.
- Variable site lengths: use `cruncher targets stats` and set
  `motif_store.site_window_lengths`.

---

@e-south
