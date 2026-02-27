## Ingestion

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


**Last updated by:** cruncher-maintainers on 2026-02-23

### Contents
- [Ingestion](#ingestion)
- [How ingestion works](#how-ingestion-works)
- [Cache layout](#cache-layout)
- [General normalization rules](#general-normalization-rules)
- [RegulonDB](#regulondb)
- [Local motif directories](#local-motif-directories)
- [Local binding-site FASTA sources](#local-binding-site-fasta-sources)
- [Curated TF binding sites](#curated-tf-binding-sites)
- [High-throughput datasets](#high-throughput-datasets)
- [Hydration](#hydration)
- [Fetching data](#fetching-data)
- [PWM creation strategy](#pwm-creation-strategy)
- [Common issues](#common-issues)

This guide describes how Cruncher ingests, normalizes, and caches motif and site data.

#### How ingestion works

1. **Fetch** raw payloads from a source adapter (for example RegulonDB).
2. **Normalize** into `MotifRecord` and `SiteInstance`.
3. **Cache** normalized records under the project-local `.cruncher/` directory.
4. **Index** everything in `catalog.json` so **cruncher** can answer "what do we have?"

---

#### Cache layout

```
<catalog.root>/
  catalog.json
  normalized/
    motifs/<source>/<motif_id>.json
    sites/<source>/<motif_id>.jsonl
  discoveries/      # MEME/STREME discovery runs (optional)
```

Workspace state (per workspace `.cruncher/`):

```
<workspace>/.cruncher/
  locks/
    <config>.lock.json
  run_index.json
```

`catalog.json` is the local source of truth for cached motifs and sites. `discoveries/` is created only when you run
`cruncher discover motifs`. By default, the catalog cache is workspace-local (`<workspace>/.cruncher`).
Matplotlib cache is shared at `.cache/matplotlib/cruncher` unless `MPLCONFIGDIR` is set.
`run_index.json` tracks known runs for the workspace.

---

#### General normalization rules

- **Sequences** must be A/C/G/T only (invalid sequences are rejected).
- **Coordinates** are stored as 0-based, half-open intervals.
- **Hydration** is required when HT peaks have coordinates but no sequences.
- **Site length variability**: if site lengths vary, set
  `catalog.site_window_lengths` per TF or dataset before building PWMs.

---

#### RegulonDB

**cruncher** queries the RegulonDB Datamarts GraphQL endpoint:

```
https://regulondb.ccg.unam.mx/graphql
```

Notes: **cruncher** uses the default trust store plus a bundled RegulonDB intermediate certificate. If the server rotates its chain, set `ingest.regulondb.ca_bundle`. If the API returns internal errors, **cruncher** fails fast; retry later or work from cache (`cruncher sources summary --scope cache`).

---

#### Local motif directories

Local motif sources let you register on‑disk datasets as first‑class sources. Each file becomes a cached motif entry, with TF names derived from the filename stem by default. MEME files can also expose MEME BLOCKS sites.

Key behaviors:

- **Root validation**: missing roots or empty matches fail fast.
- **Explicit parsing**: you must provide `format_map` and/or `default_format`.
- **TF naming**: default is file stem (preserves case), configurable via `tf_name_strategy`.
- **Provenance**: dataset metadata (DOI, comments) lives in config tags/citation, not code.
- **Path resolution**: relative roots are resolved from the workspace root.
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

#### Local binding-site FASTA sources

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

#### Curated TF binding sites

Curated binding sites are fetched from the regulon datamart and cached as `SiteInstance` records. If you set `ingest.regulondb.motif_matrix_source: alignment`, **cruncher** uses the alignment payload when present; otherwise it fails fast.

---

#### High-throughput datasets

HT datasets (ChIP‑seq, ChIP‑exo, DAP‑seq, gSELEX) are discovered and fetched via dataset queries:

- `cruncher sources datasets regulondb <config> [--tf <TF>]`
- `cruncher fetch sites --dry-run --tf <TF> <config>`

When multiple HT datasets exist for a TF, pin selection with:

- `catalog.dataset_map` (TF -> dataset ID, strongest)
- `catalog.dataset_preference` (ordered preference list)

Lockfiles record the chosen dataset ID for reproducibility.

`ingest.regulondb.ht_dataset_type` is validated against the remote
`listAllDatasetTypes` response and fails fast if the value is unknown.

HT contracts are strict (no curated fallback):
- if `ingest.regulondb.ht_sites: true`, discovery/fetch failures raise errors
- if HT returns zero records for the selected mode, fetch raises errors
- `sources datasets --dataset-source <X>` applies a row-level source filter after remote fetch to guard against mixed-source payloads

If you enable both `curated_sites` and `ht_sites`, avoid mixed-mode limits:
- `fetch sites --limit <N>` without `--dataset-id` is rejected because curated rows can consume the limit before HT rows.
- Use one explicit mode per request: pin `--dataset-id`, disable one source class, or omit `--limit`.

---

#### Hydration

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

#### Fetching data

- `cruncher fetch motifs --tf <TF> <config>` -> caches matrices.
- `cruncher fetch sites --tf <TF> <config>` -> caches site sets.
- `cruncher fetch sites --hydrate <config>` -> hydrate missing sequences only (all cached site sets by default).
- `cruncher fetch sites --dataset-id <id> <config>` -> pin a specific HT dataset
  (also enables HT access for this request).
- If HT `tfbinding` returns no rows for a known dataset, set `ingest.regulondb.ht_binding_mode: peaks`
  and provide FASTA hydration (`ingest.genome_fasta` or `--genome-fasta`) when sequences are not returned.
- `--source` defaults to the first available entry in `catalog.source_preference` (skipping entries that are
  not registered ingest sources); if the list is empty or none are available you must pass `--source` explicitly.
- `--offline` validates cache without network.
- `--update` forces refresh of cached artifacts.

---

#### PWM creation strategy

- `catalog.pwm_source=matrix` uses cached matrices (default).
- `catalog.pwm_source=sites` builds PWMs from cached binding sites.
- `min_sites_for_pwm` is enforced unless `allow_low_sites=true`.

---

#### Common issues

- Missing lockfile: run `cruncher lock <config>` before parse/sample.
- Target readiness: `cruncher targets status <config>`.
- Missing cache files: `cruncher cache verify <config>`.
- HT sites without sequences: set `ingest.genome_source` or use `--genome-fasta`.
- Variable site lengths: use `cruncher targets stats` and set
  `catalog.site_window_lengths`.

---

@e-south
