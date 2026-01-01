# Ingestion

Adapters normalize source payloads into `MotifRecord` and `SiteInstance`.

## Local catalog cache

Cruncher writes a project-local cache under `.cruncher/` (configurable via `motif_store.catalog_root`):

- `normalized/motifs/<source>/<motif_id>.json` (motif matrix + provenance)
- `normalized/sites/<source>/<motif_id>.jsonl` (binding-site instances)
- `catalog.json` (index of what’s cached: TFs, matrices, site counts)

This index is how Cruncher lists “what we have in-house” without touching the network.

`cruncher catalog list` renders one line per cached motif and includes:

- matrix availability (`matrix` vs `no-matrix`)
- site availability (`sites:<n>` vs `no-sites`)
- matrix provenance tags (e.g., `matrix:alignment` vs `matrix:sites`)
- organism metadata when provided by the source

## RegulonDB (GraphQL)

Cruncher queries the RegulonDB Datamarts GraphQL endpoint:

```
https://regulondb.ccg.unam.mx/graphql
```

Cruncher ships the current RegulonDB intermediate CA bundle so SSL verification works out of the box.
Override with `ingest.regulondb.ca_bundle` only if the server rotates certificates.

### Curated regulon TFBS

We use the **regulon** datamart to retrieve curated binding sites:

```
getRegulonBy(search) → RegulonDatamart
  regulator { name, abbreviatedName }
  regulatoryInteractions { regulatoryBindingSites { _id, leftEndPosition, rightEndPosition, strand, sequence } }
  aligmentMatrix { matrix, aligment, consensus }
```

Important schema notes:

- `regulatoryBindingSites` is a **single object** per interaction (not a list).
- Binding-site `sequence` sometimes contains uppercase motif letters embedded in lowercase flanks.
- `aligmentMatrix` may contain either a precomputed matrix (`matrix`) or aligned sequences (`aligment`).

### High-throughput TF-binding datasets

RegulonDB provides HT datasets (ChIP-seq/ChIP-exo/DAP/gSELEX), which are accessed through:

```
listAllHTSources → ["BAUMGART", "PALSSON", "ISHIHAMA", ...]
getDatasetsWithMetadata(datasetType: "TFBINDING", source: <HT_SOURCE>)
  datasets { _id, objectsTested { name, abbreviatedName, synonyms } }
getAllTFBindingOfDataset(datasetId, limit, page)
  → DatasetTFBinding { _id, chromosome, chrLeftPosition, chrRightPosition, strand, sequence, score, peakId }
getAllPeaksOfDataset(datasetId, limit, page)
  → Peaks { _id, chromosome, peakLeftPosition, peakRightPosition, score, siteIds }
```

This path is used because it deterministically enumerates HT datasets without relying on
the advancedSearch syntax.

When multiple HT datasets exist for the same TF, use:

- `cruncher sources datasets regulondb --tf <TF>` to inspect dataset IDs/methods,
- `motif_store.dataset_map` to pin a TF to a specific dataset,
- `motif_store.dataset_preference` to rank preferred datasets (first match wins).

Lockfiles store the chosen dataset ID for reproducibility.

Pragmatic note: some HT datasets return peaks only (coordinates, no sequences).
Cruncher resolves the dataset’s `referenceGenome` / `assemblyGenomeId` and hydrates
sequences from NCBI by accession (default). The genome FASTA is cached under
`.cruncher/genomes/<accession>/` and reused across runs. Hydration is strict:

- the HT coordinate must include an assembly/contig accession,
- the contig must exist in the fetched FASTA,
- coordinates must be in-bounds.

If any of these are missing, Cruncher fails fast with a remediation hint. You can
override hydration with a local FASTA by setting `genome_source=fasta` and providing
`ingest.genome_fasta`. Hydration uses `pysam`’s FASTA indexer (no bespoke parsing).
If you request HT sites and no TFBinding **or** peaks exist, Cruncher fails fast.

When RegulonDB reports `chromosome` as a generic label (e.g., `chr`), Cruncher uses
the dataset’s `referenceGenome` accession as the contig name to ensure NCBI hydration
is deterministic and verifiable.
If you need to override contig labels, set `ingest.contig_aliases` in the config.

## Normalization rules

- **Binding-site sequence**:
  - If mixed case, keep uppercase letters only (motif letters).
  - Otherwise, take the full sequence.
  - All sequences must be A/C/G/T only; invalid sequences are rejected.
- Coordinate-only HT peaks must be hydrated into sequences before PWM creation.
- HT site sets can contain variable-length sequences; use `cruncher targets stats` to inspect
  length distributions and set `motif_store.site_window_lengths` per TF or dataset.
- Hydration tags each site with `sequence_source` and `reference_genome` (when available).
- **Alignment matrix**:
  - Accepts JSON matrices, base-labeled rows, or 4-column row matrices.
  - If only an alignment payload is available, PWM is computed directly from aligned sequences.
- **Coordinates**:
  - RegulonDB positions are treated as **1-based inclusive** and converted to 0-based half-open.
  - The original convention is recorded in provenance tags.

## Configuration knobs

See `docs/config.md` for the `ingest.regulondb` block. The important toggles are:

- `motif_matrix_source`: `alignment` (default) or `sites`
- `alignment_matrix_semantics`: `probabilities` or `counts`
- `curated_sites` / `ht_sites`: enable curated and/or HT TFBS retrieval
- `ht_binding_mode`: `tfbinding` (strict) or `peaks` (peaks only)

`motif_store.site_window_lengths` and `motif_store.site_window_center` live in the
catalog config block, not the source adapter, so other HT sources can reuse the same
windowing contract.

## Fetching data

- `cruncher fetch motifs --tf <TF> <config>` stores motif matrices under `normalized/motifs/`.
- `cruncher fetch sites --tf <TF> <config>` stores binding-site instances under `normalized/sites/`.
- `cruncher fetch sites --dry-run --tf <TF> <config>` lists HT datasets without caching.
- By default, coordinate-only HT peaks are hydrated via NCBI (see `ingest.genome_source`).
- Use `--genome-fasta` to hydrate against a local reference genome (offline runs).
- Use `--hydrate` to fill missing sequences in cached site sets without refetching.
- Add `--offline` to verify cache only (no network).
- Add `--update` to refresh cached artifacts (including any downloaded genome FASTA).

## PWM creation strategy

- `motif_store.pwm_source=matrix` uses cached motif matrices (default).
- `motif_store.pwm_source=sites` builds PWMs from cached binding-site sequences at runtime.
- If fewer than `min_sites_for_pwm` sequences are available, Cruncher fails unless `motif_store.allow_low_sites=true`.
- PWM construction from sites streams counts to avoid loading all sequences into memory.

## Adding a new source adapter (checklist)

1. Implement a `SourceAdapter` in `ingest/adapters/` with:
   - `list_motifs`, `get_motif`, `list_sites`, `get_sites_for_motif`
   - Optional `list_datasets` if the source exposes HT datasets
2. Normalize payloads into `MotifRecord` / `SiteInstance` with clear provenance tags.
3. Add adapter registration to `ingest/registry.py`.
4. Provide fixture-backed tests under `tests/fixtures/<source>/` (no live network in CI).
5. Update `docs/ingestion.md` with source capabilities and caveats.
