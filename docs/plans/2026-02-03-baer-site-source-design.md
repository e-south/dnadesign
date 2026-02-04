# BaeR Site Source (FASTA) Design

## Goal

Add a minimal local binding-site source that ingests the processed BaeR ChIP-exo FASTA
from `dnadesign-data` and participates in the existing site merge behavior alongside
RegulonDB and local DAP-seq sources.

## Scope

In scope:
- `ingest.site_sources` config schema and docs.
- A site-only local adapter that reads FASTA headers and sequences.
- Registry wiring, demo config updates, and basic provenance notes.
- Focused TDD coverage for adapter parsing and cache updates.

Out of scope:
- Genomic hydration or coordinate parsing.
- Motif discovery algorithm changes.
- Backward-compat paths or new CLI subcommands.

## Schema

`ingest.site_sources` is a list of site-only sources with fields:
- `source_id`, `path`, optional `tf_name`
- `record_kind` (becomes cached `site_kind`)
- provenance fields (`citation`, `source_url`, `tags`, `organism`)

FASTA headers must start with a TF name (e.g., `BaeR|peak_0|...`).
The adapter stores header metadata in `evidence` and uses sequences directly.

## Data Flow

`cruncher fetch sites` → LocalSiteAdapter yields `SiteInstance` records →
`fetch_service` writes JSONL under `normalized/sites/<source>/<tf>.jsonl` and
updates `catalog.json`. With `motif_store.combine_sites=true`, site sets
merge by TF name during discovery and site-derived PWM workflows.

## Tests

- Adapter parses FASTA headers and preserves key evidence fields.
- Fetching sites updates catalog counts and `site_kind` from `record_kind`.
