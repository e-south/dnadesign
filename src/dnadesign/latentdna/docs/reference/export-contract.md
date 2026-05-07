# Export Contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-11

`export matrix`, `export table`, and `export anndata` build deterministic
handoff bundles with:

- `rows.parquet`
- `features.parquet`
- `matrix.npy` or `table.parquet`
- `bundle.h5ad` for AnnData exports
- `manifest.json`

Current block kinds:

- `reduced_view`
- `table_columns`

Key constraints:

- row basis is explicit and may be a view, reduced view, sample, alignment, scalar table, or distance table artifact.
- block order is stable and encoded in `features.parquet`.
- alignment-backed block projection is explicit.
- feature names use deterministic semantic prefixes such as `intermediate_embedding_20b_merged_anchor_insert_seq_mean_pc_001`.
- feature names must be unique across the full bundle; ambiguous ledgers fail fast.

AnnData exports are an interoperability layer, not the LatentDNA runtime source
of truth. They reuse the same export block contract and write:

- `X`: the aligned numeric export matrix.
- `obs`: the row-basis ledger, enriched with declared `metadata_columns`.
- `var`: the deterministic feature ledger.
- `uns["latentdna_export"]`: workspace id, export id, row basis, block
  provenance, dtype, and schema version.
- optional `obsm` entries when `export anndata` is called with explicit
  `--projection <projection-id>` artifacts whose rows match the export basis.
- optional `obsp` distance graphs when called with explicit `--neighbor
  <neighbor-id>` artifacts whose rows match the export basis.

The command does not auto-discover projections or neighbor sets. Supplemental
AnnData slots must be requested explicitly so the manifest can record their
path-backed provenance and row-alignment checks can fail before writing.

See also:

- [deliverable-contract.md](deliverable-contract.md)
- [performance-budgets.md](performance-budgets.md)
