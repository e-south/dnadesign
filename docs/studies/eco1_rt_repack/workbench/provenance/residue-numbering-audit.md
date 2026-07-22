## Residue Numbering Audit

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-19
**Status:** numbering policy selected; local materialized residue map exists

Machine-readable policy:
`residue-numbering-policy.yaml`.

This audit will record the mapping among:

- structure residue ids
- canonical Eco1 RT protein positions
- Eco1 RT CDS codon positions
- design positions emitted to `thread`

The numbering policy follows the `ec86kit` protomer-1 chain-A amino-acid map
against the 320-aa Eco1/Ec86 RT reference sequence. The sibling map covers 309
residues, canonical positions 3-311. Terminal residues `1`, `2`, and `312-320`
are missing from the fixed-backbone structure and should be treated as
`non_fixed_missing_backbone`, not protected residues.

No mutable-position mask should be accepted until the planned residue-map
artifact has a complete mapping for every mutable residue.

### Required Mapping Columns

```text
canonical_position
wt_aa
structure_chain_id
structure_residue_id
pdb_insertion_code
cds_codon_index
design_position
mapping_status
mapping_issue
```

### Acceptance Criteria

- Phase 1 contract validation fails while this audit has no selected
  machine-readable policy.
- Phase 1 structure readiness now validates the local `residue_map.parquet`;
  mask readiness still fails until contact and conservation evidence profiles
  are materialized.
- Every candidate mutable residue has `mapping_status: mapped`.
- Every manual mask anchor resolves to at least one canonical position.
- PDB residue ids and canonical Eco1 positions remain separate columns.
- Any missing density, insertion code, or chain mismatch is encoded as a
  mapping issue and fixed or excluded before sampling.
- The residue map records the sequence hash used to generate it.
