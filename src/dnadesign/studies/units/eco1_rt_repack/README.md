## Eco1 RT Repack Study Unit

This package is reserved for study-owned implementation helpers once
`eco1_rt_repack` moves beyond checked-in planning records.

Planned responsibilities:

- Eco1 profile parsing and validation.
- Study-owned candidate handoff validation.
- Study-owned RT-only downstream acceptance checks.
- Downstream RT-lnRNA promotion checks.
- Study-specific readiness summaries over checked-in record files.

Reusable fixed-backbone mechanics belong in a future `dnadesign.thread` package,
not in this study unit.

The current executable slice is a study-owned contract validator:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase0_scaffold
```

The current materialization slice is study-owned and emits only the selected
structure primitives:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure --repo-root .
```

The current evidence slice is study-owned and emits only the retained-context
contact profile:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact --repo-root .
```

The current conservation source slice is study-owned and emits unaligned source
FASTA bundles from explicit local provider caches:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences --repo-root .
```

Do not add ProteinMPNN, LigandMPNN, AlphaFold, ColabFold, mask algebra,
candidate-ranking, or fold-normalization implementations here unless they are
explicitly study-only and cannot graduate to `dnadesign.thread`.

Source code is organized by ontology:

- `operations/contracts/`: checked-in profile, artifact-chain, source,
  structure-authority, structure-artifact, evidence-artifact, and mask-case
  validators.
- `operations/materialization/<primitive>/`: study-owned runtime artifact
  materializers split by primitive: `structure`, `contact`, and
  `source_sequences`, and `conservation`.
- `tests/contracts/` and `tests/materialization/<primitive>/`: tests mirror
  the source ownership boundaries.

This package should stay narrow: validate the selected Eco1 structure authority
and residue-numbering policy, materialize `backbone_bundle.yaml` and
`residue_map.parquet`, materialize `contact_profile.parquet` from the retained
DNA/RNA context, materialize explicit source FASTA bundles before alignment,
then reject missing conservation and mask artifacts when a Phase 1 or later
contract is being validated. Generic runtime artifact-chain validation belongs
in `dnadesign.thread` once that package exists.
