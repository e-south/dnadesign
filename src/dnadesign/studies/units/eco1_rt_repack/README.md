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

The current executable path is study-owned through Phase 1 artifact
materialization and validation. The first smoke check is the contract
validator:

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

The provider-cache input to that slice is materialized through two explicit
study-owned primitives:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.provider_sources \
  --repo-root . \
  --roster-table <mestre-s1-roster.xlsx> \
  --write-unresolved-ledger
```

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.roster_cache \
  --repo-root . \
  --roster-table <mestre-s1-roster.csv-or-xlsx> \
  --provider-source-root <provider-fasta-source-root> \
  --provider-failure-ledger <provider-fasta-source-root>/provider_source_failures.yaml
```

After source sufficiency passes, conservation alignment runs through the public
`dnadesign.aligner.msa` seam:

```bash
pixi run uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.conservation_alignments --repo-root .
```

The conservation profile, manual mask authority, and current diagnostic mask
are materialized separately:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.conservation --repo-root .
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.manual_mask_authority --repo-root .
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.mask_set --repo-root .
```

Do not add ProteinMPNN, LigandMPNN, AlphaFold, ColabFold, mask algebra,
candidate-ranking, or fold-normalization implementations here unless they are
explicitly study-only and cannot graduate to `dnadesign.thread`.

Source code is organized by ontology:

- `operations/contracts/`: shared contract orchestration plus semantic
  subpackages for `conservation`, `structure`, and `masks`. Domain validators
  should not be added as flat root modules.
- `operations/materialization/<primitive>/`: study-owned runtime artifact
  materializers split by primitive: `structure`, `contact`, `conservation`,
  `conservation_alignments`, `manual_mask_authority`, `mask_set`, and
  `source_sequences`. Nested packages under a primitive own narrower ontology
  layers such as `source_sequences/contracts`,
  `source_sequences/provider_sources`, `source_sequences/roster_cache`, and
  `source_sequences/sufficiency`.
- `operations/masking/`: study-local mask row algebra shared by the mask
  materializer and Phase 1 validators.
- Source-sequence provider accession shapes live in
  `source_sequences/contracts/` and are compiled from
  `conservation-sources.yaml`; do not duplicate provider regexes in
  materializers or validators.
- CLI parsing belongs in each materialization package's `cli.py`; `pipeline.py`
  owns domain behavior and should remain callable without command-line parsing.
- `tests/contracts/` and `tests/materialization/<primitive>/`: tests mirror
  the source ownership boundaries, including nested
  `tests/materialization/source_sequences/<subprimitive>/` packages for
  provider-source, roster-cache, contract, and sufficiency checks.

This package should stay narrow: validate the selected Eco1 structure authority
and residue-numbering policy, materialize `backbone_bundle.yaml` and
`residue_map.parquet`, materialize `contact_profile.parquet` from the retained
DNA/RNA context, materialize explicit source FASTA bundles before alignment,
score accepted alignments into `conservation_profile.parquet`, and materialize
the current all-fixed diagnostic `mask_set.yaml`. Generic runtime artifact-chain
validation belongs in `dnadesign.thread` once that package exists.
