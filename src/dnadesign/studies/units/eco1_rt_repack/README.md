## Eco1 RT Repack Study Unit

This package owns Eco1-specific fixed-backbone study logic:

- selected 7V9U/Ec86 structure provenance
- Mestre-derived conservation source policy
- Wang/Ec86 direct substrate-contact priors
- NAxxH, YADD, and VTG motif protection
- the active residue mask rule
- study-local artifact materialization and contract validation

Reusable ProteinMPNN request mechanics live in
`dnadesign.thread.adapters.proteinmpnn`. Eco1 resolves study paths and
biological policy, then calls that public adapter for chain-local positions,
helper JSONL sidecars, protein-only backbone export, request hashes, and
generic request validation.

Reusable fold-check request/report contracts live in
`dnadesign.thread.foldcheck`. Eco1 reconstructs the full canonical WT and
candidate sequences, declares the first ColabFold/AlphaFold-family request, and
keeps heavy fold-model execution outside this package.

## Current Artifact Ladder

Run materializers as independent steps. Do not hand-edit generated files under
`outputs/thread/eco1_rt_conservative_v1/`.

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure --repo-root .
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure_preprocessing --repo-root .
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact --repo-root .
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry --repo-root .
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences --repo-root .
pixi run uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.conservation_alignments --repo-root .
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.conservation --repo-root .
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.manual_mask_authority --repo-root .
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.mask_set --repo-root .
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.thread_plan --repo-root .
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.proteinmpnn_request --repo-root .
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.proteinmpnn_sample_ingest --repo-root . --proteinmpnn-root .var/tools/proteinmpnn --overwrite
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.candidate_table --repo-root .
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_request --repo-root .
```

Validate the current gates:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase0_scaffold
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase1_thread_contract
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase2_real_backend_ingest
```

Phase 2 passes after `sample_table.parquet`,
`proteinmpnn_outputs/backend_run_manifest.yaml`, and `candidate_table.parquet`
are materialized. The next fold-check step starts from
`foldcheck_request/foldcheck_request_manifest.yaml`.

## Source Layout

- `operations/contracts/`: study contract validation, split into semantic
  packages for `conservation`, `contact_risk`, `foldcheck`, `masks`,
  `sampling`, and `structure`.
- `operations/materialization/<primitive>/`: one runtime artifact family per
  package. CLI parsing stays in `cli.py`; artifact behavior stays in
  `pipeline.py` and narrower helper modules.
- `operations/masking/`: executable Eco1 mask-row algebra for protected,
  non-fixed mapped, and non-fixed missing-backbone rows.
- `tests/contracts/` and `tests/materialization/<primitive>/`: test packages
  mirror source ownership. Do not add flat study-root test modules.

The current mask rule is
`eco1_rt_clade9_plurality25_direct_contact5a_v1`: protect NAxxH/YADD/VTG,
Wang/Ec86 direct substrate-contact priors, Ec86 clade 9 positions with
`>=25%` WT plurality conservation, and mapped residues within `5 A` of retained
DNA/RNA. RT1-RT7 spans are review labels, not blanket fixed residues. Terminal
residues `1`, `2`, and `312-320` are unprotected but lack selected backbone
coordinates, so they are excluded from direct fixed-backbone ProteinMPNN
mutation.
