## Eco1 RT Repack Status

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-24
**Status surface:** record-only

### Current Phase

Phase 2 backend ingest now passes locally. The study has the required
structure, source, alignment, conservation evidence, manual mask authority,
mask set, explicit thread plan, ProteinMPNN request, backend run manifest, and
sample table under `outputs/thread/eco1_rt_conservative_v1/`. The selected mask
rule is:

```text
eco1_rt_clade9_plurality25_direct_contact5a_v1
```

The rule is:

```text
protected =
  NAxxH / YADD / VTG
  OR Wang/Ec86 direct substrate-contact prior
  OR Eco1 amino acid is evolutionarily conserved at >=25% WT plurality in the Ec86 clade 9 MSA
  OR mapped residue is within 5 A of retained DNA/RNA

non_fixed = NOT protected
```

Terminal residues `1`, `2`, and `312-320` are not protected by this policy.
They are classified as `non_fixed_missing_backbone`: unprotected, but not
directly mutable by fixed-backbone ProteinMPNN until coordinates are supplied or
handled separately.

### Materialized Evidence

- Structure authority is `ec86kit_7v9u_protomer1`: PDB `7v9u`,
  RT chain `A`, retained DNA chain `D`, retained RNA chains `E/F`, paired
  protomer excluded, and no paired-protomer dimerization retention objective.
- `backbone_bundle.yaml` and `residue_map.parquet` are materialized. The
  selected fixed-backbone structure has 309 mapped positions and 11 missing
  terminal positions: `1`, `2`, `312-320`.
- `structure_preprocessing_manifest.yaml`, `contact_profile.parquet`, and
  `contact_geometry_profile.parquet` are materialized from the selected 7V9U /
  ec86kit protomer context.
- Mestre-derived source authority covers
  `ec86_clade9_conservation_v1` and
  `ec86_iia3_cluster42_1_conservation_v1`; the full Mestre roster remains
  context/candidate-pool evidence, not the conservation denominator.
- Provider source acquisition, source-record QC, source FASTA sufficiency,
  Clustal Omega alignments, and `conservation_profile.parquet` are available
  for both selected profiles.
- `conservation_profile.parquet` has 640 rows: 320 positions per selected
  profile. The mask rule uses `ec86_clade9_conservation_v1` as the
  conservation veto: Eco1 residues are protected when the Eco1 amino acid is
  the clade 9 plurality residue at frequency `>=25%`.
- Manual motif authority records NAxxH `105-109`, YADD `195-198`, and VTG
  `243-245` as protected anchors. RT1-RT7 intervals remain annotation/review
  labels and do not blanket hard-fix residues under this rule.
- Wang/Ec86 direct substrate-contact priors are protected when listed in
  `manual-mask-authority.yaml`.
- `mask_set.yaml` is materialized under
  `eco1_rt_clade9_plurality25_direct_contact5a_v1`; Phase 1 validates locally.
- `thread_plan.yaml` is materialized locally with explicit `proteinmpnn`
  backend selection, seeds `101`, `202`, `303`, temperatures `0.1` and `0.3`,
  a request hash, and `explicit_no_fallback` policy. The plan emits 123 mapped
  mutable positions and excludes terminal `non_fixed_missing_backbone`
  positions from fixed-backbone mutation.
- `proteinmpnn_request/request_manifest.yaml` is materialized locally. The Eco1
  wrapper resolves study paths and selected structure provenance, then calls
  `dnadesign.thread.adapters.proteinmpnn` for the protein-only chain export,
  chain-local position mapping, helper-compatible JSONL sidecars, request
  hashes, and generic request validation. The request declares `--omit_AAs C`
- Official ProteinMPNN commit `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57` was
  installed locally under `.var/tools/proteinmpnn` and used through an explicit
  `--proteinmpnn-root` path. The active backend batch is
  `eco1_rt_p25_5a_n96_20260624`: seeds `101`, `202`, `303`, temperatures
  `0.1` and `0.3`, and `num_seq_per_target: 16`.
- `sample_table.parquet` is materialized locally with 96 accepted ProteinMPNN
  rows. The named batch table is also retained at
  `sample_tables/eco1_rt_p25_5a_n96_20260624.parquet`.
- `candidate_table.parquet` is materialized locally with 96 accepted candidate
  rows and no protected-position or outside-mutable-position mutations. The
  named batch table is also retained at
  `candidate_tables/eco1_rt_p25_5a_n96_20260624.parquet`.

### Mask Counts

Applying the current rule gives this row classification:

| Class | Count |
| --- | ---: |
| `non_fixed` mapped residues | 123 |
| `non_fixed_missing_backbone` terminal residues | 11 |
| evolutionarily conserved at >=25% WT plurality in the Ec86 clade 9 MSA | 106 |
| within 5 A retained DNA/RNA | 120 |
| NAxxH / YADD / VTG | 12 |
| Wang/Ec86 direct substrate-contact priors | 8 |

Total unprotected positions: `134`. Directly fixed-backbone ProteinMPNN mutable
positions from the current 7V9U backbone: `123`.

### Prior Mask Checks

`contact_risk_profile.yaml` remains an evidence review. It does not protect or
release residues under the current mask.

The previous 20 A all-fixed mask is diagnostic history: it showed that broad
retained-nucleic-acid proximity fixes the whole RT and is therefore too blunt
for Eco1. The current `mask_set.yaml` uses the direct 5 A rule.

### Validator Commands

Phase 0 scaffold validation:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase0_scaffold
```

Phase 1 contract validation:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase1_thread_contract
```

Phase 2 backend-ingest validation:

```bash
uv run python -m dnadesign.studies.units.eco1_rt_repack.operations.contract_validation --repo-root . --phase phase2_real_backend_ingest
```

### Current Next Actions

1. Define fold-check and feasibility gates for accepted candidate rows.
2. Define the downstream RT-lnRNA candidate handoff accepted by
   `rt_lnrna_sponging_construct_triage`.

### Blockers

- `dnadesign.thread` now exposes generic ProteinMPNN request, sample-ingest,
  and candidate-table mechanics. Fold-check normalization, feasibility, and
  handoff tooling remain planned.
- No fold-check runtime report with WT baseline, thresholds, and runtime
  parameter hash exists.
- No assembly feasibility report exists.
- No RT-only candidate handoff or RT-lnRNA acceptance record exists.

### Non-Goals

- Wet-lab protocol execution.
- Prime-editing campaign ownership.
- Replacing the RT-lnRNA sponging construct study.
- Hiding Eco1-specific biology inside a reusable tool package.
