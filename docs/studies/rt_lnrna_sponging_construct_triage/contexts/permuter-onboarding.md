## Permuter Onboarding For RT-CDS DMS

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-25

This study uses Permuter only for generic RT-CDS variant generation. The
RT-lnRNA study owns biological promotion into the 2,000 bp dual-cassette
Construct contract.

### Boundary

- Permuter owns RT-CDS DMS variant intent: codon position, alternate amino acid,
  top-codon policy, request id, variant id, and modification provenance.
- The study owns the construct subject envelope: parent construct subject,
  `lnrna` and `rt_cds` slot sequences, source basis, study role, and
  construct-promotion status.
- Construct owns the 2,000 bp context realization and six explicit sequence
  views.
- Infer owns Evo2 feature-bundle execution and `_derived/infer` sidecars.

Do not add RT-lnRNA slot, Khan/Crawford overlay, Reader label, OPAL table, or
100 bp context semantics to Permuter.

### Active Contract

The active construct context is the 2,000 bp dual-cassette contract:

- construct dataset: `rt_lnrna_sponging_construct_triage_construct_contexts_2000bp_v1`
- slot input dataset: `rt_lnrna_sponging_construct_triage_construct_slot_inputs_v1`
- named slots: `lnrna`, `rt_cds`
- required biological sequence fields:
  - `construct_subject__lnrna_sequence`
  - `construct_subject__rt_cds_sequence`

There is no checked-in 100 bp RT-lnRNA construct contract. Treat any 100 bp view
as a future study contract until a biological/modeling reason and explicit
manifest exist.

### Construct Subject Envelope

RT-CDS DMS rows are construct subject envelopes. The base USR `sequence` column is not
the biological authority for the paired construct input. The construct_subject overlay
must make that explicit:

```yaml
construct_subject__record_kind: construct_subject_envelope
construct_subject__sequence_authority: overlay_only
construct_subject__biological_sequence_fields:
  - construct_subject__lnrna_sequence
  - construct_subject__rt_cds_sequence
```

Naive consumers should read `construct_subject__biological_sequence_fields` before
assuming that `records.parquet.sequence` is the biological design sequence.

### Operator Lane

The study-owned materialization helper is:

```python
dnadesign.studies.units.rt_lnrna_sponging_construct_triage.construct_materialization.materialize_rt_cds_dms_construct_contexts
```

It calls the public `dnadesign.permuter` facade, promotes RT-CDS variants into
construct subject envelopes, and runs Construct against the named-slot contract. Infer
continues from the Construct output dataset using explicit sequence-view names.
The Construct output dataset carries a study-owned construct-subject overlay bridge
from `construct__input_id` back to `construct_subject__id`, so downstream
Infer/label joins do not have to infer construct-subject identity from emitted
sequence ids.

With default RT-CDS DMS settings, Permuter scans the full sense-codon region of
the RT CDS. For the current Eco1 RT CDS, the terminal stop codon is excluded
from the scan, each of the 320 sense residues receives 19 non-stop amino-acid
alternates, E. coli top-codon choices are recorded in metadata, and emitted
variant CDS lengths remain unchanged.

Use the checked-in fixture at
`operations/contract/fixtures/permuter/rt-cds-dms-plan.yaml` as the operator
shape for local tests and batch planning. The adjacent
`operations/contract/fixtures/permuter/rt-cds-dms-infer-handoff.yaml` fixture is
the non-executing Permuter-to-Infer feature request; Infer owns executing that
request and writing sidecars.

### Required View Selection

Infer handoffs from this study must select one or more explicit view names:

- `dual_cassette_2000bp_seq_mean`
- `dual_cassette_2000bp_reverse_complement_seq_mean`
- `lnrna_span_in_construct_anchor_mean`
- `lnrna_span_in_construct_reverse_complement_anchor_mean`
- `rt_cds_span_in_construct_anchor_mean`
- `rt_cds_span_in_construct_reverse_complement_anchor_mean`

Do not select by `product_kind=realized_context` plus orientation alone.
