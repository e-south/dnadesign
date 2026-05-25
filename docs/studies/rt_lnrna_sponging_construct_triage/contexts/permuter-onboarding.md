## Permuter Onboarding For RT-CDS DMS

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-24

This study uses Permuter only for generic RT-CDS variant generation. The
RT-lnRNA study owns biological promotion into the 1,600 bp dual-cassette
Construct contract.

### Boundary

- Permuter owns RT-CDS DMS variant intent: codon position, alternate amino acid,
  top-codon policy, request id, variant id, and modification provenance.
- The study owns the candidate envelope: parent candidate, `lnrna` and `rt_cds`
  slot sequences, source basis, candidate role, and construct-promotion status.
- Construct owns the 1,600 bp context realization and six explicit sequence
  views.
- Infer owns Evo2 feature-bundle execution and `_derived/infer` sidecars.

Do not add RT-lnRNA slot, Khan/Crawford overlay, Reader label, OPAL table, or
100 bp context semantics to Permuter.

### Active Contract

The active construct context is the 1,600 bp dual-cassette contract:

- construct dataset: `rt_lnrna_sponging_construct_triage_construct_contexts_1600bp_v1`
- slot input dataset: `rt_lnrna_sponging_construct_triage_construct_slot_inputs_v1`
- named slots: `lnrna`, `rt_cds`
- required biological sequence fields:
  - `candidate__lnrna_sequence`
  - `candidate__rt_cds_sequence`

There is no checked-in 100 bp RT-lnRNA construct contract. Treat any 100 bp view
as a future study contract until a biological/modeling reason and explicit
manifest exist.

### Candidate Envelope

RT-CDS DMS rows are candidate envelopes. The base USR `sequence` column is not
the biological authority for the paired construct input. The candidate overlay
must make that explicit:

```yaml
candidate__record_kind: candidate_envelope
candidate__sequence_authority: overlay_only
candidate__biological_sequence_fields:
  - candidate__lnrna_sequence
  - candidate__rt_cds_sequence
```

Naive consumers should read `candidate__biological_sequence_fields` before
assuming that `records.parquet.sequence` is the biological design sequence.

### Operator Lane

The study-owned materialization helper is:

```python
dnadesign.studies.studies.rt_lnrna_sponging_construct_triage.construct_materialization.materialize_rt_cds_dms_construct_contexts
```

It calls the public `dnadesign.permuter` facade, promotes RT-CDS variants into
candidate envelopes, and runs Construct against the named-slot contract. Infer
continues from the Construct output dataset using explicit sequence-view names.

Use the checked-in fixture at
`operations/contract/fixtures/permuter/rt-cds-dms-plan.yaml` as the operator
shape for local tests and batch planning.

### Required View Selection

Infer handoffs from this study must select one or more explicit view names:

- `dual_cassette_1600bp_seq_mean`
- `dual_cassette_1600bp_fwd_rc_concat`
- `lnrna_span_in_construct_anchor_mean`
- `lnrna_span_in_construct_reverse_complement_anchor_mean`
- `rt_cds_span_in_construct_anchor_mean`
- `rt_cds_span_in_construct_reverse_complement_anchor_mean`

Do not select by `product_kind=realized_context` plus orientation alone.
