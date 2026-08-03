---
doc_id: rt-lnrna-subject-bindings
surface: study-workbench-provenance
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-08-01
---

## RT-lnRNA subject bindings

This directory binds component authorities into the finite subjects needed by
the RT-lnRNA study. It is not a global retron or RT registry.

- RT providers remain authoritative for sequences they produce. The tracked
  Eco1 repack publication is one provider, not the owner of all RT variants.
- The RT-lnRNA GenBank catalog owns the RNA-bearing part used by these
  subjects: either a complete lnRNA cassette or a construct-ready MSD handoff,
  as declared by that catalog record.
- The retron hairpin study owns MSD primitives and the source record used to
  materialize secondary structures. Its active structure provenance uses the
  study-owned `retron_msd_structure_panel_v1` identity; Reader and downstream
  objective names do not enter this binding.
- Reader aliases are exact join keys. Measurements and Reader experiment
  selection are outside this registry.
- `construct_projection_status: representable` means the component references
  satisfy this binding contract. It is not a completed Construct materialization;
  that proof remains a separate Construct-owned artifact.

`retron_subject_bindings_v1.yaml` resolves 49 logical subjects: all 46
GenBank-catalog subjects through one declarative projection, plus three
additional RT-repack compositions. GenBank supplies sequence and digest
provenance; this binding set owns composition, study-local variant ids, and
Reader aliases. A study variant id groups observations for analysis; it is not
an RT, lnRNA, or Construct identity. Forty-three subjects have validated hairpin MSD
references. The remaining six retain valid identity without a structure
dependency.

The catalog projection is curated without copying its 44 non-excluded rows:
`projection_sha256` pins the canonical projected identity tuples. Membership,
Reader alias, source path/kind/digest, handoff marker, component digest,
payload, or projection-status drift fails until the binding owner reviews and
updates that digest. Catalog records marked
`local_retron_hairpin_handoff` additionally require an exact hairpin record
whose normalized source-sequence digest matches the catalog lnRNA digest.

Load and validate the checked-in set with
`load_registered_subject_bindings()`, then use `resolve_subject_id()` or
`resolve_alias(namespace=..., value=...)` for exact lookup. Unknown fields,
identities, or aliases; duplicate identities; ambiguous aliases;
owner-contract mismatch; missing sources; digest drift; or an invalid
declared MSD-to-lnRNA span fail before a subject can be projected. Any
provider-published RT resolves through the shared `rt_part_publication_v1`
contract, which validates exact owner identity, an opaque provider reference,
canonical CDS/protein digests, and declared CDS/protein lengths. The current
Eco1 D01/D02 rows declare 963 nucleotides and 320 amino acids; those dimensions
are not hard-coded into this consumer. Each publication also declares whether
the CDS length includes or omits a terminal stop codon. The publication contains no sequence
bytes or provider-internal generated candidate ids.

For a read-only operator query, use one exact selector:

```bash
uv run python -m dnadesign.studies.units.rt_lnrna_sponging_construct_triage.subject_bindings.query \
  --rt-part-id Eco1RT-G3-D01
```

`--subject-id`, `--lnrna-part-id`, `--reader-design-id`, and
`--reader-assay-subject-id` are also supported. Component queries may return
multiple compositions; alias and subject queries remain exact and
case-sensitive. The query validates all provider paths, sequence digests, and
MSD spans before returning references.

`load_registered_subject_bindings()` remains reference-only and validates the
provider publication plus digest-pinned composition.
`load_registered_subject_binding_materialization()` resolves independently
available component bytes and returns typed `blocked_subjects` for opaque
provider parts. Unified Construct materialization fails before writes when any
requested subject is blocked unless the caller explicitly sets
`allow_partial_byte_resolution=True`. An opted-in partial report distinguishes
requested, resolved, and blocked subject counts, declares completeness, and
retains each blocked subject identity, provider reference, digest, and reason.
An exact request containing a blocked subject fails before writes even when
partial resolution is allowed. The all-or-nothing
`load_resolved_registered_subject_bindings()` remains the strict byte-complete
surface. The GenBank catalog remains placement provenance; it is not a parallel
subject selector or a fallback source for private RT bytes.
