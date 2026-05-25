# Permuter Architecture

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-24

Permuter is the variant-intent surface for dnadesign. It generates explicit
sequence variants, records what changed, and materializes USR-shaped candidate
rows when a caller asks for a dataset. It does not own model-feature sidecars,
Construct placement semantics, study biology, or Ops execution lanes.

## Ownership Boundary

Use Permuter when the primary question is "what variants should exist?"

| Plane | Owner | Contract |
| --- | --- | --- |
| Variant intent | Permuter | request id, reference, protocol, modifications, `permuter__var_id` |
| Canonical data plane | USR | sequence id, overlays, events, dataset roots, sidecar locations |
| Physical context | Construct | named slots, template realization, spans, sequence views |
| Model-derived evidence | Infer | feature bundles, aliases, `_derived/infer`, completion, resume, stale detection |
| Biological meaning | Study | candidate roles, source overlays, assay labels, promotion gates |
| Execution lane | Ops | declared commands, status, preflight, runbook discovery |

Permuter can prepare handoff artifacts for those owners, but it must not copy
their internal logic. Cross-tool callers import `dnadesign.permuter`, not
`dnadesign.permuter.src.*`.

## Stable Concepts

- A request describes a mutation policy: reference name, reference sequence,
  protocol, scope, and bounded selector such as codon or amino-acid positions.
- A variant record describes one emitted variant: variant identity, sequence,
  modifications, and Permuter-owned provenance.
- A materialized dataset is a USR-shaped table with canonical sequence identity
  plus Permuter namespaced columns.
- A handoff describes what the next owner should do; it is not execution by
  Permuter.

The public API currently exposes request/result dataclasses rather than a
separate neutral handoff schema. Do not promote a schema into
`src/dnadesign/contracts/` until both producer and consumer use it through a
stable public import.

## Identity Contract

Materialized `records.parquet` rows use two distinct identifiers:

- `id`: canonical USR sequence id derived from `(bio_type, sequence)`.
- `permuter__var_id`: Permuter variant/provenance identity from the public
  `VariantRecord.id`.

Do not add a parallel `permuter__variant_id` column. If the schema needs a new
variant identity column, version the Permuter materialization contract instead
of carrying both names.

## Infer Boundary

Permuter is allowed to reference an Infer feature request, but Infer owns
execution. In particular, Permuter must not write or mutate:

- `_derived/infer/feature_aliases.parquet`
- `_derived/infer/feature_vectors.parquet`
- `_derived/infer/feature_scalar_aliases.parquet`
- `_derived/infer/feature_scalars.parquet`
- Infer completion ledgers, runtime fingerprints, or stale/reusable inventory

The modern path is a non-executing handoff to Infer's public feature-bundle
surface. The older ad hoc Evo2 evaluator path is a Permuter scoring convenience,
not proof that a dataset has modern Infer sidecar coverage.

## Study Boundary

Study-specific terms stay out of Permuter. For RT-lnRNA, that means Permuter
does not know the 1,600 bp dual-cassette construct, `lnrna` and `rt_cds` slots,
Khan/Crawford overlays, Reader labels, OPAL candidate tables, or any future
100 bp view contract. The RT study owns those semantics and calls Permuter only
for generic variant generation.

## Fail-Fast Rules

- Reject mismatched record reference or bio-type during materialization.
- Preserve `VariantRecord.id` as `permuter__var_id`; do not silently reinterpret
  it as a USR sequence id.
- Reject conflicting `metadata.permuter.var_id` if present.
- Reject `metadata.permuter.variant_id`; it would materialize the forbidden
  duplicate `permuter__variant_id` spelling.
- Reject legacy `permuter__metric__*` columns in strict validation.
- Treat missing Construct sequence views as upstream Construct/USR completion,
  not a Permuter fallback.
