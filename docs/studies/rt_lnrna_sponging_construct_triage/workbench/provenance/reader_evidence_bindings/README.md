---
doc_id: rt-lnrna-reader-evidence-bindings
surface: study-workbench-provenance
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-07-29
---

## Reader evidence bindings

This is the study-owned join between verified Reader evidence and compositional
RT-lnRNA subjects. It is separate from both subject identity and scientific
interpretation.

The builder reads one digest-verified Reader v6 dataframe record from a
catalog-schema-v4 public payload, groups its
exact `design_id` and `assay_subject_id` values, and resolves only
namespace-qualified aliases declared by the selected subject-binding set. A
known exact alias produces a `bound` row. An unknown identity produces an
explicit `unbound` row. If two exact aliases on the same Reader row point to
different subjects, materialization fails.

Each row retains the Reader experiment, record ID, record schema and contract,
positive exact revision, revision digest, content digest, raw identities,
resolved subject ID, replicate identity basis, and inclusion state. It does
not contain measurements, treatment values, fold
changes, objectives, or claim language.

The executable implementation is
`dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence`.
The artifact contract is
`operations/contract/schemas/reader-evidence-bindings.schema.yaml`.

Reader catalogs that predate catalog schema v4 or record schema v6 are not
handoff-ready and have no compatibility path here. The 2026-07-27 competence
plate is current and its route is ready. The declared output location is a
recipe-owned, Git-ignored workbench target rather than durable provenance.
Build the artifact on demand when a consumer needs a fresh join:

```bash
uv run python -m dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence.materialize \
  --reader-root ../reader \
  --experiment-route-registry ../.agents/skills/retron-assay-study-bridge/references/reader-experiment-routes.json \
  --experiment-id 20260727_retron_Eco1_26_D01_D02_P01_P03_DP01_DP03_benchmark \
  --output /tmp/retron-reader-evidence-bindings.json
```

The CLI accepts only an exact experiment selected by the bridge-owned
`rt_competence_subject_binding` route. It runs that route's live Reader
readiness gate before loading a record or writing the output. It then resolves
only the bridge-authored Reader config path, takes replicate kind and identity
from the public `reader records ... --format json` payload, and verifies
catalog schema v4, the `sample_measurements/df` schema-v6 contract, its
positive exact revision, revision digest, and content digest before reading it.
It confirms the same catalog epoch and record identity through the public CLI
after checking the bytes, and parses exactly those checked bytes. It never
discovers experiments by substring or fuzzy matching. The current plate yields
eight bound subjects and four explicit unbound assay identities; unbound rows
remain visible rather than being guessed into the subject registry.
