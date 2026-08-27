---
doc_id: study-workspace-integration
surface: integration-contract
owner: dnadesign-maintainers
last_verified: 2026-08-26
---

## External study workspaces

Dnadesign provides reusable sequence-design, analysis, and operations APIs. A
live study is a client of those APIs; it is not part of this package.

A study workspace owns its scientific question, aliases, input selection,
objective policy, workflow configuration, evidence index, and decisions. It
should pin the dnadesign revision it uses and pass explicit workspace or
artifact paths. Dnadesign never selects a global active study or derives study
identity from a directory name.

### Public integration seams

- `dnadesign.contracts.sequence.AnnotatedSequencePartV1` carries one
  digest-pinned producer-authored sequence and its nested features into
  `dnadesign.construct.place_annotated_part()` without a producer-specific
  runtime dependency.
- `dnadesign.contracts.reader_records` verifies neutral Reader record
  handoffs without assigning study meaning to the measurements.
- `dnadesign.ops.study` loads and evaluates an explicit `operations/ops.study.yaml`
  contract.
- `dnadesign.ops.status_registries` is the Python entry-point group for
  study-owned status providers. A provider returns paths to packaged
  `status.registry.yaml` files.
- `dnadesign.opal.reader_evidence_artifacts` is the Python entry-point group
  for study-owned OPAL evidence adapters.
- Tool CLIs accept explicit workspace and configuration paths. Study packages
  should call those public surfaces rather than importing tool internals.
- `dnadesign.contracts.storage_objects` verifies the neutral storage envelope
  for private or large workspaces, durable stores, and rebuildable tool caches
  outside public Git checkouts. The producing tool still owns the content
  schema; see [external storage objects](storage-objects.md).

Example status registration in a study package:

```toml
[project.entry-points."dnadesign.ops.status_registries"]
research-studies = "research_studies.integrations.dnadesign_ops:status_registry_paths"
```

The registered function must return one or more existing
`status.registry.yaml` paths inside the provider package. Duplicate paths,
missing files, and provider references outside that package fail at load time.

### Boundary rules

- Cross-study imports are not allowed. Reusable logic moves to a versioned
  public package once more than one study needs it.
- Paths are locations, not identities. A stable `study_id` belongs in the
  study manifest and remains unchanged if the repository moves.
- Raw inputs and large generated outputs stay in controlled storage. The study
  repository records typed provenance, digests, and reviewable small assets.
- Study status and preflight commands may call dnadesign, but dnadesign does
  not own their scientific readiness rules.
- Missing paths, undeclared inputs, stale revisions, and invalid contracts
  fail before execution.

For Infer completion planning, supply either an explicit study workspace or
one or more runbooks:

```bash
uv run ops runbook fill-infer --study-dir /path/to/study
uv run ops runbook fill-infer --runbook /path/to/infer-runbook.yaml
```

Omitting both inputs is an error; there is no implicit active-study fallback.
