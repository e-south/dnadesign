## Stress Ethanol Cipro Growth Status

**Type:** contract
**Plane:** data-plane
**Owner-boundary:** studies
**Entry artifact:** `docs/studies/stress_ethanol_cipro_growth/`
**Exit artifact:** a read-only snapshot of this study's checked-in record, dataset posture, and handoff surfaces
**Registry-id:** studies.stress-ethanol-cipro-growth.status
**Summary:** Read the stress_ethanol_cipro_growth study record and report its current phase, datasets, and owner handoffs.
**Execution-kind:** iterative
**Status-kind:** stress-ethanol-cipro-growth-status

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-17

Use this only for `stress_ethanol_cipro_growth`. The provider lives in the
study package and rejects other `study_id` values. It is not a promoter-family
surface, a generic study ontology, or a compatibility route for another study.

### Direct command

```bash
uv run ops progress show studies.stress-ethanol-cipro-growth.status --json
```

Pinning the study directory is allowed when invoking from outside the repo root:

```bash
uv run ops progress show studies.stress-ethanol-cipro-growth.status \
  --repo-root <repo-root> \
  --study-dir docs/studies/stress_ethanol_cipro_growth \
  --json
```

### Owned Context

The status provider reads these study-owned files:

- `operations/ops.study.yaml`: lifecycle, current phase, artifacts, execution surfaces, and preflight shape
- `record/datasets.yaml`: affiliated dataset ids, row counts, and sync posture
- `operations/pipeline.yaml`: DenseGen, Infer, LatentDNA, Cluster, and OPAL bindings for this study
- `routes/README.md`: one-hop handoffs into owner tools
- `record/status.md`: human-readable current-state note

OPAL candidate-table details are surfaced only as part of this study status,
because their meaning depends on this study's `records.parquet` universe and
selected X column.

### Next Surface

Use [preflight](preflight.md) for command-level readiness and blockers. Use
[routes](../routes/README.md) after the state question is answered and the next task is
to operate a specific owner tool.
