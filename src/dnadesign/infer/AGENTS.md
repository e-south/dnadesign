## `infer` for agents

Supplement to the repo-root `AGENTS.md` for Infer extraction, generation,
feature sidecars, and USR writeback work.

### Boundaries

- Treat Infer as a generic model-runtime and feature-extraction tool. Promoter
  studies, anchors, core windows, and context products are request inputs, not
  internal package semantics.
- Keep sequence-view feature outputs in canonical sidecars under
  `_derived/infer/`: `feature_aliases.parquet`, `feature_vectors.parquet`,
  `feature_scalar_aliases.parquet`, and `feature_scalars.parquet`.
- Partial completion is expected for long GPU jobs. Resume logic must reuse
  complete vector and scalar sidecar rows, identify missing aliases, and fail
  fast on corrupt alias-to-payload references.
- Backcompat shims should be explicit, tested, and removable. Do not add new
  row-overlay embedding columns for modern sequence-view features.

### Key paths
- README: `src/dnadesign/infer/README.md`
- Runtime source: `src/dnadesign/infer/src/`
  - Adapters: `src/dnadesign/infer/src/adapters/`
  - Features and sidecars: `src/dnadesign/infer/src/features/`
  - Runtime and resume planning: `src/dnadesign/infer/src/runtime/`
  - Ingest: `src/dnadesign/infer/src/ingest/`
  - Writers: `src/dnadesign/infer/src/writers/`
  - Presets: `src/dnadesign/infer/src/presets/`
- Workspaces: `src/dnadesign/infer/workspaces/`
- Tests: `src/dnadesign/infer/tests/`

### Performance & safety notes
- Model weights/downloads can be large; do not download unless asked.
- Prefer small unit tests; avoid GPU end-to-end runs unless explicitly requested.
- GPU execution should be validated through dry-run, capacity, and completion
  planners before submitting long jobs.

### Outputs / naming
When writing back to USR, output columns follow:
- `infer__<model_id>**<job_id>**<out_id>`

For sequence-view feature bundles, prefer the sidecar contract above. Consumers
should join aliases to vectors/scalars and USR sequence views rather than
depending on study-specific columns.

### Commands (copy/paste)
```bash
uv run infer --help
uv run infer adapters list
uv run infer presets list
uv run infer presets show evo2/extract_logits_ll

# YAML-driven
uv run infer run --config src/dnadesign/infer/config.yaml --help

# Ad-hoc single output
uv run infer extract --help

# Generation
uv run infer generate --help

# Validation helpers
uv run infer validate --help
```

### Tests

```bash
uv run pytest -q src/dnadesign/infer/tests
uv run pytest -q src/dnadesign/infer/tests/runtime/test_sequence_view_completion_planner.py src/dnadesign/infer/tests/runtime/test_resume_planner.py
```
