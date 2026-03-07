## Infer Architecture

### Package shape

- Top-level package surface:
  - `README.md`, `docs/`, `src/`, `tests/`
  - thin entrypoints only: `__init__.py`, `__main__.py`, `cli.py`
- Internal implementation tree: `src/`
  - CLI interface package: `src/cli/app.py`, `src/cli/console.py`, `src/cli/common.py`, `src/cli/builders.py`, `src/cli/ingest.py`, `src/cli/requests.py`
  - shared logging policy: `src/_logging.py` (consumed by CLI and runtime modules)
  - explicit registry bootstrap boundary: `src/bootstrap.py`
  - CLI command groups: `src/cli/commands/run.py`, `src/cli/commands/extract.py`, `src/cli/commands/generate.py`, `src/cli/commands/presets.py`, `src/cli/commands/adapters.py`, `src/cli/commands/validate.py`, `src/cli/commands/workspace.py`
  - application API: `src/api.py`
  - runtime orchestration: `src/engine.py`
  - runtime package: `src/runtime/adapter_runtime.py`, `src/runtime/adapter_dispatch.py`, `src/runtime/batch_policy.py`, `src/runtime/ingest_loading.py`, `src/runtime/extract_execution.py`, `src/runtime/generate_execution.py`, `src/runtime/progress.py`, `src/runtime/extract_chunk_writeback.py`, `src/runtime/writeback_dispatch.py`, `src/runtime/resume_planner.py`
  - contracts and schema: `src/config.py`, `src/contracts.py`, `src/errors.py`, `src/registry.py`, `src/utils.py`, `src/workspace.py`
  - boundary subpackages: `src/ingest/*`, `src/writers/*`
  - extensions: `src/adapters/*`, `src/presets/*`

### Runtime execution contracts

- Adapter extract execution is padding-free and deterministic by token-length buckets:
  - inputs are grouped by token length before Evo2 forward calls for logits and embedding extraction.
  - batch assembly preserves input ordering in emitted outputs.
- Pooling contracts remain explicit and fail fast:
  - `pool.dim >= 1` is required for logits/embedding extract outputs.
  - pooling over the batch axis is rejected.

### Cross-Tool Boundaries

- `usr`: dataset ingest + write-back columns/overlay interactions.
- `notify`: infer events source resolution via shared `_contracts`.
- `ops`: infer runbook workflow contracts and scheduler planning/validation hooks through public infer API (`dnadesign.infer.validate_runbook_gpu_resources`), not `dnadesign.infer.src.*` internals.

For architecture evolution and evidence-backed notes, see the [dev journal](../dev/journal.md).
