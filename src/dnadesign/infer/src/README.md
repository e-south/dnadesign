## infer source tree

`infer/src/` contains internal runtime implementation modules.

### Top-level internal modules

- `cli/`: Typer command surface package (`app.py`, `commands/*`, command helpers, console rendering).
- `api.py`: stable Python execution entrypoints used by package exports.
- `engine.py`: extract and generate orchestration.
- `_logging.py`: shared logging policy used by CLI and runtime modules.
- `bootstrap.py`: explicit adapter/function registry initialization contract.
- `workspace.py`: workspace root/template resolution and scaffold contracts.
- `config.py`, `errors.py`, `contracts.py`: schema and runtime contract boundaries.
- `registry.py`: model/function registration and resolution.
- `runtime/`: execution policies, ingest loading, adapter dispatch/runtime, progress, and write-back/resume flows.

### Internal subpackages

- `adapters/`: model backend implementations and adapter registration.
- `ingest/`: input loading and validation boundaries.
- `writers/`: write-back boundaries.
- `presets/`: packaged workflow presets.

See [`../docs/architecture/README.md`](../docs/architecture/README.md) for architecture context and [`../tests/`](../tests) for behavior contracts.
