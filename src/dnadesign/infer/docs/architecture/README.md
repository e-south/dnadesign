## Infer Architecture

### Package Layers

- Interface layer: `cli.py`, `_console.py`, `__main__.py`
- Application layer: `api.py`
- Core runtime:
  - orchestration: `engine.py`
  - dispatch contracts: `adapter_dispatch.py`
  - extract execution loop: `extract_execution.py`
  - generate execution loop: `generate_execution.py`
  - progress lifecycle: `progress.py`
  - config and errors: `config.py`, `errors.py`
  - registry and utils: `registry.py`, `utils.py`
- Boundary adapters: `ingest/*`, `writers/*`
- Model adapters: `adapters/*`
- Preset catalog: `presets/*`

### Cross-Tool Boundaries

- `usr`: dataset ingest + write-back columns/overlay interactions.
- `notify`: infer events source resolution via shared `_contracts`.
- `ops`: infer runbook workflow contracts and scheduler planning/validation hooks.

For architecture evolution and evidence-backed notes, see the [dev journal](../dev/journal.md).
