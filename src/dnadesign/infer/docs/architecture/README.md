## Infer Architecture

### Package Layers

- Interface layer: `cli.py`, `_console.py`, `__main__.py`
- Application layer: `api.py`
- Core runtime: `engine.py`, `config.py`, `errors.py`, `registry.py`, `utils.py`
- Boundary adapters: `ingest/*`, `writers/*`
- Model adapters: `adapters/*`
- Preset catalog: `presets/*`

### Cross-Tool Boundaries

- `usr`: dataset ingest + write-back columns/overlay interactions.
- `notify`: infer events source resolution via shared `_contracts`.
- `ops`: infer runbook workflow contracts and scheduler planning/validation hooks.

For architecture evolution and evidence-backed notes, see the [dev journal](../dev/journal.md).
