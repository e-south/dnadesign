<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[Construct docs](../README.md)

# Construct development


Construct assembles sequences and preserves their annotation coordinates.
Python callers import from `dnadesign.construct`; implementation modules live
under `src/`. Keep sequence choices and window rules in caller configuration.

## Call Construct from Python

After [seeding the demo workspace](../getting-started.md#create-the-demo-workspace),
run this from that workspace with the same environment active:

```python
from dnadesign.construct import preflight_from_config, run_from_config

# Check records, placement coordinates and expected outputs.
preflight = preflight_from_config("config.slot_a.window.yaml")
print([row.output_length for row in preflight.planned_rows])

# Use the execution path without writing sequences or metadata.
preview = run_from_config("config.slot_a.window.yaml", dry_run=True)
print(f"planned={preview.records_total}, written={preview.records_written}")
```

```text
[1000, 1000, 1000, 1000]
planned=4, written=0
```

Remove `dry_run=True` only when ready to persist the outputs. The CLI and
Python entry points use the same planning and write rules.

## Find the owning code

- [Public interfaces](../../src/interfaces/): run, load, preflight and cross-tool helpers.
- [Configuration contracts](../../src/contracts/): validated inputs and errors.
- [Source map](../../src/README.md): assembly, annotation, persistence and CLI modules.
- [Tests](../../tests/): CLI, runtime, workspace and packaging behavior.

From the repository root, run the relevant checks before broadening validation:

```bash
# Check documentation routes and runnable configuration examples.
uv run --no-sync pytest -q src/dnadesign/construct/tests/test_docs_contract.py
# Check Construct behavior after an implementation change.
uv run --no-sync pytest -q src/dnadesign/construct/tests
```
