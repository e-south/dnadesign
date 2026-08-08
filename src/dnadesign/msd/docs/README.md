# MSD compiler

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-08

The compiler takes two inputs:

- a `retron_msd_compiler_spec_v1` request containing design parts and concrete
  sequence inputs;
- an explicit `retron_msd_design_registry_v1` file supplied by the caller.

The API does not infer a study directory or workspace layout.

```python
from dnadesign.msd import compile_msd_design_unit, load_msd_compiler_spec

resolved = load_msd_compiler_spec(
    "request.yaml",
    registry_path="msd_design_registry.yaml",
)
unit = compile_msd_design_unit(
    resolved.catalog.records[0],
    payload_sequences=resolved.payload_sequences,
    payload_complement_sequences=resolved.payload_complement_sequences,
    cap_sequences=resolved.cap_sequences,
)
```

Resolution and compilation fail before publication when required identifiers,
sequences, topology bounds, or registry entries are missing or inconsistent.
