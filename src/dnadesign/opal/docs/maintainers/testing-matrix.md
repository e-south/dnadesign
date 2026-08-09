## OPAL Workflow Pressure-Test Matrix

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-09


The matrix below exercises maintained workflows in isolated campaign copies.
Operational usage starts at:

- [Workflows](../index.md#workflows)

### What this does

Runs each demo campaign end-to-end in an isolated temp copy:

- `validate -> init -> ingest-y -> run -> verify-outputs`
- `ctx audit -> explain -> record-show -> predict -> plot`

### Run the matrix

```bash
uv run opal demo-matrix --rounds 0 --fail-fast
```

Use `--rounds 0,1` to verify resume behavior. Add `--keep` only when you need
to inspect the temporary workspaces after the run.
