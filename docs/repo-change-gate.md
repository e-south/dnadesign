## Repo Change Gate

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-17

Use this page when an agent or maintainer needs the smallest validation route
for a local `dnadesign` change. This is a router, not a second procedure tree.

### Tactical Loop

1. Identify the owner surface from the changed paths.
2. Run the smallest targeted pytest path that covers that owner surface.
3. Run the static gates that match the edit:
   - `uv run ruff check .`
   - `uv run ruff format --check .`
   - `uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .`
   - `uv run python -m dnadesign.devtools.docs.checks --repo-root .` for docs or route changes.
4. Use full `uv run pytest -q` for broad behavior changes, merge-depth
   validation, or when targeted coverage cannot be trusted.

### CI-Scope Helper

For monorepo-scale tactical validation, resolve pytest targets from an affected
tool list and a changed-file list:

```bash
uv run python -m dnadesign.devtools.ci.test_targets \
  --repo-root . \
  --affected-tools-csv "<tool1,tool2>" \
  --changed-files-file .ci_changed_files.txt
```

When `studies` is affected, this helper includes the shared `studies/tests`
suite plus the changed study-unit test suites.

### Authority

- Maintainer checks and local parity: [Developer docs](dev/README.md)
- Architecture and ownership rules: [Architecture](../ARCHITECTURE.md)
- Engineering invariants: [Design](../DESIGN.md)
