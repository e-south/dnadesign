# USR interactive notebook

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


Use the marimo notebook for interactive exploration (filters, summaries, and quick dataset inspection).

```bash
# Install project dependencies.
uv sync --locked

# Open notebook.
uv run marimo edit --sandbox --watch src/dnadesign/usr/notebooks/usr_explorer.py
```

## Path-first CLI helpers

These commands accept a dataset id or a file/directory path.

```bash
# Preview current directory parquet (interactive picker when needed).
uv run usr head .

# List columns for selected file.
uv run usr cols

# Print one cell.
uv run usr cell --row 0 --col sequence

# Explicit path examples.
uv run usr head permuter/run42/records.parquet
uv run usr cols ./some/dir --glob 'events*.parquet'
```

When running inside `src/dnadesign/usr/datasets/<namespace>/<dataset>`, commands default to that dataset.

## Next steps

- Sync workflows: [../operations/sync.md](../operations/sync.md)
- Contracts and schema details: [../reference/README.md](../reference/README.md)
