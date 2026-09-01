## `usr` notebooks for agents

- Canonical marimo rules: `docs/notebooks/marimo-reference.md`.

### Setup
```bash
uv sync --locked --group notebooks
```

### Edit
```bash
uv run marimo edit --sandbox --watch src/dnadesign/usr/notebooks/<notebook>.py
```

### Lint
```bash
uv run marimo check src/dnadesign/usr/notebooks/*.py
```
