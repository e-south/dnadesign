## `usr` notebooks for agents

- Canonical marimo rules: `docs/notebooks/marimo-reference.md`.

### Setup
The default `tools` group installs the `full` extra, including Marimo.

```bash
uv sync --locked
```

### Edit
```bash
uv run marimo edit --sandbox --watch src/dnadesign/usr/notebooks/<notebook>.py
```

### Lint
```bash
uv run marimo check --strict src/dnadesign/usr/notebooks/*.py
```
