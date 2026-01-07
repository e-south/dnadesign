## Running marimo notebooks

More in depth marimo guidance lives at `docs/marimo_reference.md`.

There are two ways to use marimo in `dnadesign`:

### 1) Install marimo into the project

```bash
uv sync --locked --group notebooks
uv run marimo edit notebooks/foo.py
```

This runs marimo inside your project environment, so it can import `dnadesign` and anything in `uv.lock`.

### 2) Sandboxed / self-contained marimo notebooks (inline dependencies)

Marimo can manage per-notebook sandbox environments using inline metadata. This is great for shareable notebooks.

1) Create/edit a sandbox notebook (marimo installed temporarily via uvx).

    ```bash
    uvx marimo edit --sandbox notebooks/sandbox_example.py
    ```

2) Run a sandbox notebook as a script.

    ```bash
    uv run notebooks/sandbox_example.py
    ```

3) Make the sandbox notebook use your local `dnadesign` repo in editable mode.

    From the repo root:

    ```bash
    uv add --script notebooks/sandbox_example.py . --editable
    ```

This writes inline metadata into the notebook so its sandbox can install dnadesign from your local checkout in editable mode.

4) Add/remove sandbox dependencies (only affects the notebook file).

    ```bash
    uv add    --script notebooks/sandbox_example.py numpy
    uv remove --script notebooks/sandbox_example.py numpy
    ```

    Note: If you run an agent to edit a marimo notebook, launch with `--watch` so changes appear live:

    ```bash
    uv run marimo edit --watch notebook.py
    ```

---

@e-south
