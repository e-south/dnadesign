## Maintaining dependencies

Prefer `uv add` / `uv remove` for dependency changes:

- Add a runtime dependency:

    ```bash
    uv add <package>
    ```

- Add to a dependency group:

    ```bash
    uv add --group dev <package>
    uv add --group notebooks marimo
    ```

- Remove a dependency:

    ```bash
    uv remove <package>
    ```

Then commit `pyproject.toml` + `uv.lock`.

If you edit `pyproject.toml` by hand, regenerate the lockfile:

```bash
uv lock
```

New users should then run:

```bash
uv sync --locked
```

---

@e-south
