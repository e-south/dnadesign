## Running marimo notebooks

This document is a practical quickstart for running repository notebooks with marimo. Read it when you want to launch notebooks quickly; deeper notebook patterns and UI guidance are in `docs/marimo-reference.md`.

### Contents
This section lists the notebook execution modes used in this repository.

- [Campaign-tied notebooks (OPAL)](#campaign-tied-notebooks-opal)
- [Project-environment notebooks](#project-environment-notebooks)
- [Sandbox notebooks with inline dependencies](#sandbox-notebooks-with-inline-dependencies)

### Campaign-tied notebooks (OPAL)
This section covers notebook flows generated and managed inside OPAL campaigns.

```bash
# List notebook status and next actions for the active campaign.
uv run opal notebook

# Generate a notebook scaffold in the campaign notebooks directory.
uv run opal notebook generate

# Run the campaign notebook.
uv run opal notebook run
```

### Project-environment notebooks
This section runs marimo directly inside the repository uv environment.

```bash
# Sync project environment.
uv sync --locked

# Open a notebook in edit mode using project dependencies.
uv run marimo edit notebooks/foo.py
```

### Sandbox notebooks with inline dependencies
This section uses marimo sandbox mode for self-contained notebook dependencies.

```bash
# Create or edit a sandbox notebook.
uvx marimo edit --sandbox notebooks/sandbox_example.py

# Execute a sandbox notebook as a script.
uv run notebooks/sandbox_example.py

# Add local dnadesign checkout as an editable inline dependency.
uv add --script notebooks/sandbox_example.py . --editable

# Add and remove notebook-local dependencies.
uv add --script notebooks/sandbox_example.py numpy
uv remove --script notebooks/sandbox_example.py numpy
```

For deeper notebook authoring and UI behavior guidance, read **[docs/marimo-reference.md](marimo-reference.md)**.
