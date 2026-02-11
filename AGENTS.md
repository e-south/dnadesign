## `dnadesign` for agents

This repository is a **uv-managed Python monorepo** containing multiple tools and pipelines under `src/dnadesign/`.

This file sets **repo-wide working rules**. Many subpackages also include a local `AGENTS.md` with tool-specific commands and data layout.

When you work inside a tool directory (e.g. `src/dnadesign/permuter/`), **also read and follow the nearest local `AGENTS.md`.**

### Scope, safety, and behavior

- Work only inside this repository unless I explicitly add directories.
- Prefer small, reviewable changes; avoid large refactors unless asked.
- Before edits: summarize what you’ll change and why.
- After edits: show a diff summary and list commands you ran.
- Do not run destructive commands without asking (e.g., deleting files, rewriting git history).
- Never exfiltrate secrets. If you see API keys/tokens/credentials, stop and ask what to do.
- No fallbacks: if you’re unsure about a command/entrypoint, check `pyproject.toml` or run `--help` rather than guessing.

### Repo layout

- Root docs: `README.md`, `docs/`
- Python code: `src/dnadesign/`
- CI: `.github/workflows/ci.yaml`
- Tooling: `.pre-commit-config.yaml`, `pyproject.toml`, `uv.lock`

Within `src/dnadesign/`, many tools follow one of these patterns:

1) **Direct package layout**
   - Example: `src/dnadesign/infer/` contains code + tests directly.

2) **Tool directory with its own internal `src/`**
   - Example: `src/dnadesign/permuter/src/` and `src/dnadesign/opal/src/`.
   - Treat the tool directory as a mini-project: configs/jobs/notebooks live next to the tool’s `src/`.

### Generated vs hand-edited content (important)

Treat these as **generated or run artifacts** unless a local `AGENTS.md` explicitly says otherwise:

- `**/outputs/**`
- `**/batch_results/**`
- `**/runs/**` (if present as artifacts)
- `**/.pytest_cache/**`
- `src/dnadesign/**/campaigns/**/outputs/**` (opal)
- Any large model weights / checkpoints / big binaries

Rules:
- **Do not hand-edit generated outputs.** Fix the code/config and re-run to regenerate.
- Ask before committing generated artifacts or large files.
- Avoid making changes under `src/dnadesign/archived/` and `src/dnadesign/prototypes/` unless explicitly requested.

### Environment & tooling (uv)

This project uses `uv` and a committed lockfile:
- `pyproject.toml` declares dependencies and optional groups.
- `uv.lock` pins the full graph.
- `.venv/` is the project environment (managed by uv).

**Preferred workflow**
- Install/sync: `uv sync --locked`
- Run commands: `uv run <cmd>`

#### Setup (baseline)

```bash
uv sync --locked
```

Setup (dev tools):

```bash
uv sync --locked --group dev
```

Setup (notebooks):

```bash
uv sync --locked --group notebooks
```

> GPU/HPC install notes: see docs/hpc/bu_scc_install.md. Don’t attempt CUDA-specific environment work unless asked.

#### Commands

Lint / format / tests:

```bash
uv run ruff check .
uv run ruff format .
uv run pytest -q
```

If pre-commit is installed (dev group):

```bash
uv run pre-commit run --all-files
```

#### Discover and run CLIs

This repo defines multiple console scripts (examples include `usr`, `infer`, `dense`, `cluster`, `opal`, `permuter`, `baserender`).

When unsure about the exact script names:

- Check `pyproject.toml` under `[project.scripts]`, or
- Run `uv run <script> --help`

Common help commands:

```bash
uv run usr --help
uv run infer --help
uv run dense --help
uv run cluster --help
uv run opal --help
uv run permuter --help
uv run baserender --help
```

#### Notebooks (marimo)

- Canonical marimo guidance lives at: `docs/marimo_reference.md`
- Tool-specific notebook conventions may exist in `src/dnadesign/<tool>/notebooks/AGENTS.md`

Preferred notebook workflows:

```bash
# Sandboxed / shareable notebook (inline deps)
uvx marimo edit --sandbox --watch path/to/notebook.py
```

#### Definition of Done (for changes in this repo)

- [ ] Diff reviewed
- [ ] Tests pass (uv run pytest -q) or explain why not applicable
- [ ] Ruff passes (uv run ruff check . and uv run ruff format .)
- [ ] No hand-edits to generated outputs; no large artifacts committed without asking
- [ ] Changes live on a branch (not main) and are ready for a Draft PR if pushing to GitHub
