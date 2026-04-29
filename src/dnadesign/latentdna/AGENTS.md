## `latentdna` for agents

Supplement to the repo-root `AGENTS.md` for LatentDNA workspace, source, geometry,
plot, and notebook work.

## Boundaries

- Treat LatentDNA as a generic latent-representation analysis tool. Promoter
  workspaces are dogfood fixtures, not internal tool semantics.
- Keep source adapters contract-driven. Prefer explicit source kinds and schemas
  over study-specific branching.
- Planned or partially materialized sources are valid only when validation can
  report them explicitly as empty/planned; alias rows that point at missing
  vector or scalar payloads must fail fast.
- Keep feature vectors and scalar diagnostics in sidecars. Do not reintroduce
  legacy row-overlay embedding columns.

## Key paths

- Tool README: `src/dnadesign/latentdna/README.md`
- Runtime source: `src/dnadesign/latentdna/src/`
- Source adapters: `src/dnadesign/latentdna/src/sources/`
- Workspace contracts: `src/dnadesign/latentdna/src/contracts/`
- Plot semantics: `src/dnadesign/latentdna/src/workspaces/plot_semantics.py`
- Workspace fixtures and generated local outputs:
  `src/dnadesign/latentdna/workspaces/`
- Tests: `src/dnadesign/latentdna/tests/`

## Generated artifacts

- Treat `workspaces/*/outputs/` as generated. Regenerate through the CLI instead
  of hand-editing JSON, plots, notebooks, or manifests.
- It is acceptable to hard wipe a workspace `outputs/` directory when the task is
  explicitly to prove regeneration or remove stale artifacts.
- Ask before committing large generated artifacts. The workspace status snapshot
  is a small tracked contract artifact when tests require it.

## Commands

```bash
uv run latentdna validate workspace --workspace <workspace_id> --deep --json
uv run latentdna workspace snapshot --workspace <workspace_id> --json
uv run latentdna deliverable list --workspace <workspace_id> --json
uv run latentdna deliverable run <deliverable_id> --workspace <workspace_id> --json
uv run latentdna notebook generate <notebook_id> --workspace <workspace_id> --json
uv run pytest -q src/dnadesign/latentdna/tests
```

Use `MPLCONFIGDIR=/tmp/dnadesign_mpl` for plot-generating commands when the
default Matplotlib cache directory is not writable.

## Layout

- Keep runtime code grouped by domain under `src/` (`sources`, `views`,
  `services`, `contracts`, `workspaces`, `plots`, `notebooks`, and similar
  packages).
- Keep integration-style tests under `tests/integrations/` and contract tests
  under `tests/contracts/`.
- Do not add new root-level runtime modules; update the package source-tree
  contracts when intentionally adding a new top-level documentation file.
