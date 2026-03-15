---
name: bu-scc-usr-sync
description: Operate dnadesign USR dataset sync against BU SCC using the repo's canonical USR roots, remotes doctor preflight, pullable-dataset inventory checks, strict sync audits, and no-delete safety. Use when the user wants to diff, pull, push, or bootstrap USR datasets between the local dnadesign clone and BU SCC. Do not use for generic scheduler submission tasks with no USR dataset transfer scope.
metadata:
  version: 0.1.0
  category: workflow-automation
  tags: [usr, bu-scc, sync, datasets, dnadesign]
---

# BU SCC USR Sync

## Purpose

Run `usr diff` / `usr pull` / `usr push` against BU SCC with the dnadesign-specific storage contract, preflight checks, and portability rules that were hardened in this repo.

## Contract

- Canonical local USR root in this repo: `src/dnadesign/usr/datasets`
- Canonical SCC base dir for this repo: `/project/<user>/dnadesign/src/dnadesign/usr/datasets`
- Use `uv run usr remotes doctor --remote <name>` before transfer
- Treat as pull-only unless the user explicitly asks to push
- Never delete datasets from SCC as part of sync/bootstrap
- Only treat directories with `records.parquet` as pullable datasets
- Preserve dataset contents and sidecars; do not rely on owner/group/permission metadata parity across hosts
- If SCC auth fails under `BatchMode=yes`, use `batch_mode: false`
- Explicit missing dataset ids are valid bootstrap pull targets when strict bootstrap mode is off
- Use the dataset id that is actually canonical in the repo, whether flat (`mg1655_promoters`) or namespace-qualified

## Execution Loop

1. Verify locus and config
- Confirm you are operating on the intended local clone and intended SCC remote.
- Confirm `USR_REMOTES_PATH` is set or pass the configured remotes file.

2. Preflight
- Run `uv run usr remotes doctor --remote <name>`.
- List local datasets with `uv run usr --root src/dnadesign/usr/datasets ls --format json`.
- Inventory remote pullable datasets by checking for `records.parquet`, not by directory names alone.

3. Decide action
- `diff` before every transfer.
- `pull` for SCC -> local bootstrap or refresh.
- `push` only when the user explicitly wants local changes propagated back.

4. Verify
- Use `--audit-json-out <path>` for machine-readable evidence.
- Run `uv run usr validate <dataset> --strict` after bootstrap or major refresh.
- Re-run `diff` to confirm no-op after a completed transfer when needed.

## Repo-Specific Notes

- Curated construct demo inputs now live in flat semantic datasets (`mg1655_promoters`, `plasmids`), not in a tool-owned dataset namespace.
- Realized construct demo outputs should also use flat semantic dataset ids such as `pdual10_slot_a_window_1kb_demo`, not a tool-owned dataset namespace.
- Human-readable record names should be carried in `usr_label__primary` / `usr_label__aliases`; tool-specific provenance belongs in tool namespaces such as `construct_seed__*`.
- Empty SCC directories are not datasets until `records.parquet` exists.

## Output

Return:
- local root
- remote name and base dir
- pullable remote dataset inventory
- datasets transferred or confirmed already present
- sync audit result
- strict validation result
- any unresolved auth or path issues
