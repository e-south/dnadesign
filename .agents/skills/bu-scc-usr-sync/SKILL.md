---
name: bu-scc-usr-sync
description: Operate dnadesign USR dataset sync against BU SCC using the repo's canonical USR roots, explicit remotes config, doctor and warm-auth preflight, pullable-dataset inventory checks, and no-delete safety. Use when the user wants to diff, pull, push, or bootstrap USR datasets between the local dnadesign clone and BU SCC. Do not use for generic scheduler submission tasks, non-USR file transfer plans, or repo changes with no sync workflow scope.
metadata:
  version: 0.2.0
  category: workflow-automation
  tags: [usr, bu-scc, sync, datasets, dnadesign]
---

# BU SCC USR Sync

## Purpose

Run `usr diff` / `usr pull` / `usr push` against BU SCC with the dnadesign
storage contract, explicit remotes-config posture, and strict no-delete
guardrails without depending on package-local skill roots.

## Scope

In scope:
- BU SCC dataset bootstrap, refresh, and explicit push workflows
- `usr remotes doctor`, `usr remotes status`, and `usr remotes warm-auth`
- explicit `--remotes-config <remotes.yaml>` guidance with `USR_REMOTES_PATH`
  as shell-session fallback
- machine-readable sync evidence via `--audit-json-out`
- strict validation after bootstrap or major refresh

Out of scope:
- generic scheduler submission work; use `.agents/skills/sge-hpc-ops/SKILL.md`
- deleting remote datasets as part of sync
- non-USR file-transfer workflows with no dataset contract
- inventing remote dataset ids that are not canonical in the repo

## Success Criteria

- local root and remote base dir are explicit
- `usr remotes doctor --remote <name>` is part of every real transfer loop
- pullable remote datasets are identified by `records.parquet`, not directory
  names alone
- pull is the default unless the user explicitly asks to push
- sync results produce an audit artifact or an explicit failure reason
- strict validation follows bootstrap or major refresh

## Workflow

1. Load the canonical sync surfaces.
- Start at `src/dnadesign/usr/docs/operations/sync.md`.
- Use `src/dnadesign/usr/docs/operations/sync-setup.md` for SSH keys, remote
  profiles, and auth posture.
- Use `docs/bu-scc/README.md` and `docs/bu-scc/quickstart.md` for SCC-specific
  storage and environment context.
- Use [sync-loop.md](references/sync-loop.md) for the short operator ladder.

2. Verify locus and remote config.
- Prefer `uv run usr --remotes-config <remotes.yaml> ...` for each sync
  command.
- Use `USR_REMOTES_PATH` only as a shell-session fallback when many commands
  will reuse the same remotes file.
- Run `uv run usr remotes doctor --remote <name>`.
- Run `uv run usr remotes status --remote <name>` to see whether a reusable SSH
  control socket is already live.
- If BatchMode auth still needs Duo or keyboard-interactive follow-up, run
  `uv run usr remotes warm-auth --remote <name>` in a real terminal before
  transfer.

3. Decide action and verify after transfer.
- `diff` before every transfer.
- `pull` for SCC -> local bootstrap or refresh.
- `push` only when the user explicitly wants local changes propagated back.
- Use `--audit-json-out <path>` for machine-readable sync evidence.
- Run `uv run usr validate <dataset> --strict` after bootstrap or major
  refresh.
- Re-run `diff` when needed to confirm no-op after transfer.

## Required Deliverables

- local root
- remote name and base dir
- auth and remotes preflight result
- pullable remote dataset inventory
- datasets transferred or confirmed already present
- sync audit result
- strict validation result
- unresolved auth, path, or policy issues

## Output

Return:
- local root and remotes-config posture
- remote name and base dir
- chosen action: diff, pull, or push
- transfer and audit outcome
- strict validation outcome
- next sync or auth step when blocked

## Trigger Tests

Should trigger:
- "Diff this USR dataset against BU SCC."
- "Bootstrap the missing dataset from SCC into this dnadesign clone."
- "Push my local USR changes back to BU SCC."
- "Why is BU SCC sync failing under Duo or keyboard-interactive auth?"
- "Which remotes-config command should I use for this repo?"

Should not trigger:
- "Submit a BU SCC batch job."
- "Move arbitrary files to SCC."
- "Refactor the USR sync implementation."
- "Debug a scheduler queue state."

## References

- [sync-loop.md](references/sync-loop.md)
- [external-sources.md](references/external-sources.md)
