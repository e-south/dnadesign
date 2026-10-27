## `notify` for agents

Supplement to repo-root `AGENTS.md` with `notify`-local boundaries only.
Reusable setup/watch/recover procedures live in the repo-local skill and
operator runbooks listed below.

### Key paths
- Code: `src/dnadesign/notify/`
- Tests: `src/dnadesign/notify/tests/`
- Operator overview: `docs/notify/README.md`
- Canonical operators runbook: `docs/notify/usr-events.md`
- Command contracts: `src/dnadesign/notify/docs/reference/command-contracts.md`
- Repo-local notify operator skill: `.agents/skills/notify-ops/SKILL.md`
- Module-local index: `src/dnadesign/notify/README.md`

### Local boundary
- Keep this file focused on `notify` code/test/doc boundaries, not step-by-step
  operator command inventory.
- Route setup/watch/recover procedures through `docs/notify/README.md`,
  `docs/notify/usr-events.md`, and `.agents/skills/notify-ops/SKILL.md`.
- Workspace shorthand is repo-rooted; outside the repo checkout, use explicit
  `--config` or the repo-root guidance in the runbooks.

### Contracts
- Notify input contract is Universal Sequence Record `.events.log` JSONL only.
- DenseGen runtime diagnostics (`outputs/meta/events.jsonl`) are out of scope.
- `profile_version` must be `2`.

### Tests
- `uv run pytest -q src/dnadesign/notify/tests/docs/test_progressive_disclosure_contracts.py`
