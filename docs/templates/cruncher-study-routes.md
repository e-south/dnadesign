## <study-id> Routes

**Owner:** dnadesign-maintainers
**Last verified:** YYYY-MM-DD

Use this page after the tracked study status answers `where are we?`.
This page keeps the study-owned handoff map in one place.

### Quick route

- Snapshot:
  `uv run ops progress show cruncher.data-plane.cruncher-study-status --study-dir docs/studies/<study-id> --json`
- Preflight:
  `uv run ops progress show cruncher.data-plane.cruncher-study-preflight --study-dir docs/studies/<study-id> --scope next --json`
- Repo-local study skill: `.agents/skills/<study-skill>/SKILL.md` or `n/a`
- Pair with: `harness-engineering`, `pragmatic-programming-principles`

### Boundary shorthand

- <guardrail 1>
- <guardrail 2>

### Primary route

- Current state: `<in_progress | planned>`
- Workspace: `<repo path>`
- Primary docs:
  `<repo path>`
- First read-only command:
  `<command>`
- Follow-up mutating commands:
  `<command>`
  `<command>`
- Route note:
  <why this route exists>

### Contrast route

- Current state: `<planned | n/a>`
- Workspace: `<repo path>`
- First read-only command:
  `<command>`
- Follow-up mutating commands:
  `<command>`
- Route note:
  <why this route stays contrast-only>

### Context surfaces

- Study note: `docs/studies/<study-id>/status.md`
- Study command ladder: `docs/studies/<study-id>/pipeline.yaml`
- `<repo path>`
