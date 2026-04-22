## <Cruncher Study Title>

**Owner:** dnadesign-maintainers
**Last verified:** YYYY-MM-DD

### At a glance

- <one-sentence study question>
- primary lane: <released-product snapback | other cruncher lane>
- supporting lane: <yiu boundary check | n/a>
- <what must stay out of scope>

### Quick route

- Snapshot:
  `uv run ops progress show cruncher.data-plane.cruncher-study-status --study-dir docs/studies/<study-id> --json`
- Preflight:
  `uv run ops progress show cruncher.data-plane.cruncher-study-preflight --study-dir docs/studies/<study-id> --scope next --json`
- Repo-local study skill: `.agents/skills/<study-skill>/SKILL.md` or `n/a`

### What is settled

- <what this study now treats as the active lane>
- <what stays contrast-only>
- <what stays framing context rather than scoring logic>

### Current phase and surfaces

- Current phase: `<phase-id>`
- Primary lane: `<lane>`
- Supporting lane: `<lane or n/a>`
- Next owner surface: `docs/studies/<study-id>/routes.md`

### Current execution surfaces

- Workspace: `<repo path>`
- Spec: `<repo path>`
- Route doc: `docs/studies/<study-id>/routes.md`

### Decision boundaries

- <constraint 1>
- <constraint 2>
- <constraint 3>

### Evidence ladder

- `<repo path>`
- `<repo path>`

### Next actions

1. <next read-only probe>
2. <next mutating command>
3. <next contrast check>
