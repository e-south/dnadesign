# Exec plan: <short title>

**Status:** active | paused | completed
**Owner:** <team-or-handle>
**Created:** <YYYY-MM-DD>
**Last updated:** <YYYY-MM-DD>

## Purpose / Big Picture
<State why this work matters now, what user/system outcome it unlocks, and what fails if we do nothing.>

## Progress
- [ ] (<YYYY-MM-DD HH:MMZ>) <milestone completed or current step>
- [ ] (<YYYY-MM-DD HH:MMZ>) <next milestone>

## Surprises & Discoveries
- Observation: <what changed your understanding during implementation>
- Evidence: <tests/logs/commands/metrics that support it>

## Decision Log
- Decision: <what you decided>
- Rationale: <why this choice was made>
- Date/Author: <YYYY-MM-DD / handle>

## Outcomes & Retrospective
<What shipped, what did not, any follow-up debt, and what to change next time.>

## Context and Orientation
<Repo paths, contracts, constraints, and assumptions a new contributor/agent needs before touching code.>

## Plan of Work
<High-level phased approach and ordering. Keep this stable; details go in Concrete Steps.>

## Concrete Steps
1. <Atomic step with explicit file paths and expected change>
2. <Atomic step with explicit file paths and expected change>
3. <Atomic step with explicit file paths and expected change>

## Validation and Acceptance
Run and record the exact commands and outcomes:
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run pytest -q`
- `uv run python -m dnadesign.devtools.docs_checks`
- <Manual CLI/API path run and expected output>

## Links
- Proposal: [link](../dev/plans)
- PR: [link](https://github.com/<org>/<repo>/pull/<number>)
- ADR: [link](../architecture/decisions/README.md)
