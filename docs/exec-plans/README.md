## Execution Plans

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-18

### At a glance
Execution plans turn approved intent into checklisted implementation steps.
Keep plans atomic, test-backed, and linked to proposals/PRs.
Plans are living documents: update progress, discoveries, and decisions while work is in-flight.

### Contents
- [Active plans](active/README.md)
- [Completed plans](completed/README.md)
- [Plan template](../templates/exec-plan.md)
- [Planning map](../../PLANS.md)
- [Design proposals](../dev/plans)

### Required metadata
- Every non-README plan under `active/` or `completed/` must include:
  - `**Status:** active | paused | completed`
  - `**Owner:** <team-or-handle>`
  - `**Created:** <YYYY-MM-DD>`
- Every plan must include at least one markdown link for traceability to a proposal, PR, or ADR.

### Required sections
- `## Purpose / Big Picture`
- `## Progress`
- `## Surprises & Discoveries`
- `## Decision Log`
- `## Outcomes & Retrospective`
- `## Context and Orientation`
- `## Plan of Work`
- `## Concrete Steps`
- `## Validation and Acceptance`

Use `../templates/exec-plan.md` as the authoritative scaffold.
Checklist items are reserved for `## Progress` so status is machine-checkable and easy to scan.
Every `Progress` checklist item must include a UTC timestamp in `(YYYY-MM-DD HH:MMZ)` format.
