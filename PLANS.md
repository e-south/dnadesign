# PLANS

**Type:** system-of-record
**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-18

## At a glance
This file is the top-level map for change planning and decision capture.
Use it to navigate proposals, execution plans, accepted decisions, and implementation journal entries.

## Contents
- [Planning lifecycle](#planning-lifecycle)
- [Where artifacts live](#where-artifacts-live)
- [How to promote a decision](#how-to-promote-a-decision)

## Planning lifecycle
1. Write or update a proposal for design intent and tradeoffs.
2. Create an execution plan with atomic implementation steps and validation checks.
3. Implement via PR with tests/docs updated together.
4. Capture accepted architectural decisions in ADR form.
5. Record notable implementation outcomes in the development journal.

Execution plans are living records during implementation, not static TODO lists. Keep progress, discoveries, and decisions current as work evolves.

## Where artifacts live
- Design proposals: `docs/dev/plans/`
- Execution plans (active): `docs/exec-plans/active/`
- Execution plans (completed): `docs/exec-plans/completed/`
- ADRs: `docs/architecture/decisions/`
- Maintainer implementation journal: `docs/dev/journal.md`

Execution plans under `docs/exec-plans/` must include required metadata and these sections:
- `Purpose / Big Picture`
- `Progress`
- `Surprises & Discoveries`
- `Decision Log`
- `Outcomes & Retrospective`
- `Context and Orientation`
- `Plan of Work`
- `Concrete Steps`
- `Validation and Acceptance`
- Checklist items are only used in `Progress`; each item includes a UTC timestamp `(YYYY-MM-DD HH:MMZ)`, and the rest of the plan is prose-driven and evidence-linked.

## How to promote a decision
- Keep proposal links in the execution plan and PR description.
- When design intent becomes stable policy, add an ADR under `docs/architecture/decisions/`.
- ADR numbering is required for new decisions going forward.
