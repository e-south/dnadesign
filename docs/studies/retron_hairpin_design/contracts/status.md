## Retron Hairpin Design Status

**Type:** contract
**Plane:** data-plane
**Owner-boundary:** studies
**Entry artifact:** `docs/studies/retron_hairpin_design/`
**Exit artifact:** a read-only snapshot of the declared route or track, command groups, and operator context
**Registry-id:** studies.retron-hairpin-design.status
**Summary:** Read the retron_hairpin_design study record and summarize its declared route or track, command groups, and native-agent bootstrap context.
**Execution-kind:** iterative
**Status-kind:** retron-hairpin-design-status

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-21

Use this only for `retron_hairpin_design`. The provider lives in the study
package and rejects other `study_id` values. Cruncher is an owner tool this
study routes to; it is not the owner of the study-status implementation.

### Quick route

- Current study snapshot:
  `uv run ops progress show studies.retron-hairpin-design.status --study-dir docs/studies/retron_hairpin_design --json`
- Command blockers for the same study:
  `uv run ops progress show studies.retron-hairpin-design.preflight --study-dir docs/studies/retron_hairpin_design --scope next --json`

### What this route reads

- `docs/studies/retron_hairpin_design/record/status.md`
- `docs/studies/retron_hairpin_design/routes/README.md`
- `docs/studies/retron_hairpin_design/operations/pipeline.yaml`
- `docs/studies/retron_hairpin_design/operations/ops.study.yaml`

### When to use it

- You are starting a new thread and need the narrowing context first.
- You want the maintained command groups instead of reconstructing them from workspaces.
- You need the explicit "do not collapse these abstractions together" list before running Cruncher.

### Notes

- This route is record-plane only. It does not mutate workspaces or run Cruncher commands.
