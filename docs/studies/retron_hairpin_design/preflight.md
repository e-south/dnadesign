## Retron Hairpin Design Preflight

**Type:** contract
**Plane:** data-plane
**Owner-boundary:** studies
**Entry artifact:** `docs/studies/retron_hairpin_design/` plus declared workspace and validation-command surfaces
**Exit artifact:** one read-only command-level preflight summary for retron_hairpin_design
**Registry-id:** studies.retron-hairpin-design.preflight
**Summary:** Run the retron_hairpin_design preflight suite across declared workspace-layout and command-validation surfaces without mutating outputs.
**Execution-kind:** iterative
**Status-kind:** retron-hairpin-design-preflight

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-21

Read-only command readiness follows the cheaper study snapshot and focuses the
current actionable route, track, or phase declared by the study contract.

### Quick route

- Read-only preflight:
  `uv run ops progress show studies.retron-hairpin-design.preflight --study-dir docs/studies/retron_hairpin_design --scope next --json`
- Snapshot first:
  `uv run ops progress show studies.retron-hairpin-design.status --study-dir docs/studies/retron_hairpin_design --json`

### What this route checks

- study-owned record files such as `routes.md` and `pipeline.yaml`
- declared workspace roots
- declared read-only validation commands such as `released-target-search` and `yiu validate`

### Notes

- `next` scope focuses the current actionable route, track, or phase and its shared study-record checks.
- `full` scope runs the whole declared suite.
- This route does not submit jobs or write Cruncher outputs.
