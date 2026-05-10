# Refresh Loop

Start with the checked-in retron hairpin study record, then refresh only the
smallest surface that answers the question.

## Blank-thread bootstrap

1. Read `docs/studies/README.md` and `docs/studies/index.yaml`.
2. Read `docs/studies/retron_hairpin_design/status.md`.
3. Run
   `uv run ops progress show cruncher.data-plane.cruncher-study-status --study-dir docs/studies/retron_hairpin_design --json`.
4. Escalate to
   `uv run ops progress show cruncher.data-plane.cruncher-study-preflight --study-dir docs/studies/retron_hairpin_design --scope next --json`
   only for blockers or next-run readiness.
5. Open `routes.md` after the record or blocker answer is settled.
6. Open `scar-nick-base-junction.md` only when the question is about stem-base
   scar profiles, top/bottom scar-nick flexibility, or B26/B43 analogs.
7. Open `pipeline.yaml` only when machine-readable command-group or native-agent
   bootstrap context is still needed.

## Minimum evidence by question

| Question | Primary surface | Minimum evidence | Fail visibly when |
| --- | --- | --- | --- |
| Where is the hairpin effort now? | `cruncher.data-plane.cruncher-study-status` | study id, current phase, primary lane, command groups, next route | required record files or the pinned study directory are missing |
| What blocks the next step here? | `cruncher.data-plane.cruncher-study-preflight --scope next` | `scope`, `phase_id`, `check_group`, `kind`, `surface_id`, `artifact_id` | `ops.study.yaml` or declared execution surfaces are missing |
| Which command group should I run next? | `routes.md` | route purpose, workspace, first read-only command, mutating follow-up | the study omits the route map |
| What scar-nick base-junction space is feasible? | `scar-nick-base-junction.md` | strict policy, scar families, profile analog coverage, schema implications | the context page is missing or stale relative to current scar-nick results |
| Does this task need harness or contract work? | study skill plus paired companion skill | explicit route to `harness-engineering` or `pragmatic-programming-principles` | the task starts changing study-owned contracts without an explicit pair-with decision |

## Pair-with rules

- Pair with `harness-engineering` when the change touches study status,
  preflight, skill routing, docs integrity, or native-agent bootstrap.
- Pair with `code-change-discipline` when the change touches lane boundaries,
  ontologies, explicit contracts, degraded modes, or no-silent-fallback rules.

## Failure routing

- Missing study record or stale selector: repair `docs/studies/index.yaml` or
  the `docs/studies/retron_hairpin_design/` bundle, then rerun status.
- Missing command-readiness evidence: rerun the pinned preflight surface and
  report `kind`, `surface_id`, and `artifact_id`.
- Boundary drift: route through the paired pragmatic pass before editing the
  study or Cruncher contracts.
