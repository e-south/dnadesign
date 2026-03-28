## OPS preflight checks

**Owner:** dnadesign-maintainers  
**Last verified:** 2026-03-27

This index summarizes the generic readiness check vocabulary that study contracts can declare in `ops.study.yaml`.

| Check kind | What it verifies | Typical non-`ok` meaning | Inputs it needs | Typical cost | Example use |
| --- | --- | --- | --- | --- | --- |
| `command` | one explicit command runs successfully in the declared cwd | the command failed or timed out | argv, cwd, fallback summary | depends on command | validate config, dry-run a tool command |
| `dataset_snapshot` | a declared dataset artifact exists and meets a row-count target | artifact missing, row count unavailable, or rows below target | artifact id, target rows, dataset index | cheap | require a canonical feature matrix before downstream work |
| `environment` | one env var or one accepted set of env vars is configured | required env flag missing | flag names, match mode | cheap | require webhook or credential env wiring |
| `gpu_availability` | the visible GPU count meets a minimum threshold | too few visible GPUs on the current host | minimum visible GPUs, inventory snapshot | host-local | gate local GPU-only prep or dry runs |
| `path_exists` | one declared artifact path exists | required file or directory is missing | artifact id, path resolution contract | cheap | require config, lane output, or notify profile path |
| `runbook_plan` | `ops runbook plan` compiles one declared runbook cleanly | planner contract failed for that runbook | runbook path, repo root | medium | prove a batch route can be planned before execution |
| `scheduler_queue` | scheduler counts stay under declared queue thresholds | queue probe unavailable or queue already too busy | backend, running/queued thresholds | medium | stop new submits when queue pressure is too high |
| `workspace_layout` | a declared workspace root exists and is a directory | workspace missing or malformed | execution surface id, workspace index | cheap | require a real workspace before tool-local checks |

### State interpretation

- `missing` means required material is absent, unreadable, or undeclared
- `attention` means the material exists but the posture is not yet acceptable
- `ok` means the declared contract is satisfied

For `scheduler_queue` checks, “probe unavailable” includes missing scheduler binaries and bounded scheduler-probe timeouts. That state should stay explicit in the rendered evidence rather than hanging the entire preflight surface.

### Where these checks come from

- the vocabulary is declared by OPS generic preflight code
- each study chooses which checks to run by declaring them in `docs/studies/<study-id>/ops.study.yaml`
- blocker semantics come from the checked-in study contract too: `required: true`
  makes a failing check eligible to block the next action, while `required:
  false` keeps the failure visible as advisory-only evidence
- snapshot surfaces stay record-backed; these checks belong to execution-readiness surfaces

For `stress_ethanol_cipro_growth`, the default notify-enabled Infer batch route
marks notify environment, notify profile/event resolution, and notify-enabled
runbook-plan checks as `required: true` so the preflight surface reports strict
submit-readiness instead of "ready with advisories."
