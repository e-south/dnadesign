# DenseGen -> USR -> Notify Lifecycle Design

Date: 2026-02-12  
Status: Proposed (validated in brainstorming)

## Scope and outcomes

This design defines what Notify should report for DenseGen lifecycle monitoring while keeping messages compact and operationally useful for Slack readers. The primary outcome is an action-oriented signal stream with controlled periodic health updates and strict, tool-scoped metrics contracts.

Chosen product behavior:
- Send lifecycle notifications for `started`, `resumed`, `running` health checkpoints, `failed`, and `completed`.
- Use a hybrid health cadence:
  - emit on each 10% quota progress boundary crossing
  - emit heartbeat every 30 minutes while the run is active
- Include DenseGen-specific compact summary metrics in health and completion notifications:
  - TFBS library coverage
  - quota progress
  - plan success ratio
  - runtime duration
  - total rows written
- Use Slack-friendly formatting with one summary line and compact bullet lines.
- Avoid tabular text formatting that may render poorly in Slack clients.

Out of scope for this pass:
- site-level inclusion metrics in notifications (explicitly deferred)
- dense, per-plan/per-array notification spam
- fallback inference of DenseGen metrics in Notify

## Contract changes and data model

DenseGen remains the source of truth for DenseGen runtime metrics; Notify remains a delivery and rendering component.

`densegen_health` USR events must include a namespaced metrics block:
- `metrics.densegen.tfbs_total_library`
- `metrics.densegen.tfbs_unique_used`
- `metrics.densegen.tfbs_coverage_pct`
- `metrics.densegen.plans_attempted`
- `metrics.densegen.plans_solved`
- `metrics.densegen.rows_written_session`
- `metrics.densegen.run_quota`
- `metrics.densegen.quota_progress_pct`

Additional existing fields still supported where present:
- `metrics.compression_ratio`
- `metrics.flush_count`

Lifecycle/status mapping:
- start: `args.status=started`
- resume: `args.status=resumed`
- running health: `args.status=running`
- terminal success: `args.status=completed`
- terminal failure: `args.status=failed` or explicit failure action

Contract policy:
- DenseGen policy is strict and assertive.
- Missing required DenseGen metrics in DenseGen lifecycle notifications are treated as schema failures for that policy, not silently dropped or inferred.
- Generic policy behavior remains unchanged for non-DenseGen tools.

## Notification content and formatting

### Message layout

All DenseGen notifications use:
1. One summary line
2. 3-5 compact bullet lines

No ASCII tables or pseudo-table blocks.

### Message templates

Started:
- `DenseGen started | run=<run_id> | dataset=<dataset_name>`
- `• Quota: <run_quota> rows`

Resumed:
- `DenseGen resumed | run=<run_id> | dataset=<dataset_name>`
- `• Progress: <quota_pct>% (<rows>/<quota>)`

Health:
- `DenseGen health | run=<run_id> | dataset=<dataset_name>`
- `• Quota: <quota_pct>% (<rows>/<quota>)`
- `• TFBS library coverage: <coverage_pct>% (<used>/<library>)`
- `• Plan success: <solved>/<attempted> (<success_pct>%)`
- `• Runtime: <hh:mm:ss>`

Failure:
- `DenseGen failed | run=<run_id> | dataset=<dataset_name>`
- `• Stage: <action_or_phase>`
- `• Error: <compact_error>`

Completed:
- `DenseGen complete | run=<run_id> | dataset=<dataset_name>`
- `• Duration: <hh:mm:ss>`
- `• Final quota: <final_pct>% (<rows>/<quota>)`
- `• Final TFBS coverage: <final_cov>% (<used>/<library>)`
- `• Final plan success: <solved>/<attempted> (<success_pct>%)`

## Triggering, dedup, and noise controls

Health notification trigger (hybrid):
- progress edge: send when quota crosses 10, 20, ..., 100
- heartbeat: send every 30 minutes while running

Dedup rule:
- Build a compact metric tuple:
  - `(quota_step, tfbs_cov_rounded, plans_solved, plans_attempted, rows_written_session)`
- If a new health candidate has the same tuple as the previous emitted health message and is not a heartbeat boundary, suppress it.

Failure policy:
- failure and completion are never suppressed
- resume is never suppressed

Reliability behavior:
- Keep existing spool/cursor semantics unchanged.
- Keep retry/backoff semantics unchanged.
- Keep SCC TLS explicitness unchanged (`--tls-ca-bundle` / `SSL_CERT_FILE`).

## Architecture and implementation plan

Implementation is split into two components:

1. DenseGen event emission
- Emit required `metrics.densegen.*` fields on `densegen_health`.
- Emit explicit `resumed` lifecycle status when resuming.
- Ensure values are cumulative run-to-date, not interval-only values.

2. Notify rendering and trigger engine
- Add a DenseGen-specific formatter path selected by policy/tool.
- Add a health trigger gate module:
  - quota-edge detector
  - heartbeat timer
  - dedup tuple tracker
- Render Slack body as summary line + bullet lines.

Design constraints:
- no fallback inference from partial events for DenseGen policy
- no mixed concerns: DenseGen computes metrics; Notify renders/transmits
- maintain generic provider contract and existing CLI surface where possible

## Testing strategy

Unit tests:
- DenseGen metrics schema validation (required keys and value types)
- status mapping for started/running/resumed/completed/failed
- quota-edge trigger behavior
- heartbeat trigger behavior
- dedup suppression behavior
- completion summary metric correctness

Integration tests:
- local qsub-style smoke harness with real watcher process and local HTTP capture endpoint
- interruption and resume flow:
  - start watcher
  - append running events
  - stop/restart watcher
  - append resumed + terminal events
  - verify cursor resume and expected notification sequence
- spool/drain reliability path under transient delivery failure

Rendering tests:
- snapshot-style checks for bullet-list message shape
- explicit assertion that table-like formatting is not produced

## Rollout and docs

Rollout order:
1. add metrics fields in DenseGen event emission and tests
2. add Notify DenseGen formatter + trigger gate and tests
3. update operator docs with final message examples and trigger semantics
4. run notify + harness test suites in CI

Docs to update during implementation:
- `docs/notify/usr_events.md`
- DenseGen workflow docs that describe USR/Notify observability
- qsub runbook examples where health cadence is relevant

This design intentionally favors compact, actionable operator signal over high-volume telemetry while preserving strict contracts and future extensibility for other tools.
