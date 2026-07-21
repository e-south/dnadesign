## Stress Readiness Checks

These fragments merge into `ops.study.yaml` `parts.preflight.checks`.

Each file or nested directory is named after the lifecycle phase it checks.
Large phase checks split by owner/action lane; `infer_batch_preparation/`
contains sequence-view, completion, runtime, validation, dry-run, Notify, and
batch-runbook fragments. Keep cross-phase scope, group bindings, and next-scope
rules in the sibling `scope.yaml`, `group-bindings.yaml`, and `next-scope.yaml`
files.

`opal_assay_b1_order_ready.yaml` is the current main-path readiness gate. It
replays the MSRB campaign and selection evidence; the study-owned preflight adds
the exact accepted synthesis-handoff record check. The earlier candidate-table
gate remains available in full-scope preflight without coupling OPAL readiness
to LatentDNA appendix visualizations.
