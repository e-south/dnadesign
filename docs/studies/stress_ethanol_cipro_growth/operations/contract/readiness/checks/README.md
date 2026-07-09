## Stress Readiness Checks

These fragments merge into `ops.study.yaml` `parts.preflight.checks`.

Each file or nested directory is named after the lifecycle phase it checks.
Large phase checks split by owner/action lane; `infer_batch_preparation/`
contains sequence-view, completion, runtime, validation, dry-run, Notify, and
batch-runbook fragments. Keep cross-phase scope, group bindings, and next-scope
rules in the sibling `scope.yaml`, `group-bindings.yaml`, and `next-scope.yaml`
files.

`opal_candidate_table_pre_assay.yaml` is the current main-path readiness gate:
it validates the OPAL candidate table row contract, including measured Reader
round-0 rows that already have candidate IDs, sequence parity, and selected X,
without coupling OPAL readiness to LatentDNA appendix visualizations.
