## Infer Batch Preparation Checks

These fragments all merge into the `infer_batch_preparation` preflight bucket.
The phase name remains for lifecycle compatibility, while the files below keep
the actual checks navigable by owner/action lane.

- `sequence-views.yaml`: sequence-view contracts and source datasets.
- `completion.yaml`: Infer feature-completion and workspace checks.
- `runtime-environment.yaml`: environment, GPU, and host-readiness checks.
- `config-validation.yaml`: Infer config validation commands.
- `dry-run.yaml`: Infer dry-run commands.
- `notify.yaml`: Notify profile and event-resolution checks.
- `batch-runbooks.yaml`: scheduler queue and runbook-plan checks.
