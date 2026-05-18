## RegulonDB Ops Contract Parts

These YAML fragments are loaded by `../ops.study.yaml`.

- `lifecycle/`: lifecycle mode and phase order.
- `surfaces/`: artifacts and execution-surface refs.
- `status/`: snapshot scope.
- `readiness/`: providerless readiness metadata and checks.

Put command groups and downstream context in
`../runtime/command-groups/pipeline.yaml`.
