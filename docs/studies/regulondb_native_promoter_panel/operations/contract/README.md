## RegulonDB Ops Contract Parts

These YAML fragments are loaded by `../ops.study.yaml`.

- `lifecycle/`: lifecycle mode and phase order.
- `surfaces/`: artifact refs and split execution-surface fragments.
- `status/`: snapshot scope.
- `readiness/`: providerless readiness scope, group bindings, next-scope rules,
  and phase-named checks.

Put command groups and downstream context in
`../runtime/command-groups/pipeline.yaml`.
