## Retron Execution Surfaces

These YAML fragments merge into `ops.study.yaml` `parts.execution_surfaces`.
Keep this directory organized by surface role so agents can inspect one owner
lane without loading every Retron command.

- `workspaces.yaml`: Cruncher workspaces required by status/preflight checks.
- `commands/`: read-only validation, compile, materialize, and primitive probe
  commands.
