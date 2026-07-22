## Stress Execution Surfaces

These fragments merge into `ops.study.yaml` `parts.execution_surfaces`.

- `workspaces.yaml`: workspace and scheduler surfaces.
- `runbooks/`: batch runbook declarations by owner lane.
- `commands/`: read-only validation and inventory commands by owner lane.

Keep generated outputs out of this directory. Add a new fragment only when a
surface family would make an existing file hard to scan.
