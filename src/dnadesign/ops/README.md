![Ops banner](assets/ops-banner.svg)

Ops provides the shared command surface for route discovery, read-only status,
and deterministic batch orchestration across tool-owned workflows.

## Documentation

- [Ops docs index](docs/README.md): catalog, progress, runbook, and packaged
  preset routing.
- [How to use Ops](docs/how-to-use-ops.md): command guide for catalog
  discovery, status checks, and runbook inspection.
- [Runbook catalog](../../../docs/runbooks/README.md): repo-level command index
  for cross-tool procedures.
- [Repository docs index](../../../docs/README.md): cross-tool workflow routing.

## Source Orientation

- `cli/commands/progress.py`: Typer/Click command routing for status commands.
- `cli/commands/progress_render.py`: text, JSON, and YAML output rendering for
  progress commands.
- `cli/commands/progress_status_specs.py`: lazy status-kind and campaign status
  accessors used by the CLI.
- `status/`: provider registry, status models, and service contracts.
- `orchestrator/`: runbook planning, execution, scheduler state, and gates.

Keep command routing, rendering, and status-provider loading separate. A study
package owns any study-specific status behavior and registers its metadata
through the `dnadesign.ops.status_registries` entry-point group.
