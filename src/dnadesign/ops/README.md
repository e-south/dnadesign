![Ops banner](assets/ops-banner.svg)

Ops manages batch orchestration and read-only status across tools. It turns
runbooks into preflight, verification, submit, and audit steps so scheduler work
stays explicit.

Start with `uv run ops catalog list --simple` when the route id is unknown.

Ops core owns the neutral CLI, registry loading, path semantics, orchestration,
and rendering surfaces. Boundary-owned `*/ops/status.registry.yaml` fragments
publish status routes, and Ops imports provider code only for the selected
route. Study-family implementation code stays under `src/dnadesign/studies/`.

Study readiness is contract-driven: `ops.study.yaml` declares scope grouping,
artifact refs, execution surfaces, and generic check kinds such as `command`,
`workspace_layout`, `scheduler_queue`, `gpu_availability`, `path_exists`,
`dataset_snapshot`, `sequence_view_contract`, and `runbook_plan`. The contract
may describe a sequential phase ladder or a nonsequential track map; Ops should
not assume one study introspection style.

Command families are neutral surfaces: `ops catalog` is discovery, `ops
progress` is observation, and `ops runbook` is deterministic batch control.

## Glossary

- discovery plane: `ops catalog`, used for route discovery and ownership docs
- observation plane: `ops progress`, used for read-only status and manifest views
- control plane: `ops runbook`, used for init, plan, active-job discovery, and execute
- record plane: checked-in study records, manifests, and audit artifacts already on disk
- execution-readiness plane: host, workspace, scheduler, and command blockers for the next action
- status kind: the shared status implementation contract behind one or more public routes
- public route / registry id: the command-facing identifier such as `ops.control-plane.orchestration`
- summary scope: the level a status summarizes, such as `repo`, `workspace`, or `host`
- cost class: the expected read cost of a status surface, such as `cheap` or `deep`

Use Ops when:
- you need a shared orchestration layer for scheduler work, packaged runbooks, or read-only status over a registered route
- you want to browse the shared command index from the terminal with `uv run ops catalog list --simple` or `uv run ops catalog list`

Do not use Ops when:
- a tool already owns the durable dataset mutation, such as Construct -> USR -> Infer shared-dataset work
- you need the boundary-local runtime semantics first; start from the tool docs or shared USR workflow docs, then return to Ops if you need orchestration around that route

For repo-wide runbook discovery, start with `docs/runbooks/README.md`; `ops`
does not keep a second registry.

## Common entrypoints

- `uv run ops catalog list --simple`: start with a quick inventory from the shell.
- `uv run ops catalog show <registry-id>`: inspect one registered route, its owner docs, and related procedures.
- `uv run ops progress explain <registry-id>`: print the required flags before you use a status surface.
- `uv run ops runbook fill-infer --study-dir docs/studies/<study-id>`: inspect a study's Infer runbooks and plan only incomplete sequence-view lanes.
- Use [How to use Ops](docs/how-to-use-ops.md) for the expanded command ladder and [Ops orchestration index](../../../docs/operations/README.md) once you are in the runbook lifecycle.

## Python API

Maintainers embedding OPS from Python should import the explicit service layer
at `dnadesign.ops.api`:

```python
from dnadesign.ops import api as ops_api
```

Use that module for intentional service entrypoints such as catalog loading,
status execution, runbook loading, plan building, execution, and active-job
discovery. Do not reach into CLI modules for maintainership automation.

## Documentation

- [Ops docs index](docs/README.md): Ops package docs, packaged presets, and links to repo-level control-plane docs.
- [How to use Ops](docs/how-to-use-ops.md): command guide for catalog discovery, runbook inspection, status checks, and manifest scaffolding.
- [Runbook catalog](../../../docs/runbooks/README.md): repo-level command index for cross-tool procedures and tool entrypoints.
- [Ops orchestration index](../../../docs/operations/README.md): runbook lifecycle docs for init, plan, execute, and status checks.
- [Orchestration runbooks](../../../docs/operations/orchestration-runbooks.md): runbook schema, command sequence, and contract rules.
- [Packaged runbook presets](runbooks/presets): checked-in starter runbooks for common orchestration routes.
- [Repository docs index](../../../docs/README.md): repo-wide docs index for cross-tool workflows.
