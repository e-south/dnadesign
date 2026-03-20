![Ops banner](assets/ops-banner.svg)

Ops is the cross-tool orchestration control plane for deterministic batch workflows. It turns runbook intent into explicit preflight, verification, and submit phases with auditable outputs, so scheduler work stays legible instead of dissolving into ad hoc shell glue.

Use Ops when:
- you need a shared orchestration layer for scheduler work, packaged runbooks, or read-only status over a registered route
- you want to browse the shared route map from the terminal with `uv run ops catalog list`

Do not use Ops when:
- a tool already owns the durable dataset mutation, such as Construct -> USR -> Infer source-of-truth work
- you need the boundary-local runtime semantics first; start from the tool docs or shared USR workflow docs, then return to Ops if you need orchestration around that route

For repo-wide runbook discovery, `ops` exposes a shared catalog view in `docs/runbooks/README.md`; it does not own a second registry.
Typical flow: browse the catalog, inspect one registered procedure, then either scaffold a manifest or move into the runbook lifecycle commands.

## Documentation

- [Ops docs index](docs/README.md): package-local map for when to use Ops, packaged presets, and the repo-wide control-plane surfaces it points to.
- [How to use Ops](docs/how-to-use-ops.md): quick command guide for catalog discovery, runbook inspection, status checks, and manifest scaffolding.
- [Runbook catalog](../../../docs/runbooks/README.md): repo-level inventory of authoritative cross-tool procedures and owner-local tool entrypoints.
- [Ops orchestration index](../../../docs/operations/README.md): task-first router for runbook lifecycle choices.
- [Orchestration runbooks](../../../docs/operations/orchestration-runbooks.md): runbook schema, command sequence, and contract rules.
- [Packaged runbook presets](runbooks/presets): checked-in starter runbooks for common orchestration routes.
- [Repository docs index](../../../docs/README.md): repo-wide route map for cross-tool workflows.
