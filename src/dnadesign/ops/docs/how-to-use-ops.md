## How to use Ops

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-20

Use this page when you already know you need the Ops control plane and want the shortest route to the next command. If you are entering from the shell, start with `uv run ops catalog list`.

### Quick terms

- `registry id`: one named procedure or workflow in the shared catalog.
- `related route`: a typed neighboring procedure or workflow linked from one registry entry.
- `progress surface`: the read-only status view for one registered route.
- `campaign manifest`: a YAML file listing the explicit steps that `ops progress campaign` should summarize.

### Discover the right runbook

- `uv run ops catalog list`: browse the shared runbook catalog from the shell.
- `uv run ops catalog list --plane data-plane --query infer`: narrow the catalog by intent when you already know the downstream path.
- `uv run ops catalog list --query "promoter feature matrix"`: find the registered route and adjacent tool docs for that topic without knowing the registry id first.
- `uv run ops catalog list --section tool-sources`: show owner-local tool docs rather than the full procedure inventory.
- `uv run ops catalog list --related-to usr.data-plane.promoter-feature-matrix`: inspect typed related procedures around one registered route once you know the anchor id.
- `uv run ops catalog list --section tool-sources --related-to usr.data-plane.promoter-feature-matrix`: inspect typed related tool docs around one registered route once you know the anchor id.

### Inspect one registered procedure

- `uv run ops catalog show <registry-id>`: open one registered procedure with owner docs, typed related procedures, typed related tool docs, exact deep docs when declared, required progress inputs, and next shell commands.

### Check status and build manifests

- `uv run ops progress show usr.data-plane.promoter-feature-matrix --usr-root <usr-root> --dataset <dataset>`: summarize one registered progress surface once you have the explicit artifact inputs.
- `uv run ops catalog show <registry-id>`: inspect the required progress flags before you run `progress show` if you do not already know the artifact contract.
- `uv run ops progress scaffold <registry-id> ...`: emit an explicit manifest skeleton for one or more registered procedures. It prints YAML to stdout unless you pass `--out`.
- `uv run ops progress scaffold --related-to <registry-id>`: expand one registered procedure into a relation-based manifest starting point. It can cross tool boundaries when the registry metadata declares related routes.
- `uv run ops progress campaign --manifest <manifest.yaml>`: read-only multi-step summary driven by an explicit manifest, not an inferred global engine.

### Continue reading

- [Runbook catalog](../../../../docs/runbooks/README.md): authoritative repo-wide procedure inventory.
- [Ops orchestration index](../../../../docs/operations/README.md): choose the right lifecycle route for init, plan, execute, and verification.
- [Orchestration runbooks](../../../../docs/operations/orchestration-runbooks.md): runbook schema, command order, and contract rules.
- [Packaged runbook presets](../runbooks/presets): checked-in starting points for common orchestration routes.
