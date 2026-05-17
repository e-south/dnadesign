## How to use Ops

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-25

When Ops is already the right surface, start with the next command below. Use
`uv run ops catalog list --simple` if the route is still unknown.

Treat the shell surfaces explicitly: `ops catalog` discovers routes, `ops progress` reads observation/status surfaces, and `ops runbook` owns control-plane execution.

Once you know the route, use `uv run ops progress explain <registry-id>` to see the required status inputs before you touch artifacts.

### Find a route

- `uv run ops catalog list --simple`: browse the catalog in a shorter view before you care about the extra labels.
- `uv run ops catalog list`: browse the shared runbook catalog from the shell.
- `uv run ops catalog list --plane data-plane --query infer`: filter the catalog when you already know the downstream path.
- `uv run ops catalog list --query "promoter feature matrix"`: find the registered route and adjacent tool docs for that topic without knowing the registry id first.
- `uv run ops catalog list --section tool-sources`: show tool docs rather than the full procedure inventory.
- `uv run ops catalog list --related-to usr.data-plane.promoter-feature-matrix`: inspect related procedures around one registered route once you know the anchor id.
- `uv run ops catalog list --section tool-sources --related-to usr.data-plane.promoter-feature-matrix`: inspect related tool docs around one registered route once you know the anchor id.

### Inspect one route

- `uv run ops catalog show <registry-id>`: open one registered procedure with owner docs, related procedures, related tool docs, deeper docs when listed, required status inputs, and next shell commands.

### Check status or build a manifest

- `uv run ops progress explain <registry-id>`: print the required status flags, a ready-to-paste `progress show` command, and any provider-specific notes before you touch artifacts.
- `uv run ops progress show usr.data-plane.promoter-feature-matrix --usr-root <usr-root> --dataset <dataset>`: summarize one status surface once you have the explicit artifact inputs.
- `uv run ops catalog show <registry-id>`: inspect the required status flags before you run `progress show` if you do not already know the artifact contract.
- `uv run ops progress scaffold <registry-id> ...`: emit an explicit manifest skeleton for one or more registered procedures. It prints YAML to stdout unless you pass `--out`.
- `uv run ops progress scaffold --related-to <registry-id>`: expand one registered procedure into a manifest starting point. It can include more than one tool when the catalog links the procedures.
- If you do not know the registry id yet, return to `uv run ops catalog list --simple`; bare `uv run ops progress scaffold` intentionally refuses to guess.
- Replace scaffold placeholders such as `<usr-root>` and narrative sentinels such as `n/a` before you run `ops progress campaign`; placeholder path values now fail explicitly instead of degrading into fake missing roots.
- `uv run ops progress campaign --manifest <manifest.yaml>`: read-only multi-step summary driven by the manifest you provide.

### Continue reading

- [Runbook catalog](../../../../docs/runbooks/README.md): repo-wide command index for procedures and tool docs.
- [Ops orchestration index](../../../../docs/operations/README.md): choose the right lifecycle route for init, plan, execute, and verification.
- [Orchestration runbooks](../../../../docs/operations/orchestration-runbooks.md): runbook schema, command order, and contract rules.
- [Packaged runbook presets](../runbooks/presets): checked-in starting points for common orchestration routes.
