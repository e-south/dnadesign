## How to use Ops

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-20

Use this page when you already know you need Ops and want the next command. Start with `uv run ops catalog list` if you are coming from the shell.

If this is your first stop in Ops, prefer `uv run ops catalog list --simple` for a shorter catalog view before you narrow by taxonomy.

### Quick terms

- `procedure id`: one named procedure or workflow in the shared catalog.
- `related procedure`: a neighboring procedure or workflow linked from one registry entry.
- `status view`: the read-only status view for one registered route.
- `campaign manifest`: a YAML file listing the explicit steps that `ops progress campaign` should summarize.

### Discover the right runbook

- `uv run ops catalog list --simple`: browse the catalog in a shorter view before you care about type, plane, or progress-kind labels.
- `uv run ops catalog list`: browse the shared runbook catalog from the shell.
- `uv run ops catalog list --plane data-plane --query infer`: filter the catalog when you already know the downstream path.
- `uv run ops catalog list --query "promoter feature matrix"`: find the registered route and adjacent tool docs for that topic without knowing the registry id first.
- `uv run ops catalog list --section tool-sources`: show tool docs rather than the full procedure inventory.
- `uv run ops catalog list --related-to usr.data-plane.promoter-feature-matrix`: inspect related procedures around one registered route once you know the anchor id.
- `uv run ops catalog list --section tool-sources --related-to usr.data-plane.promoter-feature-matrix`: inspect related tool docs around one registered route once you know the anchor id.

### Inspect one registered procedure

- `uv run ops catalog show <registry-id>`: open one registered procedure with owner docs, related procedures, related tool docs, deeper docs when listed, required progress inputs, and next shell commands.

### Check status and build manifests

- `uv run ops progress explain <registry-id>`: print the required progress flags, a ready-to-paste `progress show` command, and any adapter-specific notes before you touch artifacts.
- `uv run ops progress show usr.data-plane.promoter-feature-matrix --usr-root <usr-root> --dataset <dataset>`: summarize one status view once you have the explicit artifact inputs.
- `uv run ops catalog show <registry-id>`: inspect the required progress flags before you run `progress show` if you do not already know the artifact contract.
- `uv run ops progress scaffold <registry-id> ...`: emit an explicit manifest skeleton for one or more registered procedures. It prints YAML to stdout unless you pass `--out`.
- `uv run ops progress scaffold --related-to <registry-id>`: expand one registered procedure into a manifest starting point. It can include more than one tool when the catalog links the procedures.
- If you do not know the registry id yet, return to `uv run ops catalog list --simple`; bare `uv run ops progress scaffold` intentionally refuses to guess.
- `uv run ops progress campaign --manifest <manifest.yaml>`: read-only multi-step summary driven by the manifest you provide.

### Continue reading

- [Runbook catalog](../../../../docs/runbooks/README.md): repo-wide command index for procedures and tool docs.
- [Ops orchestration index](../../../../docs/operations/README.md): choose the right lifecycle route for init, plan, execute, and verification.
- [Orchestration runbooks](../../../../docs/operations/orchestration-runbooks.md): runbook schema, command order, and contract rules.
- [Packaged runbook presets](../runbooks/presets): checked-in starting points for common orchestration routes.
