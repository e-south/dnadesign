## Workspace model

This concept page explains the expected DenseGen workspace layout and why DenseGen commands are designed around that layout. Read it when you need predictable path resolution, safe reset behavior, and clear artifact boundaries.

### Expected layout
This section defines the workspace structure used by packaged templates and operator commands.

```text
workspace/
  config.yaml
  inputs/
  outputs/
    pools/
    libraries/
    tables/
    meta/
    plots/
    notebooks/
    logs/
```

### Why this layout matters
This section explains the operational benefits of keeping each run workspace self-contained.

- `config.yaml` is the anchor for relative path resolution.
- Run diagnostics stay colocated with produced artifacts.
- `campaign-reset` can clear outputs without touching inputs/config.
- Reproducibility improves because workspace state is explicit.

### Core workspace commands
This section lists the workspace-scoped commands used most often by operators.

```bash
# Resolve repo root and pin workspace root so paths are deterministic.
REPO_ROOT="$(git rev-parse --show-toplevel)"
WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces"

# Create a new workspace from a packaged template.
uv run dense workspace init --id my_run --root "$WORKSPACE_ROOT" --from-workspace demo_tfbs_baseline --copy-inputs --output-mode local

# Print workspace path resolution details.
uv run dense workspace where --format json
```

### Resume and fresh behavior
This section clarifies default behavior when outputs already exist.

- `dense run` resumes by default when prior run state is present.
- `dense run --fresh` clears outputs before running.
- `dense campaign-reset` clears outputs while preserving `config.yaml` and `inputs/`.

### Related documents
This section links to the adjacent docs that explain template classes and stage behavior.

- Read **[workspace templates](workspace-templates.md)** for packaged workspace taxonomy.
- Read **[pipeline lifecycle](pipeline-lifecycle.md)** for stage and artifact flow.
- Read **[DenseGen quick checklist](quick-checklist.md)** for a compact run command sequence.
