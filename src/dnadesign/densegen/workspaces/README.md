## DenseGen workspaces

This directory contains packaged DenseGen templates and local workspaces created with `dense workspace init`.

### Start here
- Workspace catalog: [catalog.md](catalog.md)
- Template behavior model: [workspace templates](../docs/concepts/workspace-templates.md)
- Workspace layout contract: [workspace model](../docs/concepts/workspace.md)
- Output artifact contract: [outputs reference](../docs/reference/outputs.md)

### Directory policy
- `demo_*`: small baseline templates used for onboarding and checks.
- `study_*`: larger campaign templates.
- `archived/`: preserved historical local runs.
- Local workspaces created by `dense workspace init` are expected under this root unless you set `DENSEGEN_WORKSPACE_ROOT`.

### Packaged workspace shape
Each packaged workspace should contain:
- `README.md`
- `config.yaml`
- `runbook.md`
- `runbook.sh`
- `inputs/`
- `outputs/` (generated at runtime)

### Workspace-local happy path
From inside a packaged workspace:

```bash
# Execute the packaged workspace runbook sequence.
./runbook.sh
```

`runbook.sh` executes the same sequence documented in `runbook.md` step-by-step commands.
Packaged runbooks share the helper at `_shared/runbook_lib.sh` so command sequencing, preflight checks, and failure diagnostics stay aligned across demo/study workspaces.

### Optional Stage-B Showcase Video
Enable this in a workspace `config.yaml` under `plots`:

```yaml
plots:
  video:
    enabled: true  # opt in to video rendering
    mode: all_plans_round_robin_single_video  # single round-robin MP4 across plans
```

Then run `dense plot` (or `dense plot --only dense_array_video_showcase`) to produce:
`outputs/plots/stage_b/all_plans/showcase.mp4`.
