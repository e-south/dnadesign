Run-scoped baserender workbenches live here. Each workspace should contain:

- `job.yml`
- `inputs/`
- `outputs/` (render artifacts + reports by default)
- `reports/` (optional workspace notes or review docs)

Use:

```bash
uv run baserender workspace init demo_run
uv run baserender workspace list
uv run baserender job validate --workspace demo_run
uv run baserender job run --workspace demo_run
# If workspaces are outside the package default:
uv run baserender job run --workspace demo_run --workspace-root /path/to/workspaces
```

Design notes:

- `job.yml` paths are resolved relative to the workspace root.
- If `results_root` is omitted in a workspace `job.yml`, baserender uses `outputs/`.
- Default image/report outputs for workspace jobs are written under `outputs/` directly.
- Keep ad-hoc workspaces out of git; track only curated demos here.

Curated demos tracked in git:

- `demo_densegen_render`
  - Input: `inputs/input.parquet` (DenseGen-style TFBS annotations)
  - Output: PNG stills under `outputs/images/`
- `demo_cruncher_render`
  - Input: `inputs/elites.parquet` + `inputs/elites_hits.parquet` + `inputs/config_used.yaml`
    (Cruncher-style elite + hit + PWM artifacts)
  - Output: PDF stills under `outputs/images/`

Run demos:

```bash
uv run baserender job validate --workspace demo_densegen_render --workspace-root src/dnadesign/baserender/workspaces
uv run baserender job run --workspace demo_densegen_render --workspace-root src/dnadesign/baserender/workspaces

uv run baserender job validate --workspace demo_cruncher_render --workspace-root src/dnadesign/baserender/workspaces
uv run baserender job run --workspace demo_cruncher_render --workspace-root src/dnadesign/baserender/workspaces
```
