## `densegen` for agents

Supplement to repo-root `AGENTS.md` with `densegen`-specific locations + run shape.

### Key paths
- README: `src/dnadesign/densegen/README.md`
- Tool code: `src/dnadesign/densegen/src/` (entrypoint lives here)
- Packaged demo template id: `demo_meme_two_tf`
- Outputs: per-workspace `outputs/` (generated; run artifacts live in tables/plots/report)

### External deps (do not install unless asked)
- MILP solver required (e.g., CBC or GUROBI).

### Generated vs hand-edited
- Hand-edited: `workspaces/*/config.yaml`, `workspaces/*/inputs/**`, pipeline code
- Generated: `workspaces/*/outputs/**` (parquet, plots, logs, manifests)

### Run ergonomics (explicit)
- If run outputs already exist (e.g., `outputs/tables/attempts.parquet` or `outputs/meta/run_state.json`),
  use `dense run --resume` to continue or `dense run --fresh` to clear outputs and start over.
  Runs do not auto-resume.

### Commands (copy/paste)
DenseGen CLI is exposed as `dense` in this repo:
```bash
uv run dense --help

# Workspace-first demo flow (no repo-root paths).
uv run dense workspace init --id demo --template-id demo_meme_two_tf --copy-inputs
cd demo
uv run dense validate-config --probe-solver
uv run dense inspect inputs
uv run dense inspect config
uv run dense stage-a build-pool
uv run dense stage-b build-libraries
uv run dense run
uv run dense inspect run --library --events
uv run dense ls-plots
uv run dense plot --only placement_map,tfbs_usage,run_health
uv run dense plot --only stage_a_summary,stage_b_summary
```

### MEME Suite / FIMO pressure testing

When the demo uses `scoring_backend: fimo`, prefer the pixi workflow so MEME Suite is on PATH
(run these from a workspace directory):

```bash
pixi run fimo --version
pixi run dense validate-config --probe-solver
pixi run dense run --fresh
```

Pixi includes a `dense` task alias in `pixi.toml`, so `pixi run dense ...` works without `uv run`.
Pixi tasks run from the repo root, so pass `-c /abs/path/to/workspace/config.yaml` (or set a shell
alias) when operating on a workspace.
If `fimo` is not found, either use `pixi run ...` or set `MEME_BIN` to the MEME bin directory
(`.pixi/envs/<env>/bin`).

### Candidate artifacts

When `keep_all_candidates_debug: true`, candidate artifacts land under
`outputs/pools/candidates/` (files named `candidates__<label>.parquet`) and are overwritten each run.
If you want to keep a snapshot, copy the directory elsewhere before rerunning.

### Tests

If you modify `densegen`, run:

```bash
uv run pytest -q
```

For FIMO-enabled tests, use pixi so MEME Suite is on PATH:

```bash
pixi run pytest
```
