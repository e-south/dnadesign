## `densegen` for agents

Supplement to repo-root `AGENTS.md` with `densegen`-specific locations + run shape.

### Key paths
- README: `src/dnadesign/densegen/README.md`
- Tool code: `src/dnadesign/densegen/src/` (entrypoint lives here)
- Default demo workspace: `src/dnadesign/densegen/workspaces/demo_meme_two_tf/`
- Demo config: `src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml`
- Outputs: per-workspace `outputs/` (generated)

### External deps (do not install unless asked)
- MILP solver required (e.g., CBC or GUROBI).

### Generated vs hand-edited
- Hand-edited: `workspaces/*/config.yaml`, `workspaces/*/inputs/**`, pipeline code
- Generated: `workspaces/*/outputs/**` (parquet, plots, logs, manifests)

### Run ergonomics (explicit)
- If `outputs/` already exists, use `dense run --resume` to continue or `dense run --fresh` to clear
  outputs and start over. Runs do not auto-resume.

### Commands (copy/paste)
DenseGen CLI is exposed as `dense` in this repo:
```bash
uv run dense --help

pixi run dense validate-config -c src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml
uv run dense inspect plan     -c src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml
pixi run dense run      -c src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml --fresh
uv run dense plot     -c src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml
# subset example:
uv run dense plot -c src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml --only tf_usage,tf_coverage
```

### MEME Suite / FIMO pressure testing

When the demo uses `scoring_backend: fimo`, prefer the pixi workflow so MEME Suite is on PATH:

```bash
pixi run fimo --version
pixi run dense validate-config -c src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml
pixi run dense run -c src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml --no-plot --fresh
```

Pixi includes a `dense` task alias in `pixi.toml`, so `pixi run dense ...` works without `uv run`.
If `fimo` is not found, either use `pixi run ...` or set `MEME_BIN` to the MEME bin directory
(`.pixi/envs/<env>/bin`).

### Candidate artifacts

When `keep_all_candidates_debug: true`, candidate artifacts land under:
`outputs/candidates/<run_id>/<input_name>/` and are overwritten each run for that run_id.
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
