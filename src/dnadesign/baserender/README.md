## baserender vNext

Contract-first sequence rendering with explicit adapters, strict schemas, and pluggable feature/effect rendering.

### Package layout

```text
src/dnadesign/baserender/
  src/         # runtime package (import root: dnadesign.baserender.src)
    workspace.py # workspace discovery + init + resolution
    core/      # model + contracts
    config/    # cruncher_showcase_job + style_v1 schema
    io/        # parquet row source
    adapters/  # densegen, generic, cruncher adapters
    pipeline/  # transforms/plugins + selection service
    render/    # renderer + effects
    outputs/   # images + video
    reporting/ # run report
  docs/        # contracts + architecture + examples
  tests/       # package-level tests
  styles/      # style presets
  workspaces/  # run-scoped workbenches
```

### CLI

```bash
baserender job validate <job.yaml>
baserender job run <job.yaml|name>
baserender job validate --workspace <name>
baserender job run --workspace <name>
baserender job run --workspace <name> --workspace-root <dir>
baserender workspace init <name>
baserender workspace list
baserender style list
baserender style show <preset>
```

No v1/v2 job support is included.

### Cruncher Showcase Job Summary

- `version: 3` (Cruncher showcase job contract)
- strict unknown-key rejection at every nesting level
- explicit adapter declaration (`densegen_tfbs`, `generic_features`, `cruncher_best_window`)
- explicit `outputs[]` list (required, non-empty)
- no implicit outputs: only declared outputs are produced
- non-workspace default `results_root` resolves to `<caller_root>/results` (`caller_root` defaults to current working directory)
- workspace jobs (`job.yaml` with sibling `inputs/` and `outputs/`) default `results_root` to `<workspace>/outputs`

See:
- `docs/contracts/cruncher_showcase_job.md`
- `docs/contracts/record_v1.md`
- `docs/contracts/style_v1.md`
- `docs/architecture/overview.md`

### Adapters

- `densegen_tfbs`: DenseGen TFBS list-of-dicts -> `Feature(kind="kmer")`
- `generic_features`: normalized feature/effect/display payload columns
- `cruncher_best_window`: Cruncher elites + hits + PWM metadata -> kmer + motif_logo

### Feature/Effect model

Rendering consumes `Record` objects with:
- `features[]` (typed by `kind`)
- `effects[]` (typed by `kind`)
- `display` metadata for overlay text and legend labels

Unknown feature/effect kinds are fatal by policy.

### Public API

Supported integration surface is exported from `dnadesign.baserender`:

- `initialize_runtime`
- `run_cruncher_showcase_job`
- `validate_cruncher_showcase_job`
- `load_record_from_parquet`
- `Record`, `Feature`, `Effect`, `Display`, `Span`
- `render_record_figure`
- `render_record_grid_figure`
- `render_parquet_record_figure`

Internal modules under `dnadesign.baserender.src.*` are non-contractual and may change.

### Examples

- DenseGen job: `docs/examples/densegen_job.yaml`
- Cruncher job: `docs/examples/cruncher_job.yaml`

### Workspace demos

Two curated self-contained workspaces are included:

- `workspaces/demo_densegen_render`
- `workspaces/demo_cruncher_render`

Run them with:

```bash
uv run baserender job run --workspace demo_densegen_render --workspace-root src/dnadesign/baserender/workspaces
uv run baserender job run --workspace demo_cruncher_render --workspace-root src/dnadesign/baserender/workspaces
```
`demo_cruncher_render` is the fast iteration path for the Cruncher elites-showcase visual contract. It runs from normalized Record-shape rows (`id`, `sequence`, `features`, `effects`, `display`) via `generic_features`, and keeps source-like Cruncher artifacts in `inputs/` for runtime-shape reference.

### Video encoding

Video output uses MP4/H.264 with:
- even dimensions
- `yuv420p`
- `+faststart`
- letterboxing when target frame is larger than rendered frame
