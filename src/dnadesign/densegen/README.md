## DenseGen — Dense Array Generator

**DenseGen** packs transcription factor binding sites (TFBSs) into dense synthetic nucleic acid sequences by wrapping the [dense-arrays](https://github.com/e-south/dense-arrays) ILP solver.

DenseGen is a staged pipeline:

1. **Stage‑A sampling** — mine TFBSs from PWM inputs (via [FIMO](https://meme-suite.org/meme/doc/fimo.html)) to build TFBS pools.
2. **Stage‑B sampling** — iterative subsampling from the above TFBS libraries.
3. **Optimization + postprocess** — assemble dense arrays under constraints.
4. **Artifacts** — write Parquet tables, manifests, plots, and audit reports under `outputs/`.

For a walkthrough, start with the demo:
- **Demo walkthrough:** [docs/demo/demo_basic.md](docs/demo/demo_basic.md)

---

### Contents

- [Quick start](#quick-start)
- [Documentation map](#documentation-map)

---

### Quick start

Prerequisites include Python, dense-arrays, and a MILP solver. CBC is open-source; GUROBI is supported if installed and licensed. Stage‑A PWM sampling requires MEME Suite (`fimo` on PATH).

From the repo root:

```bash
uv sync --locked
pixi install
pixi run fimo --version

# Option A: cd into the workspace
cd src/dnadesign/densegen/workspaces/demo_meme_three_tfs  # enter demo workspace
CONFIG="$PWD/config.yaml"  # point to workspace config

# Option B: run from anywhere in the repo
CONFIG=src/dnadesign/densegen/workspaces/demo_meme_three_tfs/config.yaml  # config path from repo root

# Choose a runner (pixi is the default in this repo; uv is optional).
# If `dense` is already an alias, remove it before defining the function.
unalias dense 2>/dev/null
dense() { pixi run dense -- "$@"; }  # convenience wrapper

# Optional: uv-only wrapper
# dense() { uv run dense "$@"; }

# From here on, commands use $CONFIG for clarity; if you're in the workspace, you can omit -c.
```

Run the packaged demo workspace:

```bash
# 1) Validate schema + solver availability
dense validate-config --probe-solver -c "$CONFIG"

# 2) Stage‑A: build TFBS pools (required before `dense run` unless you pass --rebuild-stage-a)
dense stage-a build-pool --fresh -c "$CONFIG"

# 3) Run generation (use --resume or --fresh if outputs already exist)
#    By default, this auto-runs plots configured in `plots`; use --no-plot to skip.
dense run -c "$CONFIG"

# 4) Plot a minimal diagnostics set (useful to re-render plots on demand)
dense plot --only stage_a_summary,placement_map -c "$CONFIG"
```

---

### Documentation map

Progressive guides:

* **Demo walkthrough:** [docs/demo/demo_basic.md](docs/demo/demo_basic.md)
* **Workspace layout:** [docs/guide/workspace.md](docs/guide/workspace.md)
* **Inputs (Stage‑A):** [docs/guide/inputs.md](docs/guide/inputs.md)
* **Sampling (Stage‑A + Stage‑B):** [docs/guide/sampling.md](docs/guide/sampling.md)
* **Generation (constraints + Stage‑B):** [docs/guide/generation.md](docs/guide/generation.md)
* **Outputs + metadata:** [docs/guide/outputs-metadata.md](docs/guide/outputs-metadata.md)
* **Postprocess:** [docs/guide/postprocess.md](docs/guide/postprocess.md)

References:

* **CLI operator manual:** [docs/reference/cli.md](docs/reference/cli.md)
* **Config schema:** [docs/reference/config.md](docs/reference/config.md)
* **Outputs + manifests:** [docs/reference/outputs.md](docs/reference/outputs.md)
* **Motif artifact contract:** [docs/reference/motif_artifacts.md](docs/reference/motif_artifacts.md)

Developer notes:

* **Architecture:** [docs/dev/architecture.md](docs/dev/architecture.md)

---

@e-south
