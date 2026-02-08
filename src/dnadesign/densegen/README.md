## DenseGen — Dense Array Generator

DenseGen places transcription factor binding sites (TFBSs) into synthetic sequences by driving the [dense-arrays](https://github.com/e-south/dense-arrays) ILP solver.

Pipeline stages:

1. **Stage-A sampling**: build TFBS pools from inputs (PWM mining via [FIMO](https://meme-suite.org/meme/doc/fimo.html), binding sites, or sequence sources).
2. **Stage-B library sampling**: choose per-run solver libraries from Stage-A pools.
3. **Optimization + postprocess**: solve sequence layouts under plan constraints.
4. **Artifacts**: write tables, manifests, plots, and reports under `outputs/`.

Start with the demo index:
- [docs/demo/README.md](docs/demo/README.md)

If you are integrating the full stack (DenseGen -> USR -> Notify), start here:
- [docs/demo/demo_usr_notify.md](docs/demo/demo_usr_notify.md)

Note on event logs:
- DenseGen writes runtime diagnostics to `outputs/meta/events.jsonl`.
- Notify consumes USR dataset mutation events from `<usr_root>/<dataset>/.events.log`.

---

### Contents

- [Choose a demo](#choose-a-demo)
- [Quick start (binding-sites baseline)](#quick-start-binding-sites-baseline)
- [Quick start (canonical PWM)](#quick-start-canonical-pwm)
- [Documentation map](#documentation-map)

---

### Choose a demo

- **Binding-sites baseline demo** (recommended first): no added plan constraints.
  - [docs/demo/demo_binding_sites.md](docs/demo/demo_binding_sites.md)
- **Canonical Cruncher PWM flow** (main workflow): three-TF motif artifacts from Cruncher.
  - [docs/workflows/cruncher_pwm_pipeline.md](docs/workflows/cruncher_pwm_pipeline.md)
  - [docs/demo/demo_pwm_artifacts.md](docs/demo/demo_pwm_artifacts.md)

### Quick start (binding-sites baseline)

From the repo root:

```bash
uv sync --locked
uv run dense workspace init --id binding_sites_trial --from-workspace demo_binding_sites --copy-inputs --output-mode local
cd src/dnadesign/densegen/workspaces/binding_sites_trial
uv run dense validate-config --probe-solver
uv run dense run --fresh
uv run dense inspect run --library --events
```

### Quick start (canonical PWM)

The canonical demo requires MEME Suite (`fimo` on `PATH`) in addition to solver availability.

```bash
uv sync --locked
pixi install
pixi run fimo --version

uv run dense workspace init --id meme_three_tfs_trial --from-workspace demo_meme_three_tfs --copy-inputs --output-mode usr
cd src/dnadesign/densegen/workspaces/meme_three_tfs_trial
uv run dense validate-config --probe-solver
uv run dense stage-a build-pool --fresh
uv run dense run --fresh --no-plot
uv run dense inspect run --library --events
```

To generate additional sequences later, increase quotas in `config.yaml` and resume with:

```bash
uv run dense run --resume --no-plot
```

---

### Documentation map

Guides:

- [docs/guide/workspace.md](docs/guide/workspace.md)
- [docs/guide/inputs.md](docs/guide/inputs.md)
- [docs/guide/sampling.md](docs/guide/sampling.md)
- [docs/guide/generation.md](docs/guide/generation.md)
- [docs/guide/outputs-metadata.md](docs/guide/outputs-metadata.md)
- [docs/guide/postprocess.md](docs/guide/postprocess.md)

Reference:

- [docs/reference/cli.md](docs/reference/cli.md)
- [docs/reference/config.md](docs/reference/config.md)
- [docs/reference/outputs.md](docs/reference/outputs.md)
- [docs/reference/motif_artifacts.md](docs/reference/motif_artifacts.md)

Workflows:

- [docs/workflows/cruncher_pwm_pipeline.md](docs/workflows/cruncher_pwm_pipeline.md)
- [docs/workflows/usr_notify_hpc.md](docs/workflows/usr_notify_hpc.md)

---

@e-south
