## DenseGen — Dense Array Generator

**DenseGen** packs transcription factor binding sites (TFBSs) into dense synthetic nucleic acid sequences, wrapping the ["dense-arrays"](https://github.com/e-south/dense-arrays) ILP solver with added features. It involves two distinct sampling stages:

1. **Stage‑A sampling (input sampling)** — mine TFBSs from PWM artifacts (via [FIMO](https://meme-suite.org/meme/doc/fimo.html)) to build TF‑aware TFBS pools. The goal is to retain a final TFBS library that
    - Stays near the PWM score frontier.
    - Is meaningfully diverse (primarily in PWM‑tolerant/ambiguous positions).
    - Allows variable TFBS lengths via flanks.

2. **Stage‑B sampling (library sampling)** — feed TFBS pools to the dense-array solver.

DenseGen also plans sequence constraints/quotas, dense-array generation with run‑scoped, fail‑fast Parquet I/O and manifests for reproducibility, and produces audit reports plus plots (e.g., TF usage and coverage).

For a full walkthrough with expected outputs, see [DenseGen demo](docs/demo/demo_basic.md).

### Contents

1. [Quick start](#quick-start)
2. [More documentation](#more-documentation)

---

### Quick start

Prerequisites include Python, dense-arrays, and a MILP solver. CBC is open-source; [GUROBI](https://www.gurobi.com/) is supported if installed and licensed. Stage‑A FIMO sampling requires MEME Suite (`fimo` on PATH; use `pixi run` if needed).

```bash
# 1) Enter the pre‑staged demo workspace (config.yaml is auto‑discovered).
cd src/dnadesign/densegen/workspaces/demo_meme_two_tf

# 2) Validate schema + solver availability before long runs.
dense validate-config --probe-solver

# 3) Inspect Stage‑A inputs and sampling settings.
dense inspect inputs

# 4) Inspect resolved outputs + Stage‑A/Stage‑B settings.
dense inspect config

# 5) Stage‑A: materialize TFBS pools (required before `dense run` unless you pass --rebuild-stage-a).
dense stage-a build-pool --fresh

# 6) Stage‑B: materialize solver libraries (optional, for inspection).
dense stage-b build-libraries

# 7) Run generation (use --resume or --fresh if outputs already exist).
dense run

# 8) Inspect run summary (library + events are optional add‑ons).
dense inspect run --library --events

# 9) Emit an audit report.
dense report --format md

# 10) List plots and render a subset.
dense ls-plots
dense plot --only placement_map,tfbs_usage,run_health
```

If you rerun a workspace that already has run outputs (e.g., `outputs/tables/attempts.parquet` or `outputs/meta/run_state.json`), choose `--resume` (continue) or `--fresh` (clear outputs and start over).

`dense run` will reuse existing Stage‑A pools by default; use `dense stage-a build-pool --fresh` or `dense run --rebuild-stage-a` when pools are missing or stale.

If you want to scaffold a new workspace from a packaged template, run `dense workspace init` from `src/dnadesign/densegen/workspaces/` to keep new workspaces colocated.

---

### More documentation

- [Demo walkthrough](docs/demo/demo_basic.md) - progressive end‑to‑end tour of all commands.
- [Guide: workspace layout](docs/guide/workspace.md) - workspace-first structure and rationale.
- [Guide: inputs (Stage‑A)](docs/guide/inputs.md) - input ingestion + Stage‑A sampling.
- [Guide: generation (Stage‑B)](docs/guide/generation.md) - Stage‑B sampling and constraints.
- [Guide: outputs + metadata](docs/guide/outputs-metadata.md) - what outputs mean and how to join them.
- [Guide: postprocess](docs/guide/postprocess.md) - gap‑fill policies.
- [Reference: CLI](docs/reference/cli.md) - operator manual (commands + flags).
- [Reference: config](docs/reference/config.md) - strict schema definition.
- [Reference: outputs](docs/reference/outputs.md) - output formats + manifests.
- [Reference: motif artifacts](docs/reference/motif_artifacts.md) - contract for PWM artifacts.
- [Dev: architecture](docs/dev/architecture.md) - pipeline + module map.
- [Workflow: Cruncher → DenseGen](docs/workflows/cruncher_pwm_pipeline.md) - artifact‑first handoff.

---

@e-south
