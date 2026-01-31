## DenseGen — Dense Array Generator

**DenseGen** packs transcription factor binding sites (TFBSs) into dense synthetic nucleic acid sequences, wrapping the ["dense-arrays"](https://github.com/e-south/dense-arrays) ILP solver. It has two sampling stages:

1. **Stage‑A sampling** — mine TFBSs from PWM artifacts (via [FIMO](https://meme-suite.org/meme/doc/fimo.html)) to build TFBS pools.
2. **Stage‑B sampling** — sample solver libraries from those pools.

DenseGen also plans sequence constraints/quotas, dense-array generation with run‑scoped, fail‑fast Parquet I/O and manifests for reproducibility, and produces audit reports plus plots (e.g., TF usage and coverage).

For a full walkthrough with expected outputs, see [DenseGen demo](docs/demo/demo_basic.md).

### Contents

1. [Quick start](#quick-start)
2. [More documentation](#more-documentation)

---

### Quick start (minimal)

Prerequisites include Python, dense-arrays, and a MILP solver. CBC is open-source; [GUROBI](https://www.gurobi.com/) is supported if installed and licensed. Stage‑A FIMO sampling requires MEME Suite (`fimo` on PATH; use `pixi run` if needed).

```bash
# 1) Enter the demo workspace (config.yaml is auto‑discovered).
cd src/dnadesign/densegen/workspaces/demo_meme_two_tf

# 2) Validate schema + solver availability.
dense validate-config --probe-solver

# 3) Stage‑A: materialize TFBS pools (required before `dense run` unless you pass --rebuild-stage-a).
dense stage-a build-pool --fresh

# 4) Run generation (use --resume or --fresh if outputs already exist).
dense run

# 5) Plot a minimal diagnostics set.
dense plot --only stage_a_summary,stage_b_summary,run_health
```

For a full walkthrough with inspection/reporting steps, see the demo guide.

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
