## DenseGen — Dense Array Generator

**DenseGen** packs transcription factor binding sites (TFBSs) into compact, synthetic nucleotide sequences. It wraps the [`dense-arrays`](https://github.com/e-south/dense-arrays) ILP optimizer and adds library sampling, constraint planning, run-scoped IO, and plotting so you can generate, audit, and reproduce dense arrays end-to-end.


### Contents

1. [Quick start](#quick-start)
2. [More documentation](#more-documentation)

---

### Quick start

Prerequisites include Python, dense-arrays, and a MILP solver. CBC is open-source; [GUROBI](https://www.gurobi.com/) is supported if installed and licensed.

Use the canonical demo config (small, Parquet-only). The demo uses MEME motif files
copied from the Cruncher basic demo workspace (`inputs/local_motifs`) and parsed with
Cruncher’s MEME parser for DRY, consistent parsing.
FIMO-backed PWM sampling is supported when MEME Suite is available (`fimo` on PATH via `pixi run`).
Stratified FIMO sampling uses canonical p‑value bins by default; see the guide for mining workflows.

```bash
pixi run dense workspace init --id demo --root /tmp --template src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml --copy-inputs
CFG=/tmp/demo/config.yaml

pixi run dense validate-config -c "$CFG" --probe-solver
pixi run dense inspect inputs -c "$CFG"
pixi run dense stage-a build-pool -c "$CFG"
pixi run dense stage-b build-libraries -c "$CFG"
pixi run dense run -c "$CFG" --no-plot
pixi run dense inspect run --run /tmp/demo --library --top-per-tf 5
pixi run dense report -c "$CFG" --format md
pixi run dense plot -c "$CFG" --only tf_usage,tf_coverage
```

For a full end-to-end walkthrough with expected outputs, see
[DenseGen demo](docs/demo/demo_basic.md).

---

### More documentation

Docs live in `docs/`:
- [DenseGen demo](docs/demo/demo_basic.md) - canonical end-to-end walkthrough.
- [DenseGen guide](docs/guide/index.md) - concepts and data flow.
- [Reference](docs/reference/cli.md) - CLI, config, outputs (schema-level detail).

---

@e-south
