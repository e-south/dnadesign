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

```bash
uv run dense validate -c src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml
uv run dense describe -c src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml
uv run dense run -c src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml --no-plot
uv run dense plot -c src/dnadesign/densegen/workspaces/demo_meme_two_tf/config.yaml --only tf_usage,tf_coverage
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
