# DenseGen — Dense Array Generator

DenseGen designs compact synthetic promoters by packing transcription factor binding sites (TFBSs)
into fixed-length sequences. It wraps the dense-arrays optimizer and adds strict sampling,
constraints, batching, outputs, and plotting.

DenseGen is decoupled from USR: USR is an optional input/output adapter. Parquet-only workflows
are fully supported and do not require USR modules.

## Quick start

Use the provided smoke run (small, Parquet-only):

```bash
uv run dense validate -c src/dnadesign/densegen/runs/smoke_v2/config.yaml
uv run dense describe -c src/dnadesign/densegen/runs/smoke_v2/config.yaml
uv run dense run -c src/dnadesign/densegen/runs/smoke_v2/config.yaml --no-plot
uv run dense plot -c src/dnadesign/densegen/runs/smoke_v2/config.yaml --only tf_usage,tf_coverage
```

Runs are job-scoped: outputs/logs/plots live inside the run directory.
Source code lives in `src/dnadesign/densegen/src/`.

## Documentation

DenseGen docs live in `src/dnadesign/densegen/docs/README.md`. Start here:

- CLI: `src/dnadesign/densegen/docs/cli.md`
- Architecture: `src/dnadesign/densegen/docs/architecture.md`
- Guide: `src/dnadesign/densegen/docs/guide/README.md`
- Usage demo: `src/dnadesign/densegen/docs/guide/demo.md`
- Reference: `src/dnadesign/densegen/docs/reference/config.md`, `src/dnadesign/densegen/docs/reference/outputs.md`
- Developer spec: `src/dnadesign/densegen/docs/dev/developer-spec.md`
- Roadmap: `src/dnadesign/densegen/docs/roadmap/OVERHAUL_PLAN.md`
