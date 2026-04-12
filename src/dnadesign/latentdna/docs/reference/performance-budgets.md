# Performance Budgets

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-11

The package currently treats promoter-study scale as a normal operating target, but the checked-in benchmark harness is still fixture-scale smoke coverage.

Current benchmark harness:

- `tests/perf/test_benchmark_harness.py`

Registered smoke slices:

- `bench_view_materialize`
- `bench_delta_build`
- `bench_neighbors_fit`
- `bench_projection_fit`
- `bench_distance_score`
- `bench_export_x2`
- `bench_deliverable_atlas_2x2`

Each smoke record emits:

- wall time
- throughput
- peak RSS
- artifact size
- correctness summary

Interpretation:

- these fixture-scale benchmarks are regression guards for the harness contract and artifact reuse paths
- they are not substitutes for live promoter-study pressure runs on the real USR planes
- real-study pressure evidence should continue to be recorded in the development journal

See also:

- [../dev/journal.md](../dev/journal.md)
- [../workflows/promoter-study-latent-atlas.md](../workflows/promoter-study-latent-atlas.md)
