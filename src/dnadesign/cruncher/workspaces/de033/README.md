# de033 Released-Product Workspace

Operational released-product Snapback workspace for the dual-enzyme `0/3/3` lane under the default no-frequent-cutter policy.

Included files:
- `configs/runbook.yaml`
- `runbook.md`

Single-command refresh:
- `uv run cruncher workspaces run --workspace de033 --runbook configs/runbook.yaml`

Suggested next steps:
1. `uv run cruncher snapback released-target-search --workspace-root src/dnadesign/cruncher/workspaces/de033 --nick-preset neb_nicking_v1 --nick-additional-preset thermo_nicking_v1 --release-preset type_iis_release_v1 --nick-boundary 0 --paired-bp 3 --cap-nt 3 --json`
2. `uv run cruncher snapback released-solve --workspace-root src/dnadesign/cruncher/workspaces/de033 --nick-preset neb_nicking_v1 --nick-additional-preset thermo_nicking_v1 --release-preset type_iis_release_v1 --nick-boundary 0 --paired-bp 3 --cap-nt 3 --run-dir outputs/released_solve --materialize-top-k 8 --render-format pdf --emit-renders --force-overwrite --json`

Workspace-scoped output roots:
- released-product solve bundle under `outputs/released_solve/`
- optional invalid explicit audit surface under `outputs/released_design/`

Operational invariants in this workspace:
- target-search uses explicit built-in nickase and release-enzyme presets and resolves the whole local nickase preset surface as `neb_nicking_v1 + thermo_nicking_v1`
- default operational policy excludes nickases carrying `FREQUENT_CUTTER`; `Nt.CviPII` is intentionally suppressed here
- released-solve exhausts the resolved allowed nickase x release-enzyme placement space before ranking and materializing hits
- redundant exact hits are collapsed to one representative per exposed post-nick `stem + cap` geometry before ranking and materialization
- internal-cut nickases whose recognition span overlaps the active strand origin are rejected; this prevents false positives such as `Nt.Bpu10I` from leaking into the exact frontier
- any release-site geometry that would begin left of logical origin `0` is rejected, and nickase geometry may extend left of origin only when the omitted leading prefix is one contiguous fully degenerate `N` block after top-strand normalization; the workspace does not permit negative local coordinates in search, solve, or plot outputs
- current built-in allowed posture is `near_hits_only`; the current near frontier is `Nt.Bpu10I + BsaI-HFv2` at boundary `2` and `Nt.BsmAI + BspQI` at boundary `6`
- `configs/snapback/de033.released.snapback.yaml` is retained only as an invalid explicit audit fixture for the degenerate-prefix-aware nonnegative-origin contract; `released-design` and `released-show` should report `invalid_precursor`, not a green operational bundle
- final geometry is evaluated on the exposed post-release bottom strand
- the effective cap loop is fixed at `3 nt`
- materialized hit bundles stay under `outputs/released_solve/analysis/materialized_hits/` and publish one triptych plot per hit when `--emit-renders` is enabled
