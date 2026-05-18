---
doc_id: study-retron-hairpin-design-route-product-released-product-snapback
surface: study-route-detail
study_id: retron_hairpin_design
owner: dnadesign-maintainers
last_verified: 2026-05-18
parent_route: ../README.md
type: route
plane: data-plane
owner_boundary: cruncher/snapback
surface_role: primitive-owner
current_state: in_progress
entry_artifact: cap-or-shortening-geometry-request
exit_artifact: snapback_released_product_primitives
---

## Released-Product Snapback Route

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Use this route when the task is actual shortening construction or evaluation.
This is the cap/shortening primitive owner, not the Retron MSD compiler.

### Route Contract

- Type: `route`
- Plane: `data-plane`
- Surface role: `primitive-owner`
- Owner-boundary: `cruncher/snapback`
- Current state: `in_progress`
- Entry artifact: cap or shortening geometry request
- Exit artifact: released-product Snapback hit bundle
- Workspace: `src/dnadesign/cruncher/workspaces/de033`
- Primary doc: `src/dnadesign/cruncher/docs/guides/snapback_released_workflow.md`

### Commands

First read-only probe:

```bash
cd src/dnadesign/cruncher/workspaces/de033
uv run cruncher snapback released-target-search \
  --workspace-root . \
  --nick-preset neb_nicking_v1 \
  --nick-additional-preset thermo_nicking_v1 \
  --release-preset type_iis_release_v1 \
  --release-variant-id BspQI \
  --nick-boundary 0 \
  --paired-bp 3 \
  --cap-nt 3 \
  --allow-top-active-routes \
  --allow-precut-footprint-outside-active-product \
  --json
```

Follow-up mutating solve:

```bash
cd src/dnadesign/cruncher/workspaces/de033
uv run cruncher snapback released-solve \
  --workspace-root . \
  --nick-preset neb_nicking_v1 \
  --nick-additional-preset thermo_nicking_v1 \
  --release-preset type_iis_release_v1 \
  --release-variant-id BspQI \
  --nick-boundary 0 \
  --paired-bp 3 \
  --cap-nt 3 \
  --allow-top-active-routes \
  --allow-precut-footprint-outside-active-product \
  --run-dir outputs/released_solve \
  --materialize-top-k 16 \
  --render-format pdf \
  --emit-renders \
  --force-overwrite \
  --json
```

### Deliverables

- Bundle root: `src/dnadesign/cruncher/workspaces/de033/outputs/released_solve`
- Solve report: `analysis/solve_report.json`
- Hit table: `export/table__hits.csv`
- Per-hit bundles: `analysis/materialized_hits/hit_<rank>/`
- Per-hit plot: `plots/released_hit_triptych.pdf`
- Study cap source lookup: `docs/studies/retron_hairpin_design/compiler/catalog/msd_cap_sources.yaml`

### Notes

The active lane uses `neb_nicking_v1 + thermo_nicking_v1`, excludes
`FREQUENT_CUTTER` nickases by default, pins the Type IIS release enzyme to
`BspQI`, and evaluates retained active top and bottom products. No release-site
geometry may begin left of logical origin `0`; nickase geometry may extend left
of origin only when the omitted prefix is one contiguous fully degenerate `N` block
in the oriented top-strand view.

The study-owned cap lookup records `C26=AGGC`, `C43=tCCTCAGcccGCTGAGGa`, and
the selected `C172-C176` de033 source labels in
`compiler/catalog/msd_cap_sources.yaml`. These are checked-in entries, not a
general rule that any future `C###` id should be resolved from this workspace.

`released-design` and `released-show` remain optional audit paths for the
checked-in invalid fixture. The downstream-`BspQI` spec under
`configs/snapback/de033.released.snapback.yaml` is expected to report
`invalid_precursor`.

For the prior explicit `Nt.Bpu10I` MSD-HOPV5 visual comparison, use
`src/dnadesign/cruncher/workspaces/msd-HOPV5_snapback`. It is a visual-only
sibling workspace and not a `de033` solve hit.
