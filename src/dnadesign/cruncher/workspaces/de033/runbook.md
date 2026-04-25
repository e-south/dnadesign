## de033 Released-Product Runbook

**Workspace Path**
- src/dnadesign/cruncher/workspaces/de033/

**Purpose**
- Run the operational dual-enzyme `0/3/3` released-product lane where final geometry is judged on the exposed post-release bottom strand.
- Search the real built-in `neb_nicking_v1` plus `thermo_nicking_v1` nickase catalogs against the real built-in `type_iis_release_v1` release-enzyme catalog before materializing a bundle.
- Materialize and render the top whole-catalog released-product hits under `outputs/released_solve/analysis/materialized_hits/`, with redundant exact or near hits collapsed to one representative per exposed post-nick `stem + cap` geometry.
- Enforce the degenerate-prefix-aware nonnegative-origin contract: no release-site geometry may begin left of logical origin `0`, and a nickase may extend left of origin only when the omitted leading prefix is one contiguous fully degenerate `N` block after top-strand normalization. No rendered plot element may carry a negative coordinate.
- Exclude demo-only overlays by default so the study lane does not silently pass on toy catalogs.
- Exclude nickases carrying `FREQUENT_CUTTER` by default so `Nt.CviPII` is not treated as an operational solution.
- Treat `de033` as the operational search/solve workspace. The checked-in downstream-`BspQI` explicit spec is now an invalid audit fixture for the degenerate-prefix-aware nonnegative-origin contract, not a green bundle target.

**Run This Single Command**

    uv run cruncher workspaces run --workspace de033 --runbook configs/runbook.yaml

### Step-by-Step Commands

    set -euo pipefail
    cd src/dnadesign/cruncher/workspaces/de033
    cruncher() { uv run cruncher "$@"; }

    # Standard machine-runbook sequence (matches configs/runbook.yaml).
    cruncher snapback released-target-search --workspace-root . --nick-preset neb_nicking_v1 --nick-additional-preset thermo_nicking_v1 --release-preset type_iis_release_v1 --nick-boundary 0 --paired-bp 3 --cap-nt 3 --json
    cruncher snapback released-solve --workspace-root . --nick-preset neb_nicking_v1 --nick-additional-preset thermo_nicking_v1 --release-preset type_iis_release_v1 --nick-boundary 0 --paired-bp 3 --cap-nt 3 --run-dir outputs/released_solve --materialize-top-k 8 --render-format pdf --emit-renders --force-overwrite --json

### Optional follow-up commands

    open outputs/released_solve/analysis/materialized_hits/hit_01/plots/released_hit_triptych.pdf
    uv run cruncher snapback released-design --spec configs/snapback/de033.released.snapback.yaml --force-overwrite --json
    uv run cruncher snapback released-show --run outputs/released_design --json
    uv run cruncher snapback released-solve --workspace-root . --nick-preset neb_nicking_v1 --nick-additional-preset thermo_nicking_v1 --release-preset type_iis_release_v1 --nick-boundary 0 --paired-bp 3 --cap-nt 3 --run-dir outputs/released_solve --materialize-top-k 8 --render-format pdf --emit-renders --force-overwrite --allow-frequent-cutter-nickases --json
