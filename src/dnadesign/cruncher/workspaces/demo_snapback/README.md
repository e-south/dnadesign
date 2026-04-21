# Snapback Workspace

Scaffolded by `cruncher snapback init-workspace`.

Included files:
- `configs/runbook.yaml`
- `configs/snapback/demo_teto_bpu10i_cap.snapback.yaml`
- `configs/snapback/demo_teto_catalog_scan.snapback.solve.yaml`
- `inputs/nickases/local.nickases.yaml`
- `runbook.md`

Canonical single-command refresh:
- `uv run cruncher workspaces run --workspace demo_snapback --runbook configs/runbook.yaml`

Suggested next steps:
1. `uv run cruncher snapback validate --spec configs/snapback/demo_teto_bpu10i_cap.snapback.yaml`
2. `uv run cruncher snapback design --spec configs/snapback/demo_teto_bpu10i_cap.snapback.yaml`
3. `uv run cruncher snapback solve --spec configs/snapback/demo_teto_catalog_scan.snapback.solve.yaml`

Canonical refresh:
1. `uv run cruncher snapback design --spec configs/snapback/demo_teto_bpu10i_cap.snapback.yaml --force-overwrite`
2. `uv run cruncher snapback solve --spec configs/snapback/demo_teto_catalog_scan.snapback.solve.yaml --force-overwrite`

Workspace-scoped output roots:
- explicit design bundle under `outputs/design/`
- solve summary bundle under `outputs/solve/`
- materialized top hits under `outputs/solve/analysis/materialized_hits/hit_<rank>/`

Snapback invariants in this scaffold:
- solve uses co-design by default across the resolved nickase catalog
- the solve scaffold resolves built-in `neb_nicking_v1` plus `thermo_nicking_v1`
- the local `Nt.Bpu10I` overlay remains the explicit design example only
- omitted solve boundary and retained-length windows resolve to compact-first defaults
- retained homology starts exactly at the resolved nick boundary
- the effective cap loop is fixed at 3 nt
- pre-nick and exposed visuals use the nick as the single snapback origin boundary

Design bundles emit a three-state QA triptych:
- producer-owned QA JSON views under `analysis/views/`
- shared `snapback_visual_v1` contracts under `analysis/views/`
- one composite JSONL triptych contract under `analysis/views/snapback_triptych.snapback_visual.v1.jsonl`
- one BaseRender job under `baserender_jobs/snapback_triptych.job.yaml`
- one rendered `png|svg|pdf` triptych under `plots/` after `uv run baserender job run ...`
