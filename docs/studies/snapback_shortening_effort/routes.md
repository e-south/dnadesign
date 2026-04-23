## Snapback Shortening Effort Routes

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-22

Use this page after the tracked study status answers `where are we?`.
Use preflight when you need blocker or command-readiness answers.
This page keeps the study-owned handoff map in one place.

### Quick route

- Snapshot:
  `uv run ops progress show cruncher.data-plane.cruncher-study-status --study-dir docs/studies/snapback_shortening_effort --json`
- Preflight:
  `uv run ops progress show cruncher.data-plane.cruncher-study-preflight --study-dir docs/studies/snapback_shortening_effort --scope next --json`
- Repo-local study shortcut:
  `.agents/skills/snapback-hairpin-study/SKILL.md`
- Pair with:
  `harness-engineering` for study-surface hardening and
  `pragmatic-programming-principles` for boundary or contract changes.

### Cold-start contract

- If the request is explicitly about `snapback_shortening_effort`, pin the
  study with the two `cruncher-study-*` commands above even when
  `docs/studies/index.yaml` names another repo-wide active study.
- After study status or preflight answers the state or blocker question, stay
  on this page for the ordered post-probe handoff.
- Open `pipeline.yaml` only when the task needs machine-readable command-group
  or native-agent bootstrap metadata.
- Open `ops.study.yaml` only when the task needs lifecycle or preflight
  declarations.

### Boundary shorthand

- `released-product Snapback` means the dual-enzyme precursor lane where final geometry is evaluated on the exposed post-release bottom strand while the nick stays anchored at origin.
- `preserved-site Snapback` means the older one-enzyme lane and stays a separate contract.
- `YIU` means mismatch-centric payload rendering over a fixed 4 nt internal window; it is not the shortening topology engine here.
- `retron context` means biological framing from the checked-in audit notes, not scoring hooks or implicit solver relaxations.

### Ordered post-probe handoff

1. Recover state with the pinned snapshot command above, or use the pinned
   preflight command above when the question is blocker or execution readiness.
2. Stay on the primary route below for the read-only released-product probe in
   `de033` and inspect allowed exact-hit versus bounded near-hit posture against
   the real release-enzyme catalog.
3. After the read-only probe is clean, materialize the whole-catalog released
   solve bundle so ranked hits and per-hit plots are published under
   `outputs/released_solve`. The solve surface now collapses redundant exact or
   near hits to one representative per exposed post-nick `stem + cap` geometry.
4. Treat `released-design` and `released-show` as an optional audit path only.
   The checked-in downstream-`BspQI` spec under
   `configs/snapback/de033.released.snapback.yaml` is expected to report
   `invalid_precursor` under the degenerate-prefix-aware nonnegative-origin
   contract.
5. Use the YIU contrast route below only when the task is boundary auditing or
   contrast rendering, not when the task is shortening design.

### Primary route: released-product Snapback

Use this route when the task is actual shortening construction or evaluation.
This is the active study lane.

- Type: `route`
- Plane: `data-plane`
- Surface role: `primary-execution`
- Owner-boundary: `cruncher`
- Current state: `in_progress`
- Workspace: `src/dnadesign/cruncher/workspaces/de033`
- Primary doc:
  `src/dnadesign/cruncher/docs/guides/snapback_released_workflow.md`
- First read-only command:
  `cd src/dnadesign/cruncher/workspaces/de033 && uv run cruncher snapback released-target-search --workspace-root . --nick-preset neb_nicking_v1 --nick-additional-preset thermo_nicking_v1 --release-preset type_iis_release_v1 --nick-boundary 0 --paired-bp 3 --cap-nt 3 --json`
- Follow-up mutating commands:
  `cd src/dnadesign/cruncher/workspaces/de033 && uv run cruncher snapback released-solve --workspace-root . --nick-preset neb_nicking_v1 --nick-additional-preset thermo_nicking_v1 --release-preset type_iis_release_v1 --nick-boundary 0 --paired-bp 3 --cap-nt 3 --run-dir outputs/released_solve --materialize-top-k 8 --render-format pdf --emit-renders --force-overwrite --json`
- Bundle root:
  `src/dnadesign/cruncher/workspaces/de033/outputs/released_solve`
- Solve deliverables:
  `analysis/solve_report.json`, `export/table__hits.csv`, and materialized
  bundles under `analysis/materialized_hits/hit_<rank>/` with one
  `plots/released_hit_triptych.pdf` per hit
- Route note:
  use this route for the actual shortening construction model. The active lane
  now has a whole-catalog solve surface with per-hit plots and resolves the
  local nickase presets as `neb_nicking_v1 + thermo_nicking_v1`. Default
  operational policy excludes `FREQUENT_CUTTER` nickases such as `Nt.CviPII`,
  no release-site geometry may begin left of logical origin `0`, and nickase
  geometry may extend left of origin only when the omitted prefix is one
  contiguous fully degenerate `N` block in the oriented top-strand view. `de033`
  currently operates as a bounded near-hit surface rather than an exact-hit
  bundle lane.

### Contrast route: YIU boundary check

Use this route only when you need a contrast surface for boundary language or a
reminder of what YIU does and does not model.

- Type: `route`
- Plane: `data-plane`
- Surface role: `contrast-check`
- Owner-boundary: `cruncher`
- Current state: `planned`
- Workspace: `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr`
- Primary docs:
  `src/dnadesign/cruncher/docs/guides/yiu_workflow.md`
  `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr/runbook.md`
- First read-only command:
  `cd src/dnadesign/cruncher/workspaces/demo_monotypic_tetr && uv run cruncher yiu validate --spec configs/yiu/tetr_teto2_wt_direct.yiu.yaml`
- Follow-up mutating commands:
  `cd src/dnadesign/cruncher/workspaces/demo_monotypic_tetr && uv run cruncher yiu render --spec configs/yiu/tetr_teto2_wt_direct.yiu.yaml --force-overwrite --emit-renders`
  `cd src/dnadesign/cruncher/workspaces/demo_monotypic_tetr && uv run cruncher yiu show --bundle outputs/plots/yiu__tetr_teto2_wt_direct`
- Route note:
  use this route only to keep the YIU boundary explicit and auditable.

### Context surfaces

- Study note:
  `docs/studies/snapback_shortening_effort/status.md`
- Study command ladder:
  `docs/studies/snapback_shortening_effort/pipeline.yaml` for machine-readable
  command groups and native-agent bootstrap support
- Study lifecycle and preflight contract:
  `docs/studies/snapback_shortening_effort/ops.study.yaml`
- Consolidated retron/P4 and YIU executive summary:
  `src/dnadesign/cruncher/docs/dev/2026-04-19-retron-p4-hairpin-variant-audit.md`
- Route note:
  `routes.md` is the canonical human handoff; the other notes are study
  context or machine-readable support, not replacement route maps.
