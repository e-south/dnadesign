## Retron Hairpin Design Effort Routes

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-09

Use this page after the tracked study status answers `where are we?`.
Use preflight when you need blocker or command-readiness answers.
This page keeps the study-owned handoff map in one place.

### Quick route

- Snapshot:
  `uv run ops progress show cruncher.data-plane.cruncher-study-status --study-dir docs/studies/retron_hairpin_design --json`
- Preflight:
  `uv run ops progress show cruncher.data-plane.cruncher-study-preflight --study-dir docs/studies/retron_hairpin_design --scope next --json`
- Repo-local study shortcut:
  `.agents/skills/retron-hairpin-study/SKILL.md`
- Pair with:
  `harness-engineering` for study-surface hardening and
  `code-change-discipline` in the `pragmatic-programming-principles` lane for
  boundary or contract changes.

### Cold-start contract

- If the request is explicitly about `retron_hairpin_design`, pin the
  study with the two `cruncher-study-*` commands above even when
  `docs/studies/index.yaml` names another repo-wide active study.
- After study status or preflight answers the state or blocker question, stay
  on this page for the ordered post-probe handoff.
- Open `pipeline.yaml` only when the task needs machine-readable command-group
  or automation bootstrap metadata.
- Open `ops.study.yaml` only when the task needs lifecycle or preflight
  declarations.

### Boundary shorthand

- `released-product Snapback` means the BspQI-pinned dual-enzyme precursor lane where final geometry is evaluated on retained active top and bottom products and rebased so the nick boundary is origin `0` in final-geometry space.
- `preserved-site Snapback` means the older one-enzyme lane and stays a separate contract.
- `scar-nick` means the base-junction route for Type IIS retained scars plus
  terminal nick processing through the `scar_nick` subpackage. It is about
  which four-base basal scars can survive the nick-disposal process, not about
  direct phenotype prediction.
- `YIU` means mismatch-centric payload rendering over a fixed 4 nt internal window; it is not the shortening topology engine here.
- `retron context` means biological framing from the checked-in audit notes, not scoring hooks or implicit solver relaxations.

### Ordered post-probe handoff

1. Recover state with the pinned snapshot command above, or use the pinned
   preflight command above when the question is blocker or execution readiness.
2. Stay on the primary route below for the read-only released-product probe in
   `de033` and inspect allowed exact-hit versus bounded near-hit posture against
   the real release-enzyme catalog with the default Type IIS release enzyme
   pinned to `BspQI`.
3. After the read-only probe is clean, materialize the whole-catalog released
   solve bundle so ranked BspQI-pinned hits and per-hit plots are published under
   `outputs/released_solve`. The solve surface now collapses redundant exact or
   near hits to one representative per exposed post-nick `stem + cap` geometry.
4. Treat `released-design` and `released-show` as validation-only for the
   checked-in invalid fixture.
   Treat `released-design` and `released-show` as an optional audit path only.
   The checked-in downstream-`BspQI` spec under
   `configs/snapback/de033.released.snapback.yaml` is expected to report
   `invalid_precursor` under the degenerate-prefix-aware nonnegative-origin
   contract.
5. Use the YIU contrast route below only when the task is boundary auditing or
   contrast rendering, not when the task is shortening design.
6. Use the MSD-HOPV5 visual route only for an explicit prior-design
   comparison. It is a visual-only sibling workspace, not a `de033` solve hit.
7. Use the scar-nick base-junction route below when the task is profile-diverse
   `S0=M` scar feasibility, top/bottom nick flexibility, strict terminal nick
   policy, or `scar_nick` schema evolution.

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
  `cd src/dnadesign/cruncher/workspaces/de033 && uv run cruncher snapback released-target-search --workspace-root . --nick-preset neb_nicking_v1 --nick-additional-preset thermo_nicking_v1 --release-preset type_iis_release_v1 --release-variant-id BspQI --nick-boundary 0 --paired-bp 3 --cap-nt 3 --allow-top-active-routes --allow-precut-footprint-outside-active-product --json`
- Follow-up mutating commands:
  `cd src/dnadesign/cruncher/workspaces/de033 && uv run cruncher snapback released-solve --workspace-root . --nick-preset neb_nicking_v1 --nick-additional-preset thermo_nicking_v1 --release-preset type_iis_release_v1 --release-variant-id BspQI --nick-boundary 0 --paired-bp 3 --cap-nt 3 --allow-top-active-routes --allow-precut-footprint-outside-active-product --run-dir outputs/released_solve --materialize-top-k 16 --render-format pdf --emit-renders --force-overwrite --json`
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
  bundle lane. Near-hit ranking and plots include retained duplex left of the
  nick in `effective_stem_bp`; boundary-`2` / paired-`3` is therefore rendered
  and reported as a 5 bp effective stem.

### Visual-only route: MSD-HOPV5 comparison

Use this route when the task is to show the prior explicit `Nt.Bpu10I` MSD-HOPV5 example
beside current solve outputs without mixing generated artifacts.

- Type: `route`
- Plane: `data-plane`
- Surface role: `comparison-visual`
- Owner-boundary: `cruncher`
- Current state: `ready`
- Workspace: `src/dnadesign/cruncher/workspaces/msd-HOPV5_snapback`
- Follow-up mutating command:
  `cd src/dnadesign/cruncher/workspaces/msd-HOPV5_snapback && uv run cruncher snapback visual --spec configs/snapback/msd-HOPV5.visual.snapback.yaml --force-overwrite --json`
- Bundle root:
  `src/dnadesign/cruncher/workspaces/msd-HOPV5_snapback/outputs/msd-HOPV5_visual`
- Route note:
  this route validates the explicit precursor, nick boundary, stem, cap, and
  foldback decomposition before rendering. It does not run catalog search and
  does not overwrite `de033`.

### Context route: scar-nick base-junction

Use this route when the task is base-junction scar feasibility, B26/B43 profile
calibration, profile-diverse `S0=M` scar analogs, top-versus-bottom nick
flexibility, or schema work for the nick-disposal model.

- Type: `context`
- Plane: `data-plane`
- Surface role: `base-junction-context`
- Owner-boundary: `cruncher`
- Current state: `context-ready`
- Workspace: `src/dnadesign/cruncher/workspaces/scar_nick_teto`
- Primary note:
  `docs/studies/retron_hairpin_design/scar-nick-base-junction.md`
- Tool-owned detail:
  `src/dnadesign/cruncher/src/scar_nick/`
- Route note:
  the strict study policy is exact terminal nick, top or bottom strand allowed,
  zero protected bases downstream of the nick, downstream degenerate `N` only,
  and `S0=M` for ligation. The current context records that exact provided L/R
  sequences cover `0/9`, unique provided profile classes cover `6/7`, and
  provided constructs by profile analog cover `8/9`. BbsI-HF gives `10/256`
  retained scars, PaqCI gives `14/256`, and BsaI-HFv2 gives `0/256`.
  The study target is `scar_nick`-feasible profile coverage across `S3/S2/S1`
  with `S0=M`, no middle-middle hard `S2/S1` double mismatch, and single-hard,
  `X+W`, W-only, W+W, or S3-edge double-hard profiles such as `MXMM`,
  `WXMM`, `XWMM`, `MWXM`, `MXWM`, `XMWM`, `WMMM`, `MWMM`, `MMWM`, `WWMM`,
  `WMWM`, `MWWM`, `XXMM`, and `XMXM`;
  exact B26 sequence preservation is calibration context, not the selection
  objective. Current `scar_nick` outputs treat `nicked_strand`,
  `surviving_strand`, retained scar source, and profile-bucket coverage as
  first-class schema/ranking fields for the checked-in BbsI-HF route. Use
  `export/table__scar_nick_candidate_pair_calls.csv` as the flat left/right
  pair-call handoff table, and rerun the route before making PaqCI-specific
  capacity claims.

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
  `docs/studies/retron_hairpin_design/status.md`
- Study command ladder:
  `docs/studies/retron_hairpin_design/pipeline.yaml` for machine-readable
  command groups and automation bootstrap support
- Study lifecycle and preflight contract:
  `docs/studies/retron_hairpin_design/ops.study.yaml`
- Scar-nick base-junction note:
  `docs/studies/retron_hairpin_design/scar-nick-base-junction.md`
- Consolidated retron/P4 and YIU executive summary:
  `src/dnadesign/cruncher/docs/dev/2026-04-19-retron-p4-hairpin-variant-audit.md`
- Snapback phenomenology dev spec:
  `docs/studies/retron_hairpin_design/snapback-phenomenology-dev-spec.md`
- Route note:
  `routes.md` is the canonical human handoff; the other notes are study
  context or machine-readable support, not replacement route maps.
