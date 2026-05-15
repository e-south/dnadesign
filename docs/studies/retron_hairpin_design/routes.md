## Retron Hairpin Design Effort Routes

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-14

Retron MSD product work starts from the user's provided parts and desired
output, not from study phase. Status/preflight surfaces are only for explicit
progress or blocker questions.

### Quick route

- Compiler/product route:
  `Study route: MSD design references` below.
- Progress status route for explicit status/history questions only:
  `uv run ops progress show cruncher.data-plane.cruncher-study-status --study-dir docs/studies/retron_hairpin_design --json`
- Execution preflight route for explicit blocker/readiness questions only:
  `uv run ops progress show cruncher.data-plane.cruncher-study-preflight --study-dir docs/studies/retron_hairpin_design --scope next --json`
- Repo-local study shortcut:
  `.agents/skills/retron-hairpin-study/SKILL.md`
- Pair with:
  `harness-engineering` for study-surface hardening and
  `code-change-discipline` in the `pragmatic-programming-principles` lane for
  boundary or contract changes.

### Routing contract

- If the request supplies an MSD label or explicit parts, start from the
  compiler/product route. Do not run study status or preflight first.
- If the request is explicitly about `retron_hairpin_design` progress,
  history, or blockers, pin the study with the two `cruncher-study-*` commands
  above even when `docs/studies/index.yaml` names another repo-wide active
  study.
- After study status or preflight answers a progress or blocker question, stay
  on this page for primitive or compiler routing.
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

### Primitive route handoff

1. For complete MSD labels or complete explicit parts, use the study-owned MSD
   design-reference route below and compile directly.
2. For missing cap/shortening constraints, stay on the primary route below for
   the read-only released-product probe in
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
8. Use the study-owned MSD design-reference route when a lab-facing shorthand
   such as `pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM` needs to become a
   frozen `msd_design_reference_v1` or `msd_design_catalog_v1` record. Do not
   add a top-level `retron-msd` tool; invoke the module under studies.

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
- Workspace runbook:
  `src/dnadesign/cruncher/workspaces/scar_nick_teto/runbook.md`
- Primary note:
  `docs/studies/retron_hairpin_design/scar-nick-base-junction.md`
- Tool-owned detail:
  `src/dnadesign/cruncher/src/scar_nick/`
- Same-workspace configs:
  `configs/scar_nick/teto_upstream_processing.bbsI_hf.scar_nick.yaml`
  and
  `configs/scar_nick/teto_upstream_processing.paqci_core_panel.scar_nick.yaml`
- Follow-up mutating commands:
  `uv run cruncher scar-nick design --spec src/dnadesign/cruncher/workspaces/scar_nick_teto/configs/scar_nick/teto_upstream_processing.bbsI_hf.scar_nick.yaml --force-overwrite`
  `uv run cruncher scar-nick design --spec src/dnadesign/cruncher/workspaces/scar_nick_teto/configs/scar_nick/teto_upstream_processing.paqci_core_panel.scar_nick.yaml --force-overwrite`
  `uv run baserender job run src/dnadesign/cruncher/workspaces/scar_nick_teto/outputs/scar_nick/teto_upstream_processing_bbsI_hf/baserender_jobs/scar_nick_terminal_nick.job.yaml`
  `uv run baserender job run src/dnadesign/cruncher/workspaces/scar_nick_teto/outputs/scar_nick/teto_upstream_processing_paqci_core_panel/baserender_jobs/scar_nick_terminal_nick.job.yaml`
- Bundle roots:
  `src/dnadesign/cruncher/workspaces/scar_nick_teto/outputs/scar_nick/teto_upstream_processing_bbsI_hf`
  and
  `src/dnadesign/cruncher/workspaces/scar_nick_teto/outputs/scar_nick/teto_upstream_processing_paqci_core_panel`
- Route note:
  the strict study policy is exact terminal nick, top or bottom strand allowed,
  zero protected bases downstream of the nick, downstream degenerate `N` only,
  and `S0=M` for ligation. The same `scar_nick_teto` workspace can hold
  multiple release-enzyme specs and output run dirs; these are not independent
  workspaces. The current regenerated strict panel records BbsI-HF at `6/256`
  retained scars, PaqCI at `10/256`, and BsaI-HFv2 at `0/256`.
  The study target is `scar_nick`-feasible profile coverage across `S3/S2/S1`
  with `S0=M`, no middle-middle hard `S2/S1` double mismatch, and single-hard,
  `X+W`, W-only, W+W, or S3-edge double-hard profiles such as `MXMM`,
  `WXMM`, `XWMM`, `MWXM`, `MXWM`, `XMWM`, `WMMM`, `MWMM`, `MMWM`, `WWMM`,
  `WMWM`, `MWWM`, `XXMM`, and `XMXM`;
  exact B26 sequence preservation is calibration context, not the selection
  objective. Current BbsI-HF plus PaqCI outputs cover 13 of those 14 active
  buckets; `WMWM` remains uncovered under the current strict catalog policy.
  Current `scar_nick` outputs treat `nicked_strand`,
  `surviving_strand`, retained scar source, and profile-bucket coverage as
  first-class schema/ranking fields for the checked-in BbsI-HF route. Use
  `export/table__scar_nick_candidate_pair_calls.csv` as the flat left/right
  pair-call handoff table, and rerun the route before making PaqCI-specific
  capacity claims.

### Study route: MSD design references

Use this route when the task starts from a lab-facing construct label and needs
a frozen design reference, one MSD sequence unit, or Reader joins.

- Type: `study-contract`
- Plane: `data-plane`
- Surface role: `record-plane design-reference-normalization`
- Owner-boundary: `studies/retron_hairpin_design`
- Current state: `compiler-ready`
- Registry:
  `docs/studies/retron_hairpin_design/msd_design_registry.yaml`
- Study-selected label list:
  `docs/studies/retron_hairpin_design/msd_design_hit_labels.txt`
- Public module:
  `src/dnadesign/studies/retron_hairpin_design/cli.py`
- Typed compiler spec:
  `retron_msd_compiler_spec_v1` YAML/JSON accepted by `lint`, `compile`, and
  `materialize` through `--spec`
- Lint command:
  `uv run python -m dnadesign.studies.retron_hairpin_design.cli lint --id "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM"`
- Spec lint command:
  `uv run python -m dnadesign.studies.retron_hairpin_design.cli lint --spec path/to/retron_msd_compiler_spec.yaml --format json`
- Compile command:
  `uv run python -m dnadesign.studies.retron_hairpin_design.cli compile --input docs/studies/retron_hairpin_design/msd_design_hit_labels.txt --out-dir /tmp/dnadesign_retron_msd_design_references --format json`
- Materialize command:
  `uv run python -m dnadesign.studies.retron_hairpin_design.cli materialize --input docs/studies/retron_hairpin_design/msd_design_hit_labels.txt --out-dir /tmp/dnadesign_retron_msd_sequences --payload-sequence TetR=<payload-sequence> --cap-sequence C26=<cap-sequence> --cap-sequence C172=<cap-sequence> --render-format png --format json`
- Route note:
  this compiler is intentionally study-owned and is not registered as a top-level `uv run retron-msd` tool. It parses lab-facing labels or typed
  `retron_msd_compiler_spec_v1` design parts into the same trusted structure:
  `construct_id`, payload/target, cap id, left/right scar-nick bases, and
  optional profile code; then it recomputes the `S3/S2/S1/S0` profile and
  fails fast if the provided code drifts or `S0` is not ligatable. Compiler
  specs may point at solved Snapback cap primitives or scar-nick stem-base
  primitives only through public `dnadesign.cruncher.snapback` and
  `dnadesign.cruncher.scar_nick` APIs. `selector.mode=rank` is the preferred
  explicit combination surface; rank lists, ranges, and all-hit selectors must
  fail instead of silently running combinatorics until an expansion contract is
  deliberately added. Registry metadata stores route notes, nickase, and nick
  orientation when known. The emitted
  `msd_design_catalog_v1` is the Reader-facing bridge; Reader should not parse Construct, Folding, BaseRender, or Cruncher internals. Ad hoc compiles should
  write to explicit transient directories such as `/tmp/dnadesign_retron_msd_*`;
  Reader-linked runs should snapshot the same shallow bundle into the owning
  Reader experiment `inputs/designs/` directory: `README.md`, `manifest.json`,
  `msd_design_catalog_v1.json`, `reference_index.tsv`, and flat
  `references/*.msd_design_reference_v1.json` files. Do not add per-design
  Construct/Folding workspaces for this path. The materialized sequence is one
  MSD unit per design: 5' flank plus left base, payload primary, cap geometry,
  payload complement, right base plus 3' flank. The CLI does not expose a
  repeat-count flag, so it cannot chain complete MSD units together. The
  sequence bundle keeps the top level to `README.md`, `manifest/`, and
  `variants/`. Catalogs, `manifest/sequence_manifest.json`,
  `manifest/sequence_index.tsv`, generated composition configs, and provenance
  live under `manifest/`; each `variants/<msd_design_id>/` directory groups
  forward/reverse-complement GenBank and FASTA under `sequences/`,
  `secondary_structure.native.png` plus `composition_overview.svg` under
  `plots/`, curated metadata under `manifest/`, and raw producer output under
  `runtime/construct/`. Visible GenBank/CSV labels should be display labels
  such as `msd[teto]`, `Cap`, `Left Base`, and `Right Base`; raw ids remain
  machine metadata, not operator-facing labels.

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
- Linear ssDNA composition handoff:
  `docs/studies/retron_hairpin_design/linear-ssdna-composition.md`
- Generic linear ssDNA composition dev spec:
  `docs/dev/plans/2026-05-13-generic-linear-ssdna-composition-spec.md`
- Generic linear ssDNA composition execution plan:
  `docs/exec-plans/completed/2026-05-13-generic-linear-ssdna-composition.md`
- Construct linear ssDNA dogfood config:
  `src/dnadesign/construct/workspaces/retron43_teto_manual_x8/config.composition.yaml`
- Generated Construct/BaseRender dogfood bundle:
  `src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8`
  after `uv run construct compose run --config ...`; generated contents include
  `visual/sequence_evidence_map_v1.json`,
  `baserender_jobs/component_span_qa_svg.yaml`,
  `folding/secondary_structure_prediction_request_v1.yaml`,
  `folding/folding_preflight.json`,
  `folding/secondary_structure_prediction_v1.json`,
  `visual/viennarna_secondary_structure/viennarna_secondary_structure_svg_v1.json`,
  `visual/viennarna_secondary_structure/secondary_structure.annotated.svg`,
  and the BaseRender SVG under `visual/renders/component_span_qa_svg/` after
  the BaseRender job runs.
- Folding dogfood commands:
  `uv run folding preflight --request src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8/folding/secondary_structure_prediction_request_v1.yaml`
  `uv run folding run --request src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8/folding/secondary_structure_prediction_request_v1.yaml`
  `uv run folding plot --prediction src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8/folding/secondary_structure_prediction_v1.json --assembled-sequence src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8/folding/secondary_structure_input_sequence.json --visual-contract src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8/visual/sequence_evidence_map_v1.json --output-dir ../visual/viennarna_secondary_structure --layout naview`
  Local status is `ok` through the uv-managed ViennaRNA Python API
  (`backend.interface: python_api`, `python_module: RNA`). The external
  `RNAfold` executable is still optional and not on the local PATH.
- ViennaRNA-native plot dogfood:
  the Construct run writes a native structure SVG, an annotated SVG, and an
  annotation manifest under `visual/viennarna_secondary_structure/`. Current
  dogfood QA records the canonical 88 nt component-unit prediction, 28 parsed
  base pairs, no cross-copy pairings, and one recovered declared payload
  pairing. BaseRender remains the separate linear component-span renderer.
- Route note:
  `routes.md` is the canonical human handoff; the other notes are study
  context or machine-readable support, not replacement route maps.
