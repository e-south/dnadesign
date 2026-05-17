## Retron Hairpin Design Effort

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-14

### At a glance

- This study now routes Retron MSD product work as a genetic compiler:
  user-provided or study-selected parts become frozen
  `msd_design_reference_v1` / `msd_design_catalog_v1` records first.
- Complete labels or complete part sets should compile directly. Missing
  cap/shortening constraints route to released-product Snapback; missing
  basal left/right base, terminal-nick, or profile constraints route to
  scar-nick.
- Sequence artifact output is one MSD unit per design: 5' flank + left base,
  payload primary, snapback foldback geometry with a 3 nt `Cap` subsection,
  payload complement, right base + 3' flank.
- Construct, Folding, BaseRender, and ViennaRNA plotting are service handoffs
  after part selection. They should consume explicit files or producer bundles,
  not create one workspace per MSD ID. The compiler route does not expose a
  repeat-count flag.
- Released-product Snapback in `de033` remains the primitive owner for
  cap/shortening geometry.
- Scar-nick through the `scar_nick` subpackage remains the primitive owner for
  Type IIS retained scar space, terminal nick feasibility, B26/B43 calibration,
  and profile-diverse `S0=M` scar analogs.
- `YIU` stays in the record as a contrast check on boundary language. It is not the topology engine for this effort.
- The retron/P4 note stays in scope as framing evidence only. It motivates
  compact released products and disrupted basal-stem architecture, but it does
  not become Cruncher scoring logic.

### Quick route

- Compiler/product route:
  `uv run python -m dnadesign.studies.studies.retron_hairpin_design.cli compile --input docs/studies/retron_hairpin_design/msd_design_hit_labels.txt --out-dir /tmp/dnadesign_retron_msd_design_references --format json`
- GenBank/native-structure-PNG/review-PNG route after concrete subcomponents are available:
  `uv run python -m dnadesign.studies.studies.retron_hairpin_design.cli materialize --input docs/studies/retron_hairpin_design/msd_design_hit_labels.txt --out-dir /tmp/dnadesign_retron_msd_sequences --payload-sequence TetR=<payload-sequence> --cap-sequence C26=<cap-sequence> --cap-sequence C172=<cap-sequence> --render-format png --format json`
- Status route for explicit progress/history questions only:
  `uv run ops progress show studies.retron-hairpin-design.status --study-dir docs/studies/retron_hairpin_design --json`
- Preflight route for explicit blocker/readiness questions only:
  `uv run ops progress show studies.retron-hairpin-design.preflight --study-dir docs/studies/retron_hairpin_design --scope next --json`
- Repo-local study shortcut:
  `.agents/skills/retron-hairpin-study/SKILL.md`

### What is settled

- The primary product path is study-owned Retron MSD design-reference
  compilation, not a new generic top-level tool and not a workspace family.
- The compiler validates user-provided payload, cap, left base, right base,
  and optional profile code; it recomputes `S3/S2/S1/S0` and fails fast on
  profile drift or non-ligatable `S0`.
- The selected 177-194 scar-nick labels compile into one catalog from
  `docs/studies/retron_hairpin_design/msd_design_hit_labels.txt`.
- The released-product Snapback primitive remains available in `de033`.
- The tracked study default is the retained-active released-product policy,
  with retained top and bottom product routes evaluated.
- The Type IIS release enzyme is pinned to `BspQI` for the `de033`
  operational route; `BsaI-HFv2` is not part of the default route.
- Near-hit evaluation still treats retained duplex left of the nick as part of
  the effective folded stem, but the current BspQI-pinned retained-active
  screen reports exact origin-`0`, stem-`3`, cap-`3` hits.
- The operational catalog surface is `neb_nicking_v1 + thermo_nicking_v1`, with
  `FREQUENT_CUTTER` nickases excluded by default.
- The checked-in downstream-`BspQI` explicit spec remains a validation fixture and is
  expected to remain `invalid_precursor`.
- The validation fixture is expected to report `invalid_precursor` under the
  degenerate-prefix-aware nonnegative-origin rule because it does not provide a
  single contiguous fully degenerate `N` block.
- `YIU` stays contrast-only and does not absorb shortening topology semantics.
- The scar-nick strict policy is exact terminal nick, top or bottom nick
  allowed, zero protected bases downstream, downstream degenerate `N` only, and
  `S0=M` for ligation.
- Under that strict policy, exact supplied L/R pairs are not catalog-feasible
  for the current enzyme set, but profile analogs cover most of the desired
  match/mismatch classes.
- Current regenerated `scar_nick_teto` specs keep BbsI-HF and PaqCI in one
  workspace with separate output run dirs. BbsI-HF retains 6/256 strict scars;
  PaqCI retains 10/256 by adding `TTCA`, `TTCC`, `TTCG`, and `TTCT`;
  BsaI-HFv2 retains 0/256 under the same strict policy.
- Exact B26 `MXMX` remains a biological control architecture, but it is not
  scar-compatible under the `S0=M` ligation constraint.
- The scar-nick design target is now profile-diverse, `S0=M`,
  ligation-aware `scar_nick` coverage across `S3/S2/S1`, not exact B26
  sequence preservation or an `MXXM`-centered panel.
- Use `routes.md` for product routing, primitive handoffs, and deeper boundary
  notes.

### Compiler and primitive surfaces

- Compiler module:
  `src/dnadesign/studies/studies/retron_hairpin_design/`
- Compiler registry:
  `docs/studies/retron_hairpin_design/msd_design_registry.yaml`
- Study-selected labels:
  `docs/studies/retron_hairpin_design/msd_design_hit_labels.txt`
- Compiler outputs:
  caller-chosen transient directories such as
  `/tmp/dnadesign_retron_msd_design_references`, or later the owning Reader
  experiment `inputs/designs/` directory.
- Snapback primitive workspace:
  `src/dnadesign/cruncher/workspaces/de033`
- Snapback primitive runbook:
  `src/dnadesign/cruncher/workspaces/de033/runbook.md`
- Base-junction context note:
  `docs/studies/retron_hairpin_design/scar-nick-base-junction.md`
- Scar-nick workspace:
  `src/dnadesign/cruncher/workspaces/scar_nick_teto`
- Scar-nick workspace runbook:
  `src/dnadesign/cruncher/workspaces/scar_nick_teto/runbook.md`
- Scar-nick source configs:
  `src/dnadesign/cruncher/workspaces/scar_nick_teto/configs/scar_nick/teto_upstream_processing.bbsI_hf.scar_nick.yaml`
  and
  `src/dnadesign/cruncher/workspaces/scar_nick_teto/configs/scar_nick/teto_upstream_processing.paqci_core_panel.scar_nick.yaml`
- Contrast workspace: `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr`
- Direct YIU contrast spec: `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr/configs/yiu/tetr_teto2_wt_direct.yiu.yaml`

### Decision boundaries

- Keep `released-product Snapback`, `preserved-site Snapback`, and `YIU` as
  separate contracts.
- Keep scar-nick base-junction semantics separate from released-product
  Snapback cap/shortening semantics.
- Keep retron logic in the study as motivation and review context, not as
  hidden scoring hooks or silent solver relaxations.
- Keep the route ladder explicit: label/parts first for compiler requests,
  primitive solver only when a constraint is missing, and status/preflight only
  for explicit progress or blocker questions. Use `pipeline.yaml` and
  `ops.study.yaml` only when machine-readable command grouping or preflight
  declarations are the real need.

### Evidence ladder

- Study route map:
  `docs/studies/retron_hairpin_design/routes.md` for the canonical product
  and primitive handoff
- Regenerable released-product solve bundle:
  `src/dnadesign/cruncher/workspaces/de033/outputs/released_solve`
  with a solve report, hit table, and materialized per-hit triptych plots when
  produced by the runbook. Generated outputs are ignored and may be absent after
  workspace cleanup.
- Explicit MSD-HOPV5 visual comparison:
  `src/dnadesign/cruncher/workspaces/msd-HOPV5_snapback` renders the prior
  `Nt.Bpu10I` MSD-HOPV5 example without treating it as a released-product solve result.
- Study command ladder:
  `docs/studies/retron_hairpin_design/pipeline.yaml` for machine-readable
  command groups and bootstrap support
- Scar-nick base-junction context:
  `docs/studies/retron_hairpin_design/scar-nick-base-junction.md`
- Regenerable scar-nick profile-panel bundles:
  `src/dnadesign/cruncher/workspaces/scar_nick_teto/outputs/scar_nick/teto_upstream_processing_bbsI_hf`
  and
  `src/dnadesign/cruncher/workspaces/scar_nick_teto/outputs/scar_nick/teto_upstream_processing_paqci_core_panel`.
  These outputs are generated from same-workspace configs; current BbsI-HF plus
  PaqCI coverage reaches 13/14 active profile buckets, with `WMWM` still
  uncovered under the strict catalog policy.
- Linear ssDNA composition handoff:
  `docs/studies/retron_hairpin_design/linear-ssdna-composition.md`
- Study-owned MSD design registry:
  `docs/studies/retron_hairpin_design/msd_design_registry.yaml`
- Study-selected MSD label list:
  `docs/studies/retron_hairpin_design/msd_design_hit_labels.txt`
- Generic linear ssDNA composition dev spec:
  `docs/dev/plans/2026-05-13-generic-linear-ssdna-composition-spec.md`
- Generic linear ssDNA composition execution plan:
  `docs/exec-plans/completed/2026-05-13-generic-linear-ssdna-composition.md`
- Construct dogfood config:
  `src/dnadesign/construct/workspaces/retron43_teto_manual_x8/config.composition.yaml`
- Generated Construct/BaseRender local bundle:
  `src/dnadesign/construct/workspaces/retron43_teto_manual_x8/outputs/retron43_teto_manual_x8`
  with `visual/sequence_evidence_map_v1.json` and
  `baserender_jobs/component_span_qa_svg.yaml` after running the config, plus
  `folding/secondary_structure_prediction_request_v1.yaml`,
  `folding/folding_preflight.json`, and
  `folding/secondary_structure_prediction_v1.json` when folding is enabled.
  The local dogfood route uses the uv-managed ViennaRNA Python API
  (`viennarna`, imported as `RNA`) and currently records `status=ok`;
  `RNAfold` CLI remains an optional fallback and is not on the local PATH. The
  `outputs/` tree is generated and should not be committed unless explicitly
  requested.
- Generated ViennaRNA-native plot artifacts are opt-in through
  `visual.emit: [viennarna_secondary_structure_svg_v1]` and may also be
  republished with `uv run folding plot`. Current dogfood folding QA records
  `predicted_pair_count=259`, `cross_copy_pair_count=259`, and
  `intended_missed_count=8` for the declared intra-copy payload pairings.
- Study-owned MSD design-reference compilation is available through
  `uv run python -m dnadesign.studies.studies.retron_hairpin_design.cli`. It consumes
  user-provided labels plus study registry metadata, emits a shallow
  design-reference bundle with `README.md`, `manifest.json`,
  `reference_index.tsv`, `msd_design_catalog_v1.json`, and flat per-design
  `msd_design_reference_v1` records under `references/` into an explicit
  caller-chosen transient directory, and is intentionally not a top-level
  `retron-msd` script or persistent workspace family.
- Released-product workflow:
  `src/dnadesign/cruncher/docs/guides/snapback_released_workflow.md`
- Released-product artifact reference:
  `src/dnadesign/cruncher/docs/reference/released_snapback_artifacts.md`
- YIU workflow:
  `src/dnadesign/cruncher/docs/guides/yiu_workflow.md`
- Consolidated retron/P4 and YIU note:
  `src/dnadesign/cruncher/docs/dev/2026-04-19-retron-p4-hairpin-variant-audit.md`

### Next actions

1. For a lab-facing ID or complete part set, lint or compile through
   `uv run python -m dnadesign.studies.studies.retron_hairpin_design.cli`.
2. For missing parts, open `docs/studies/retron_hairpin_design/routes.md` and
   route to the smallest primitive owner: Snapback, scar-nick, or YIU contrast.
3. When the question shifts from solving primitives to composing sequence
   artifacts, open `docs/studies/retron_hairpin_design/linear-ssdna-composition.md`.
   The compiler materializes one MSD unit per design; the older manual x8
   Construct dogfood remains a separate fixture.
4. Run the pinned study preflight only when the real question is blocker or
   execution-readiness posture.
