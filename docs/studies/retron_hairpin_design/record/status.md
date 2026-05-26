## Retron Hairpin Design Effort

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

### At a glance

- This study now routes Retron MSD product work through a study-owned compiler:
  user-provided or study-selected parts become frozen
  `msd_design_reference_v1` / `msd_design_catalog_v1` records first.
- Complete labels or complete part sets should compile directly. User-provided
  labels must be preserved exactly; do not substitute scar-compatible bases,
  profiles, caps, or construct numbers to make validation pass. Missing
  cap/shortening constraints route to released-product Snapback; missing basal
  left/right base, terminal-nick, or profile constraints route to scar-nick.
- Sequence artifact output is one MSD unit per design: 5' flank + left base,
  payload primary, a user-selected cap/foldback segment, payload complement,
  right base + 3' flank. Snapback subsection labels are emitted only when
  topology is supplied.
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
  `uv run python -m dnadesign.studies.units.retron_hairpin_design.interfaces.cli.app compile --input docs/studies/retron_hairpin_design/compiler/inputs/msd_design_hit_labels.txt --allow-non-ligatable-s0 --out-dir /tmp/dnadesign_retron_msd_design_references --format json`
- GenBank/native-structure-PNG/review-PNG route for the full checked-in cohort:
  `uv run python -m dnadesign.studies.units.retron_hairpin_design.interfaces.cli.app materialize --spec docs/studies/retron_hairpin_design/compiler/inputs/msd_design_177_194_cap_sources_spec.yaml --out-dir /tmp/dnadesign_retron_msd_sequences --render-format png --format json`
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
  profile drift or non-ligatable `S0` unless the caller explicitly opts into an
  `S0!=M` control.
- The selected 177-194 scar-nick labels compile into one catalog from
  `docs/studies/retron_hairpin_design/compiler/inputs/msd_design_hit_labels.txt`.
- Known cap sequences live in
  `docs/studies/retron_hairpin_design/compiler/catalog/msd_cap_sources.yaml`:
  `C26=AGGC`, `C43=tCCTCAGcccGCTGAGGa`, and selected `C172-C176` de033 source
  labels. The compiler must not infer de033 sequence or topology from a future
  `C###` id by pattern.
- Persistent experimental meaning for the 177-194 cohort now lives in
  `docs/studies/retron_hairpin_design/workbench/design_sets/scar_nick_profile_panel_v1.yaml`;
  `msd_design_hit_labels.txt` is a convenience compiler input.
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
- Use `../routes/README.md` for one-hop routing, `../routes/` for owner-surface detail, and
  `workbench/` for durable hypotheses, effect tags, and compiler/materialization
  provenance.

### Compiler and primitive surfaces

- Compiler module:
  `src/dnadesign/studies/units/retron_hairpin_design/`
- Compiler registry:
  `docs/studies/retron_hairpin_design/compiler/catalog/msd_design_registry.yaml`
- Cap source lookup:
  `docs/studies/retron_hairpin_design/compiler/catalog/msd_cap_sources.yaml`
- Study-selected labels:
  `docs/studies/retron_hairpin_design/compiler/inputs/msd_design_hit_labels.txt`
- Full cohort materialization spec:
  `docs/studies/retron_hairpin_design/compiler/inputs/msd_design_177_194_cap_sources_spec.yaml`
- Study workbench:
  `docs/studies/retron_hairpin_design/workbench/`
- Compiler outputs:
  caller-chosen transient directories such as
  `/tmp/dnadesign_retron_msd_design_references`, or later the owning Reader
  experiment `inputs/designs/` directory.
- Snapback primitive workspace:
  `src/dnadesign/cruncher/workspaces/de033`
- Snapback primitive runbook:
  `src/dnadesign/cruncher/workspaces/de033/runbook.md`
- Base-junction context note:
  `docs/studies/retron_hairpin_design/contexts/cruncher/scar-nick-base-junction.md`
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
  for explicit progress or blocker questions. Use `../operations/runtime/command-groups/pipeline.yaml` and
  `../operations/ops.study.yaml` only when machine-readable command grouping or preflight
  declarations are the real need.

### Evidence ladder

Durable evidence pointers live in `evidence/design-evidence.md`. Keep this
status note focused on current route, settled boundaries, and next actions.

### Next actions

1. For a lab-facing ID or complete part set, lint or compile through
   `uv run python -m dnadesign.studies.units.retron_hairpin_design.interfaces.cli.app`.
2. For missing parts, open `docs/studies/retron_hairpin_design/routes/README.md` and
   route to the smallest primitive owner: Snapback, scar-nick, or YIU contrast.
3. For provenance questions, open `docs/studies/retron_hairpin_design/workbench/`.
4. When the question shifts from solving primitives to composing sequence
   artifacts, open `docs/studies/retron_hairpin_design/contexts/composition/linear-ssdna-composition.md`.
   The compiler materializes one MSD unit per design; the older manual x8
   Construct dogfood remains a separate fixture.
5. Run the pinned study preflight only when the real question is blocker or
   execution-readiness posture.
