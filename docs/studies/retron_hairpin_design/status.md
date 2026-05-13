## Retron Hairpin Design Effort

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-09

### At a glance

- This study tracks retron TetO hairpin design across two related surfaces:
  cap/shortening geometry through released-product Snapback and base-junction
  scar geometry through scar-nick.
- The active cap/shortening execution lane is `released-product Snapback` in
  `de033`.
- The active base-junction context lane is scar-nick through the `scar_nick`
  subpackage. It owns Type IIS retained scar space, terminal nick feasibility,
  B26/B43 calibration, and profile-diverse `S0=M` scar analogs.
- `YIU` stays in the record as a contrast check on boundary language. It is not the topology engine for this effort.
- The retron/P4 note stays in scope as framing evidence only. It motivates
  compact released products and disrupted basal-stem architecture, but it does
  not become Cruncher scoring logic.

### Quick route

- Snapshot:
  `uv run ops progress show cruncher.data-plane.cruncher-study-status --study-dir docs/studies/retron_hairpin_design --json`
- Preflight:
  `uv run ops progress show cruncher.data-plane.cruncher-study-preflight --study-dir docs/studies/retron_hairpin_design --scope next --json`
- Repo-local study shortcut:
  `.agents/skills/retron-hairpin-study/SKILL.md`

### What is settled

- The active construction lane is still `released-product Snapback` in `de033`.
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
- BbsI-HF retains 10/256 strict scars; PaqCI retains 14/256 by adding
  `TTCA`, `TTCC`, `TTCG`, and `TTCT`; BsaI-HFv2 retains 0/256.
- Exact B26 `MXMX` remains a biological control architecture, but it is not
  scar-compatible under the `S0=M` ligation constraint.
- The scar-nick design target is now profile-diverse, `S0=M`,
  ligation-aware `scar_nick` coverage across `S3/S2/S1`, not exact B26
  sequence preservation or an `MXXM`-centered panel.
- Use `routes.md` for the ordered command ladder and the deeper boundary notes.

### Current phase and surfaces

- Current phase: `snapback_released_solve`
- Next-scope preflight stays read-only.
- Next owner surface: `src/dnadesign/cruncher/workspaces/de033/runbook.md`
- Primary workspace: `src/dnadesign/cruncher/workspaces/de033`
- Base-junction context note:
  `docs/studies/retron_hairpin_design/scar-nick-base-junction.md`
- Scar-nick workspace:
  `src/dnadesign/cruncher/workspaces/scar_nick_teto`
- Contrast workspace: `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr`
- Direct YIU contrast spec: `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr/configs/yiu/tetr_teto2_wt_direct.yiu.yaml`

### Decision boundaries

- Keep `released-product Snapback`, `preserved-site Snapback`, and `YIU` as
  separate contracts.
- Keep scar-nick base-junction semantics separate from released-product
  Snapback cap/shortening semantics.
- Keep retron logic in the study as motivation and review context, not as
  hidden scoring hooks or silent solver relaxations.
- Keep the route ladder explicit: status first, preflight for blockers, and
  `routes.md` as the canonical post-probe handoff. Use `pipeline.yaml` and
  `ops.study.yaml` only when machine-readable command grouping or preflight
  declarations are the real need.

### Evidence ladder

- Study route map:
  `docs/studies/retron_hairpin_design/routes.md` for the canonical
  post-probe handoff
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
- Released-product workflow:
  `src/dnadesign/cruncher/docs/guides/snapback_released_workflow.md`
- Released-product artifact reference:
  `src/dnadesign/cruncher/docs/reference/released_snapback_artifacts.md`
- YIU workflow:
  `src/dnadesign/cruncher/docs/guides/yiu_workflow.md`
- Consolidated retron/P4 and YIU note:
  `src/dnadesign/cruncher/docs/dev/2026-04-19-retron-p4-hairpin-variant-audit.md`

### Next actions

1. Run the pinned study preflight when the real question is blocker or
   next-run readiness.
2. Open `docs/studies/retron_hairpin_design/routes.md` for the ordered
   post-probe handoff; it owns the released probe, released solve, validation
   fixture audit, scar-nick base-junction context, and the contrast-only YIU
   branch.
