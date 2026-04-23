## Snapback Shortening Effort

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-22

### At a glance

- This study tracks one narrowing question: can released-product Snapback own the shortening problem without forcing an exact preserved-site nickase into the final compact object?
- The active execution lane is `released-product Snapback` in `de033`.
- `YIU` stays in the record as a contrast check on boundary language. It is not the topology engine for this effort.
- The retron/P4 note stays in scope as framing evidence only. It motivates compact exposed-bottom products and shorter uninterrupted duplex burden, but it does not become Cruncher scoring logic.

### Quick route

- Snapshot:
  `uv run ops progress show cruncher.data-plane.cruncher-study-status --study-dir docs/studies/snapback_shortening_effort --json`
- Preflight:
  `uv run ops progress show cruncher.data-plane.cruncher-study-preflight --study-dir docs/studies/snapback_shortening_effort --scope next --json`
- Repo-local study shortcut:
  `.agents/skills/snapback-hairpin-study/SKILL.md`

### What is settled

This study no longer treats `find a better exact preserved-site nickase` as the
main route. The working construction question is the released-product lane:
start from a precursor, keep the nickase valid until nick, release a
downstream sacrificial region, and judge the exposed post-release bottom strand.
The checked-in operational surface now resolves the whole local nickase preset
catalog as `neb_nicking_v1 + thermo_nicking_v1` rather than silently probing
`NEB` alone.

Operational policy now excludes nickases carrying `FREQUENT_CUTTER`. The lane
also rejects any candidate whose release-site geometry would begin left of
logical origin `0`, and it rejects any nickase placement whose omitted
left-of-origin prefix contains protected bases. The only allowed left-of-origin
exception is a single contiguous fully degenerate `N` block at the leading edge
of the oriented top-strand nickase geometry. That keeps the earlier outside-site
exact frontier out of the accepted solution set for this study, so the checked-in
`de033` workspace currently operates as a whole-catalog near-hit search/solve
lane where redundant hits are collapsed to unique exposed post-nick `stem +
cap` geometries. The checked-in downstream-`BspQI` explicit spec now serves as
an invalid-origin audit fixture rather than a green `released-show` bundle.

This record keeps `YIU` close by because it is useful for boundary discipline.
It is still a mismatch-window tool over a fixed `4 nt` internal junction, so it
cannot quietly absorb the shortening problem by renaming a mismatch pool as a
bulge or scaffold model.

### Current phase and surfaces

- Current phase: `snapback_released_solve`
- Next owner surface: `src/dnadesign/cruncher/workspaces/de033/runbook.md`
- Primary workspace: `src/dnadesign/cruncher/workspaces/de033`
- Contrast workspace: `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr`
- Direct YIU contrast spec:
  `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr/configs/yiu/tetr_teto2_wt_direct.yiu.yaml`
- Next-scope preflight stays read-only: it rechecks the `de033` workspace and
  reruns `released-target-search` before the mutating solve command.

### Decision boundaries

- Keep `released-product Snapback`, `preserved-site Snapback`, and `YIU` as
  separate contracts.
- Keep retron logic in the study as motivation and review context, not as
  hidden scoring hooks or silent solver relaxations.
- Keep the route ladder explicit: status first, preflight for blockers, and
  `routes.md` as the canonical post-probe handoff. Use `pipeline.yaml` and
  `ops.study.yaml` only when machine-readable command grouping or preflight
  declarations are the real need.

### Evidence ladder

- Study route map:
  `docs/studies/snapback_shortening_effort/routes.md` for the canonical
  post-probe handoff
- Active released-product solve bundle:
  `src/dnadesign/cruncher/workspaces/de033/outputs/released_solve`
  with a solve report, hit table, and materialized per-hit triptych plots
- Study command ladder:
  `docs/studies/snapback_shortening_effort/pipeline.yaml` for machine-readable
  command groups and bootstrap support
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
2. Open `docs/studies/snapback_shortening_effort/routes.md` for the ordered
   post-probe handoff; it owns the released probe, whole-catalog solve with
   per-hit plots, and the contrast-only YIU branch.
