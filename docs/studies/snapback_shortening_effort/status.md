## Snapback Shortening Effort

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-21

### At a glance

- This study tracks one narrowing question: can released-product Snapback own the shortening problem without forcing an exact preserved-site nickase into the final compact object?
- The active execution lane is `released-product Snapback` in `demo_snapback`.
- `YIU` stays in the record as a contrast check on boundary language. It is not the topology engine for this effort.
- The retron/P4 note stays in scope as framing evidence only. It motivates compact retained products and shorter uninterrupted duplex burden, but it does not become Cruncher scoring logic.

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
downstream sacrificial region, and judge the retained post-release product.

This record keeps `YIU` close by because it is useful for boundary discipline.
It is still a mismatch-window tool over a fixed `4 nt` internal junction, so it
cannot quietly absorb the shortening problem by renaming a mismatch pool as a
bulge or scaffold model.

### Current phase and surfaces

- Current phase: `snapback_released_probe`
- Next owner surface: `docs/studies/snapback_shortening_effort/routes.md`
- Primary workspace: `src/dnadesign/cruncher/workspaces/demo_snapback`
- Contrast workspace: `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr`
- Released-product explicit spec:
  `src/dnadesign/cruncher/workspaces/demo_snapback/configs/snapback/demo_released_origin_033.released.snapback.yaml`
- Direct YIU contrast spec:
  `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr/configs/yiu/tetr_teto2_wt_direct.yiu.yaml`

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
- Study command ladder:
  `docs/studies/snapback_shortening_effort/pipeline.yaml` for machine-readable
  command groups and bootstrap support
- Released-product workflow:
  `src/dnadesign/cruncher/docs/guides/snapback_released_workflow.md`
- YIU workflow:
  `src/dnadesign/cruncher/docs/guides/yiu_workflow.md`
- Consolidated retron/P4 and YIU note:
  `src/dnadesign/cruncher/docs/dev/2026-04-19-retron-p4-hairpin-variant-audit.md`

### Next actions

1. Run the pinned study preflight when the real question is blocker or
   next-run readiness.
2. Open `docs/studies/snapback_shortening_effort/routes.md` for the ordered
   post-probe handoff; it owns the released probe, bundle materialization, and
   contrast-only YIU branch.
