## Snapback Shortening Effort

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-21

### Scope

This tracked study keeps one narrow shortening effort coherent across:

- released-product Snapback as the active execution lane
- YIU as boundary and contrast context, not as the topology model
- retron/P4 notes as framing for why compact retained products matter

### Current phase

- Current phase: `snapback_released_probe`
- Primary lane: `released-product snapback`
- Supporting lane: `direct TetR/TetO YIU boundary check`

### Current execution surfaces

- Workspace: `src/dnadesign/cruncher/workspaces/demo_snapback`
- Workspace: `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr`
- Released-product explicit spec:
  `src/dnadesign/cruncher/workspaces/demo_snapback/configs/snapback/demo_released_origin_033.released.snapback.yaml`
- Direct YIU contrast spec:
  `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr/configs/yiu/tetr_teto2_wt_direct.yiu.yaml`

### Hard boundaries

- Treat released-product Snapback as the shortening architecture under test.
- Do not treat YIU mismatch selection as a topology-aware bulge or scaffold model.
- Keep retron logic as motivation and review context, not as first-pass Cruncher scoring or hidden heuristics.
- Preserve the current released-product stance:
  target-first paired nickase + release-enzyme search stays separate from preserved-site Snapback and from YIU.

### Context refs

- Consolidated retron/YIU executive summary:
  `src/dnadesign/cruncher/docs/dev/2026-04-19-retron-p4-hairpin-variant-audit.md`
- `src/dnadesign/cruncher/docs/guides/snapback_released_workflow.md`
- `src/dnadesign/cruncher/docs/guides/yiu_workflow.md`

### Next actions

1. Run the read-only study preflight to confirm the tracked command surfaces still resolve.
2. Run the released-product target-search probe in `demo_snapback` and inspect exact/near-hit posture.
3. Materialize the explicit released-design bundle only after the read-only probe is clean.
4. Re-run the direct TetR/TetO YIU validate/render path only as a contrast check for boundary language, not as the primary shortening engine.
