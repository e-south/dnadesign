## YIU Boundary Check Route

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Use this route only when the task needs a contrast surface for boundary
language or a reminder of what YIU does and does not model.

### Route Contract

- Type: `route`
- Plane: `data-plane`
- Surface role: `contrast-check`
- Owner-boundary: `cruncher`
- Current state: `planned`
- Workspace: `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr`
- Primary docs:
  `src/dnadesign/cruncher/docs/guides/yiu_workflow.md`
  `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr/runbook.md`

### Commands

```bash
cd src/dnadesign/cruncher/workspaces/demo_monotypic_tetr
uv run cruncher yiu validate --spec configs/yiu/tetr_teto2_wt_direct.yiu.yaml
uv run cruncher yiu render --spec configs/yiu/tetr_teto2_wt_direct.yiu.yaml --force-overwrite --emit-renders
uv run cruncher yiu show --bundle outputs/plots/yiu__tetr_teto2_wt_direct
```

### Boundary

YIU is mismatch-centric payload rendering over a fixed 4 nt internal window. It
is not the shortening topology engine for this Retron effort and must not absorb
Snapback cap/shortening semantics or scar-nick base-junction feasibility.
