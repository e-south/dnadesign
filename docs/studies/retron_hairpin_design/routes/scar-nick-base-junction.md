## Scar-Nick Base-Junction Route

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Use this route when the task is base-junction scar feasibility, B26/B43 profile
calibration, profile-diverse `S0=M` scar analogs, top-versus-bottom nick
flexibility, or schema work for the nick-disposal model.

### Route Contract

- Type: `context`
- Plane: `data-plane`
- Surface role: `base-junction-context`
- Owner-boundary: `cruncher`
- Current state: `context-ready`
- Workspace: `src/dnadesign/cruncher/workspaces/scar_nick_teto`
- Workspace runbook: `src/dnadesign/cruncher/workspaces/scar_nick_teto/runbook.md`
- Study note: `docs/studies/retron_hairpin_design/contexts/scar-nick-base-junction.md`
- Tool-owned detail: `src/dnadesign/cruncher/src/scar_nick/`

### Commands

```bash
uv run cruncher scar-nick design \
  --spec src/dnadesign/cruncher/workspaces/scar_nick_teto/configs/scar_nick/teto_upstream_processing.bbsI_hf.scar_nick.yaml \
  --force-overwrite
uv run cruncher scar-nick design \
  --spec src/dnadesign/cruncher/workspaces/scar_nick_teto/configs/scar_nick/teto_upstream_processing.paqci_core_panel.scar_nick.yaml \
  --force-overwrite
uv run baserender job run \
  src/dnadesign/cruncher/workspaces/scar_nick_teto/outputs/scar_nick/teto_upstream_processing_bbsI_hf/baserender_jobs/scar_nick_terminal_nick.job.yaml
uv run baserender job run \
  src/dnadesign/cruncher/workspaces/scar_nick_teto/outputs/scar_nick/teto_upstream_processing_paqci_core_panel/baserender_jobs/scar_nick_terminal_nick.job.yaml
```

### Notes

The strict study policy is exact terminal nick, top or bottom strand allowed,
zero protected bases downstream of the nick, downstream degenerate `N` only, and
`S0=M` for ligation. The same `scar_nick_teto` workspace can hold multiple
release-enzyme specs and output run dirs; these are not independent workspaces.

Current regenerated strict panel notes record BbsI-HF at `6/256` retained
scars, PaqCI at `10/256`, and BsaI-HFv2 at `0/256`. Current BbsI-HF plus PaqCI
outputs cover 13 of the 14 active profile buckets; `WMWM` remains uncovered
under the current strict catalog policy.

Use `export/table__scar_nick_candidate_pair_calls.csv` as the flat left/right
pair-call handoff table, and rerun the route before making PaqCI-specific
capacity claims.
