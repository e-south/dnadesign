## Infer Lanes Route Detail

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Parent router: [README.md](../README.md).

### Infer lanes

- Type: `route`
- Plane: `control-plane`
- Surface role: `operator`
- Owner-boundary: `infer`
- Current state: `complete` for supported Evo2 7B sequence-view sidecars
- Entry artifact: `usr_prom_eth_cip_anchor`, `construct_prom_eth_cip_context`,
  `construct_prom_eth_cip_reference_core60`, and
  `construct_prom_eth_cip_reference_contexts`
- Exit artifact: dataset-local `_derived/infer/` sidecars plus checked-in infer
  lane configs
- Primary doc/workspace: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/README.md`
- First command: `uv run ops runbook fill-infer --study-dir docs/studies/stress_ethanol_cipro_growth --no-submit`
- Route note: Infer lanes are execution configs layered on top of the current
  study phase; they do not replace the study lifecycle record. The supported
  Evo2 7B lanes now plan no runnable GPU work. Notify runbooks remain the
  historical execution surfaces for one USR event stream per lane, including
  the split reference core60, reference-context-forward, and
  reference-context-reverse lanes.
