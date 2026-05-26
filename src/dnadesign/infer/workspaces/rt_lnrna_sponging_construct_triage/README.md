## RT-lnRNA Sponging Construct Triage Infer Workspace

This workspace owns the checked-in Infer config for the study's six declared
Construct sequence views. The source USR dataset is
`workspaces/studies/rt_lnrna_sponging_construct_triage/usr/rt_lnrna_sponging_construct_triage_construct_contexts_2000bp_v1`.

Use the study runbook surface for batch planning:

```bash
uv run ops runbook fill-infer --study-dir docs/studies/rt_lnrna_sponging_construct_triage
```

The config uses explicit `view_name` selectors and fixed anchor-window pooling
bounds supplied by Construct sequence views. Generated sidecars, Notify state,
logs, and audit JSON belong under `outputs/`.
