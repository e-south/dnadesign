## RegulonDB Native Promoter Infer Workspace

This workspace defines Evo2 7B sequence-view extraction lanes for the
RegulonDB native promoter panel.

Configured lanes:

- `config.sequence_views.native_full.evo2_7b.yaml`: runs the existing
  `source_record` views from `usr_regulondb_native_promoters` with `seq_mean`
  pooling.
- `config.sequence_views.core60_tss_upstream.evo2_7b.yaml`: runs the
  materialized derived `analysis_window` views from
  `usr_regulondb_native_promoter_core60` with `core60_mean` pooling.

Both lanes request the intermediate block mean, output-layer mean,
mean-per-token log likelihood, and total log likelihood sidecars.

Preflight commands:

```bash
uv run infer validate config --config src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.native_full.evo2_7b.yaml
uv run infer validate sequence-view-completion --config src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.native_full.evo2_7b.yaml --format json --mode inventory
uv run notify setup resolve-events --tool infer --config src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.native_full.evo2_7b.yaml --json
uv run infer validate config --config src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.core60_tss_upstream.evo2_7b.yaml
uv run infer validate sequence-view-completion --config src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.core60_tss_upstream.evo2_7b.yaml --format json --mode inventory
uv run notify setup resolve-events --tool infer --config src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.core60_tss_upstream.evo2_7b.yaml --json
```

Study-level fill command:

```bash
uv run ops runbook fill-infer --study-dir docs/studies/regulondb_native_promoter_panel
uv run ops runbook fill-infer --study-dir docs/studies/regulondb_native_promoter_panel --submit
```

The fill command reads checked-in study runbook surfaces, inspects sequence-view
completion, skips complete lanes, blocks lanes with missing sequence products or
stale sidecars, and plans/submits only lanes with missing vectors or scalars.

The batch runbooks live under `src/dnadesign/ops/runbooks/presets/`. Feature
outputs are written as canonical sidecars under each input dataset's
`_derived/infer` directory.
