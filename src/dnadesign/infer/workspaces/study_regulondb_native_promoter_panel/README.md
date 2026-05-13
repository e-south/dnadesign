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
- `config.sequence_views.native_full_plus_tss_upstream_core60.evo2_7b.yaml`:
  additive dogfood lane that runs full 81 bp native `source_record` views once
  and records both full-sequence `seq_mean` and explicit `[0,60)`
  `core60_mean` pooled sidecars. This config is not part of the default
  study-level `fill-infer` quota unless a runbook is promoted for it.

All configs request the intermediate block mean, output-layer mean,
mean-per-token log likelihood, and total log likelihood sidecars.

Current local status as of 2026-04-30:

- Standard `fill-infer` lanes are complete locally: native/full and derived
  core60 both report zero missing products, vectors, scalars, or stale
  sidecars.
- The additive native/full plus `[0,60)` config is also complete locally:
  6,364 contexts, 12,728 reusable vectors, and 6,364 reusable scalars.
- A read-only Evo2 7B recompute check sampled 128 native records and 128
  derived core60 views; fresh output-layer means, intermediate embedding means,
  and log-likelihood scalars matched persisted sidecars exactly.
- Direct reruns of a complete config still perform sidecar reconciliation. Use
  the study-level `fill-infer` runbook path for batch operations so complete
  lanes are skipped before GPU submission.

Preflight commands:

```bash
uv run infer validate config --config src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.native_full.evo2_7b.yaml
uv run infer validate sequence-view-completion --config src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.native_full.evo2_7b.yaml --format json --mode inventory
uv run notify setup resolve-events --tool infer --config src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.native_full.evo2_7b.yaml --json
uv run infer validate config --config src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.core60_tss_upstream.evo2_7b.yaml
uv run infer validate sequence-view-completion --config src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.core60_tss_upstream.evo2_7b.yaml --format json --mode inventory
uv run notify setup resolve-events --tool infer --config src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.core60_tss_upstream.evo2_7b.yaml --json
uv run infer validate config --config src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.native_full_plus_tss_upstream_core60.evo2_7b.yaml
uv run infer validate sequence-view-completion --config src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.native_full_plus_tss_upstream_core60.evo2_7b.yaml --format json --mode inventory
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
