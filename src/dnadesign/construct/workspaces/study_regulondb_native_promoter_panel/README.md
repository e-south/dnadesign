## RegulonDB Native Promoter Construct Workspace

This workspace derives `usr_regulondb_native_promoter_core60` from
`usr_regulondb_native_promoters`.

The current RegulonDB native promoter USR dataset has TSS coverage for all
3,182 records, but its `promoter_boxes` relation has no -10/-35 rows. The
core60 route therefore uses the source-window contract instead of box
centering: RegulonDB PromoterSet sequences are 81 bp windows with TSS at
sequence offset `60`, so the core60 analysis window is `[0,60)`.

Commands:

```bash
uv run construct workspace doctor --workspace src/dnadesign/construct/workspaces/study_regulondb_native_promoter_panel
uv run construct workspace validate-project --workspace src/dnadesign/construct/workspaces/study_regulondb_native_promoter_panel --project native_tss_upstream_core60
uv run construct workspace run-project --workspace src/dnadesign/construct/workspaces/study_regulondb_native_promoter_panel --project native_tss_upstream_core60 --dry-run
uv run construct workspace run-project --workspace src/dnadesign/construct/workspaces/study_regulondb_native_promoter_panel --project native_tss_upstream_core60 --format json
uv run usr validate usr_regulondb_native_promoter_core60 --strict
```

The 2026-04-29 materialization wrote 3,182 `analysis_window` sequence views and
3,181 canonical 60 bp sequence records. That one-row difference is expected USR
sequence deduplication for duplicate derived windows. The dataset root is a
generated USR artifact under
`src/dnadesign/usr/datasets/usr_regulondb_native_promoter_core60`; manage it
through USR data sync rather than git.
