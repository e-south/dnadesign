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
```

Materializing the dataset writes generated USR parquet artifacts under
`src/dnadesign/usr/datasets/usr_regulondb_native_promoter_core60`; manage that
root through USR data sync rather than git.
