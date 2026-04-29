## RegulonDB Native Promoter LatentDNA Workspace

This workspace is the downstream analysis contract for
`usr_regulondb_native_promoters`.

The current checked-in config validates before Evo2 feature sidecars exist:
native feature views are declared as `role: planned`, while metadata cohorts are
read directly from the USR dataset. After the native Infer batch runs, remove or
revise the planned role before materializing the native feature views.

The TSS-upstream core60 dataset is declared in the study binding, but it is not
included as a workspace source until Construct materializes
`usr_regulondb_native_promoter_core60`.
