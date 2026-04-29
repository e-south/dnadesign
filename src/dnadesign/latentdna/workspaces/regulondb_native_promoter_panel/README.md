## RegulonDB Native Promoter LatentDNA Workspace

This workspace is the downstream analysis contract for
`usr_regulondb_native_promoters`.

The current checked-in config validates before Evo2 feature sidecars exist:
native and core60 feature views are declared as `role: planned`, while metadata
cohorts are read directly from the USR dataset. After each Infer batch runs,
remove or revise the planned role before materializing that feature family.

The TSS-upstream core60 dataset and its 7B sidecars are declared as planned
workspace sources. They remain absent from the published workspace snapshot
until Construct materializes `usr_regulondb_native_promoter_core60` and Infer
writes the matching `_derived/infer` vector/scalar sidecars.
