## RegulonDB Native Promoter LatentDNA Workspace

This workspace is the downstream analysis contract for
`usr_regulondb_native_promoters`.

The active config expects both the native source-record 7B sequence-mean
feature sidecars and the TSS-upstream core60 7B core60-mean feature sidecars.
The core60 route preserves one row per Infer alias, then joins parent
RegulonDB metadata from `usr_regulondb_native_promoters` through explicit
metadata lookup derivations.

`latentdna validate workspace --workspace regulondb_native_promoter_panel
--deep` is the contract check for row-count parity, materialized matrix shape,
lookup metadata availability, and notebook control-plane health before using
the workspace for exploratory plots.
