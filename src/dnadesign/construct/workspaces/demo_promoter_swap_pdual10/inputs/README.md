## demo_promoter_swap_pdual10 inputs

- `seed_manifest.yaml` is written by `construct seed promoter-swap-demo`.
- The manifest records the seeded dataset ids, record ids, lengths, slot intervals, and template checksum.
- This demo defaults promoter and plasmid records to the workspace-local USR root `outputs/usr_datasets/`.
- The seeded dataset ids are `mg1655_promoters` and `plasmids`.
- Human-readable record names are materialized into `records.parquet` as `usr_label__primary` and `usr_label__aliases`.
- No FASTA inputs are required for the ordinary tracer-bullet flow.
