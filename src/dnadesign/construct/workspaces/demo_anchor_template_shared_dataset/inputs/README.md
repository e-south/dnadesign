## demo_anchor_template_shared_dataset inputs

- `seed_manifest.yaml` is written by `construct seed anchor-template-demo`.
- The manifest records the seeded dataset ids, record ids, lengths, slot intervals, and template checksum.
- This demo defaults anchor-part and template records to the workspace-local USR root `outputs/usr_datasets/`.
- The seeded dataset ids are `anchor_parts_demo` and `template_parts_demo`.
- Human-readable record names are materialized into `records.parquet` as `usr_label__primary` and `usr_label__aliases`.
- Both packaged projects write into the shared output dataset `anchor_template_shared_dataset_demo`.
- No FASTA inputs are required for the ordinary tracer-bullet flow.
