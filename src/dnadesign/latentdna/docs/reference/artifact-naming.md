# Artifact Naming Grammar

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-15

Artifact IDs state:

1. what is represented or compared
2. the representation family, when relevant
3. the model family, when relevant
4. the sequence scope, when relevant

Canonical examples:

- `intermediate_embedding_20b_merged_anchor_insert_seq_mean`
- `intermediate_embedding_20b_full_context_1kb`
- `pooled_logits_7b_merged_anchor_insert_seq_mean`
- `design_centroid_margins`
- `context_delta_distribution_intermediate_embedding_7b`
- `representation_health_summary`
- `design_structure_summary`
- `sigma35_ordinal_audit`
- `context_robustness_summary`
- `appendix_umap_gallery`

Forbidden naming patterns in study-facing surfaces:

- atlas-first bundle names
- UI-metaphor names
- benchmark-owned export ids
- `primary` or `main` when they encode internal priority rather than meaning
