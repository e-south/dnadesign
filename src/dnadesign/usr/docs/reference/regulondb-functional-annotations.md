# RegulonDB Functional Annotation Sidecars

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-14

This contract projects BioCyc SmartTable regulator GO terms onto the existing
`usr_regulondb_native_promoters` regulatory-interaction sidecar. It is an
additive relation-sidecar enrichment. It does not rewrite native promoter
records, merged stress-study anchor rows, construct contexts, or Infer outputs.

## Source Route

Primary source:

```text
RegulonDB 13 TF-RISet promoter interactions
-> usr_regulondb_native_promoters/_relations/regulatory_interactions.parquet
-> BioCyc KB 29.6 SmartTable regulator GO terms
-> usr_regulondb_native_promoters/_relations/*go*.parquet
```

The BioCyc files are resolved through the public `dnadesign-data` catalog:

- `biocyc_29_6_smarttable_regulator_go_terms`
- `biocyc_29_6_smarttable_regulator_go_coverage`

Do not hard-code sibling-repo paths in downstream tools. Use
`dnadesign_data.catalog.sources.resolve_source_record` or
`dnadesign-data-sources resolve`.

## Output Sidecars

The projection writes three dataset-local relation sidecars:

- `_relations/regulator_go_terms.parquet`: one row per interacting RegulonDB
  regulator and BioCyc GO term.
- `_relations/promoter_regulator_go_terms.parquet`: one row per USR promoter,
  regulator, and GO term.
- `_relations/regulator_go_coverage.parquet`: one row per interacting regulator,
  with `usr_promoter_count`, `source_promoter_count`,
  `regulatory_interaction_count`, `matched_go_term_count`, and mapping status.

The sidecars include BioCyc KB version, SmartTable ID, and SHA-256 hashes for
the source terms and coverage TSVs.

## Contract Checks

The projection fails fast when:

- `records.parquet` or `_relations/regulatory_interactions.parquet` is missing.
- regulatory interactions have blank `usr_id`, orphan `usr_id`, or blank
  `regulator_id` values.
- BioCyc terms or coverage TSVs are missing required columns.
- an interacting regulator is absent from the BioCyc coverage table.
- covered-regulator fraction is below the configured minimum.

Default coverage minimum is `0.95`. The current local materialization covers
`203/205` interacting regulators with at least one BioCyc GO term.

## Run

Dry-run the contract:

```bash
PYTHONPATH=/path/to/dnadesign-data/src \
uv run python src/dnadesign/usr/scripts/project_regulondb_functional_annotations.py \
  --data-root /path/to/dnadesign-data
```

Write sidecars:

```bash
PYTHONPATH=/path/to/dnadesign-data/src \
uv run python src/dnadesign/usr/scripts/project_regulondb_functional_annotations.py \
  --data-root /path/to/dnadesign-data \
  --write
```

## Claim Boundary

These sidecars say that native RegulonDB promoters are associated with
regulators that have source-backed GO annotations. They do not say that a
truncated core60 sequence in pDual10 carries the full causal biology, and they
do not label synthetic promoters mechanistically.
