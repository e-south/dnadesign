# OPAL Synthesis Handoff

Study-owned physical synthesis handoff for selected stress-study OPAL promoter
candidates.

This package keeps OPAL candidate IDs and promoter-core sequences separate from
cloning logistics. It builds a vendor-neutral manifest first, then projects that
manifest into vendor-specific formats such as an Azenta/GeneWiz workbook.

## Boundaries

- Do not add cloning flanks to OPAL candidate `records.parquet`.
- Do not put vendor workbook semantics in OPAL core.
- Keep `id` as the canonical OPAL candidate identifier.
- Keep `synthesis_name` as a vendor/order alias.
- Treat rendered manifests and workbooks as generated `outputs/**` artifacts.
- Treat cloning restriction-site policy as a strategy-level physical-order
  constraint. OPAL can use the same generic eligibility primitive before
  selection, but this package still revalidates the assembled insert before
  writing vendor files.

## Batch Zero

Batch zero is the pre-assay seed order for the three stress OPAL campaigns. It
is not a trained OPAL model round. The selected rows come from the checked-in
batch-0 selector, which combines DenseGen design metadata, the current
LatentDNA representation choice, and the campaign setup rules before any
measured SFXI labels exist.

The DenseGen axis probe also has round-zero simulation logic, but synthesis
handoff does not import that probe. The probe is an in-silico learning-loop
harness. Physical ordering uses either the canonical batch-0 selector here or,
after assays begin, OPAL `selection-set` records backed by campaign ledgers.

The current checked-in batch-0 source resolves 18 promoters. Ethanol and
ciprofloxacin use single-axis pre-assay priors; mixed activator-plus-LexA rows
are reserved for the AND objective in batch zero only:

- ethanol-factor: 4 `baeR` and 2 `cpxR`
- ciprofloxacin-factor: 6 `lexA`
- AND-objective: 4 `baeR+lexA` and 2 `cpxR+lexA`

The selector requires actual parsed TFBS regulators for these slots. It does
not treat a broad DenseGen composition label as enough evidence for a
motif-bearing order candidate. Exact slot-pattern predicates use DenseGen
zero-based slots ordered by `offset_raw`, strong slots use sigma-35 `f/e`,
exploratory slots use `d/c`, spacers are constrained to 16-19 bp, and all
current selected promoters carry the `TATAAT` sigma-10 core.

The handoff assigns deterministic aliases like `SECG-B0-ETH-01` while
preserving the canonical OPAL candidate `id` in the manifest. `SECG` means
stress ethanol/ciprofloxacin growth, `B0` is the pre-assay batch-zero seed, and
`ETH` is the campaign short code.

## Operator Commands

Preview the checked-in batch-0 lifecycle record and artifact status without
rebuilding batch-0 selector inputs or writing generated artifacts:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff \
  --handoff-id stress-opal-batch0-sfxi-v1 \
  --json
```

Write the campaign-scoped manifest, Azenta/GeneWiz workbook, GenBank, and
feature-table files:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff \
  --handoff-id stress-opal-batch0-sfxi-v1 \
  --write \
  --json
```

`--handoff-id` reads the checked-in lifecycle record at
`docs/studies/stress_ethanol_cipro_growth/record/synthesis_handoffs.yaml`. For
batch zero, preview mode reports lifecycle fields plus the exact campaign-local
artifact paths and current hash/readback status. Add `--source batch0` when a
no-write run should explicitly rebuild the current selector and validate
expected campaign row counts against the lifecycle record.

The current strategy config
`configs/sfxi_promoter_insert_v1.yaml` declares the cloning restriction policy:
BamHI `GGATCC` is allowed only in the 5 prime flank and EcoRI `GAATTC` is
allowed only in the 3 prime flank. Manifest construction scans the final
assembled insert, not just the 60 nt core, so a junction-spanning site fails
before any workbook or GenBank output is trusted.

After measured labels exist, use the same handoff command with a new checked-in
lifecycle record. The measured-round record should set
`source_authority=opal_selection_set`, `selection_epoch=opal_model_round`,
`model_as_of_round=<as_of_round>`, `assay_batch_index=<physical_batch_index>`,
and either one top-level `run_id` or campaign-scoped
`expected_campaigns[].run_id` values. First inspect the OPAL-selected set for
each campaign:

```bash
uv run opal selection-set show \
  -c src/dnadesign/opal/campaigns/<campaign_slug>/configs/campaign.yaml \
  --round <as_of_round> \
  --json
```

Then generate the physical handoff from the checked-in record:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff \
  --handoff-id <measured_round_handoff_id> \
  --write \
  --json
```

`--handoff-id` supplies the batch id, assay batch index, model round, and OPAL
run IDs. If a campaign has multiple run IDs for the same round, the lifecycle
record must carry the selected `expected_campaigns[].run_id` value so future
operators do not reconstruct rerun choices from shell history.

The lower-level measured-round source remains available while drafting a new
record or in tests:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff \
  --source opal-round \
  --round <as_of_round> \
  --run-id stress_eth_cip_ethanol_rf_sfxi_topn=<run_id> \
  --run-id stress_eth_cip_cipro_rf_sfxi_topn=<run_id> \
  --run-id stress_eth_cip_and_rf_sfxi_topn=<run_id> \
  --write \
  --json
```

The lower-level source modes remain available for implementation and fixture
work:

- `--source batch0`: wrap the checked-in pre-assay selector directly.
- `--source opal-round --round <as_of_round>`: wrap OPAL `selection-set` rows
  from measured campaign ledgers while drafting a lifecycle record.
- `--source selected-csv --selected-csv <path>`: fixture/manual-debug input;
  do not use it as the production physical-order source.

Default campaign-local output folders:

- `src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/outputs/synthesis_handoff/stress-opal-batch0-sfxi-v1/`
- `src/dnadesign/opal/campaigns/stress_eth_cip_cipro_rf_sfxi_topn/outputs/synthesis_handoff/stress-opal-batch0-sfxi-v1/`
- `src/dnadesign/opal/campaigns/stress_eth_cip_and_rf_sfxi_topn/outputs/synthesis_handoff/stress-opal-batch0-sfxi-v1/`

Measured-round folders use `stress-opal-r<round>-sfxi-v1` unless an explicit
`--batch-id` is provided.

Each folder contains:

- `<batch_id>__<campaign_slug>__synthesis_manifest.csv`: canonical manifest
  with candidate IDs, aliases,
  campaign/round/source provenance, `selection_epoch`, `assay_batch_index`,
  `model_as_of_round`, core and final sequence hashes, flank spans, and
  restriction-site validation status.
- `<batch_id>__<campaign_slug>__azenta_gene_synthesis.xlsx`:
  GeneWiz/Azenta-ready workbook projection with `Sequence Name` and `Sequence`
  columns.
- `<batch_id>__<campaign_slug>__genbank_inserts/`: directory containing one
  GenBank file per final 90 nt order insert, with flank, promoter-core,
  DenseGen TFBS, sigma-35, sigma-10, alias, hash, and campaign provenance
  qualifiers when DenseGen annotations are available. Sigma-35 feature labels
  include the DenseGen categorical variant, for example `-35 (f)`, and carry
  `sigma35_variant` plus `sigma35_sequence` qualifiers for detached GenBank
  review. Individual filenames are prefixed by
  `<batch_id>__<campaign_slug>__<synthesis_name>__`.
- `<batch_id>__<campaign_slug>__genbank_features.csv`: tabular feature sidecar
  used to render and audit the GenBank coordinates.

DenseGen coordinate projection is intentionally fail-fast:

- TFBS annotations must validate against `offset`; `offset_raw` is not a
  fallback for TFBS.
- Sigma-70 fixed elements (`-35` and `-10`) must validate against
  `offset_raw`; padded `offset`/`end` values are not used for these sites.
- The feature table records `densegen_coordinate_key`,
  `densegen_expected_sequence`, `densegen_offset`, `densegen_offset_raw`, and
  `densegen_orientation` so coordinate provenance is visible in the generated
  artifact.

## First Slice

The first implementation slice supports:

- explicit `SelectedCandidate` rows
- checked-in batch-0 selected rows
- OPAL ledger selected rows for measured rounds
- versioned `CloningStrategy` transforms
- manifest validation with hashes and core spans
- Azenta/GeneWiz workbook rendering and readback validation
- GenBank rendering and readback validation with DenseGen positional
  annotations for batch-0 stress promoters
- CLI help, fixture CSV dry-run/write flow, batch-0 campaign-scoped writes, and
  measured-round campaign-scoped writes
- OPAL `selection-set` reader/export command as the canonical measured-round
  selected-row contract
- record-driven `--handoff-id` resolution for checked-in lifecycle handoffs
