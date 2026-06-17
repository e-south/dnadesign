---
doc_id: study-stress-ethanol-cipro-growth-opal-synthesis-handoff
surface: study-context
study_id: stress_ethanol_cipro_growth
owner: dnadesign-maintainers
last_verified: 2026-06-17
type: dev-spec
plane: data-plane
owner_boundary: stress_ethanol_cipro_growth
surface_role: physical-synthesis-handoff
entry_artifact: opal_selected_candidates
exit_artifact: vendor_neutral_synthesis_manifest
implementation_tracker: ../../../../exec-plans/active/2026-06-17-stress-opal-synthesis-handoff.md
---

# OPAL Synthesis Handoff

## Plan Intent Summary

Create a study-owned handoff surface that turns selected OPAL promoter
candidates into order-ready synthesis records without changing OPAL candidate
semantics or coupling the active-learning runtime to a vendor workbook format.

## Worth-Doing Preflight

Best case: selected stress-study promoters can move from OPAL campaign ledgers
to physical synthesis orders with deterministic names, case-aware cloning
transforms, and manifest-backed vendor exports. That matters because round
selection is only experimentally useful if the exact ordered sequence can be
traced back to canonical candidate ID, campaign, round, run, and cloning
strategy.

## Scope

In scope:

- Study-owned synthesis handoff package under
  `src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff/`.
- Vendor-neutral manifest records that preserve canonical OPAL candidate IDs.
- Versioned cloning strategy transforms that add lowercase 5 prime and 3 prime
  flanks around uppercase promoter cores.
- Deterministic, globally unique order aliases that are distinct from
  canonical candidate IDs.
- Azenta/GeneWiz workbook rendering as the first vendor adapter.
- Readback validation from rendered workbook back to the manifest.
- OPAL-ledger selected rows for measured post-assay rounds.
- Tracking through
  `docs/exec-plans/active/2026-06-17-stress-opal-synthesis-handoff.md`.

Out of scope for the first implementation slice:

- Mutating `usr_prom_eth_cip_opal_candidates/records.parquet`.
- Adding cloning flanks to generic OPAL core.
- Modeling vendor pricing, quote state, shipping, purchase orders, or account
  metadata.
- Inferring the final production cloning strategy from historical workbook
  examples without an explicit checked-in strategy config.
- Submitting or committing generated `outputs/**` handoff artifacts by default.

## Ownership Boundaries

- OPAL owns campaign config validation, model fitting, scoring, active
  selection, and ledgers.
- The stress study owns selected-candidate physical handoff semantics, cloning
  strategy naming, order aliases, vendor-neutral manifests, and vendor export
  renderers.
- Vendor adapters are projections from the study-owned manifest. They do not
  define the canonical handoff ontology.
- Candidate IDs remain canonical. Human order names are aliases and must never
  replace `id`.

## Core Contract

The first production contract has these concepts:

- `SelectedCandidate`: OPAL-selected promoter candidate with `campaign_slug`,
  `as_of_round`, `run_id`, selection rank, canonical `id`, uppercase promoter
  `sequence`, `selection_epoch`, `assay_batch_index`, and
  `model_as_of_round`.
- `CloningStrategy`: named, versioned transform with left flank, right flank,
  expected core length, case rules, and expected final length.
- `SynthesisInsert`: canonical candidate plus transformed final order sequence,
  sequence spans, hashes, and validation state.
- `SynthesisBatch`: one batch namespace containing one or more campaign
  selections and globally unique order aliases.
- `VendorExport`: a renderer-specific projection such as Azenta/GeneWiz
  workbook rows.

Required invariants:

- Input promoter cores are uppercase `ACGT`.
- Current promoter cores are 60 nt unless an explicit strategy says otherwise.
- Cloning flanks are lowercase `acgt`.
- Final sequence is exactly
  `left_flank.lower() + core_sequence.upper() + right_flank.lower()`.
- Final length equals `left_flank_len + core_len + right_flank_len`.
- Every canonical `id` is unique in a synthesis batch unless a deduplication
  policy is explicitly configured.
- Every order alias is globally unique within the study alias ledger.
- Manifest row count equals vendor export row count.
- Workbook readback exactly matches manifest aliases and final sequences.

## Batch Zero Semantics

Batch zero is the pre-assay seed order for the three stress-study OPAL
campaigns. It is not a trained OPAL active-learning round. The source is the
study-owned batch-0 selector under
`src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/`,
which chooses reviewed seed rows from DenseGen design metadata and LatentDNA
prior margins before measured SFXI labels exist.

Current batch-0 selection shape:

- ethanol-factor campaign: 6 promoters with 3 `baeR`, 1 `cpxR`, 1
  `baeR+lexA`, and 1 `cpxR+lexA`
- ciprofloxacin-factor campaign: 6 promoters with 4 `lexA`, 1 `baeR+lexA`,
  and 1 `cpxR+lexA`
- AND-objective campaign: 6 promoters with 4 `baeR+lexA` and 2 `cpxR+lexA`
- total physical synthesis rows: 18 promoters

The current selector treats the BaeR prior as an acquisition prior, not a
phenotype claim. Slots require parsed DenseGen TFBS regulators so a
metadata-only composition label is not enough for selection. BaeR slots exclude
CpxR TFBS except for explicit CpxR comparator slots. Strong slots use sigma-35
`f/e`; exploratory slots use `d/c`; spacers are constrained to 16-19 bp. The
current DenseGen sigma-70 core map is `f=TTGACA`, `e=TAGACA`, `d=TTTACA`,
`c=TTGTGA`, and `b=CTGACA`; batch zero excludes `b`. All current batch-0
promoter cores carry the `TATAAT` sigma-10 sequence.

The manifest records `selection_source=batch0_pre_assay`,
`selection_epoch=pre_assay_seed`, `assay_batch_index=0`,
`model_as_of_round=null`, and `run_id=batch0_pre_assay_review` so operators do
not confuse this source with a future OPAL model round. Future measured rounds
should use OPAL `selection-set` records backed by campaign ledgers, with
`selection_epoch=opal_model_round` and explicit `run_id` plus `as_of_round`.

## Anti-Drift Source Map

There are three nearby surfaces, but only two are synthesis sources:

- Canonical pre-assay physical source: `decision/opal/batch0/select.py`.
  Synthesis handoff calls this selector and does not reimplement its row choice.
- Canonical measured-round physical source: OPAL `selection-set`, backed by
  campaign ledgers under
  `src/dnadesign/opal/campaigns/<campaign_slug>/outputs/ledger/`, resolved by
  campaign config, `as_of_round`, and `run_id`.
- Non-source precedent: `densegen_axis_probe` round-zero and round-loop code.
  That probe is a scratch/simulation harness for in-silico labels. It may show
  the lifecycle shape, but it must not become a second physical-order selector.

The synthesis handoff CLI therefore has one stable command surface. Human
operators should start from a checked-in lifecycle handoff id whenever a
physical order batch is being previewed, generated, accepted, ordered, or
assayed:

- `--handoff-id stress-opal-batch0-sfxi-v1`: resolve the checked-in batch-0
  lifecycle record, infer the source authority, validate expected campaign row
  counts and lifecycle fields, and report exact campaign-scoped artifact paths.

The lower-level source modes remain available for implementation, fixtures, and
drafting new measured-round records:

- `--source batch0`: wrap the checked-in pre-assay selector.
- `--source opal-round --round <as_of_round>`: wrap selected rows from OPAL
  ledgers after measured labels have been ingested and a round has run.

Both modes emit the same vendor-neutral manifest and workbook renderer outputs.

The checked-in lifecycle record is
`docs/studies/stress_ethanol_cipro_growth/record/synthesis_handoffs.yaml`.
Generated workbooks remain under `outputs/**`; accepted/ordered/assayed state
is tracked by updating the small record row with hashes and readback status.

Measured-round lifecycle records must also make rerun choice explicit. If one
OPAL `run_id` genuinely applies to all campaigns, put it in the top-level
`run_id` field. Otherwise put `run_id` on each `expected_campaigns[]` row. The
CLI passes those values into OPAL `selection-set` when invoked by
`--handoff-id`, which prevents future operators from selecting a different
same-round rerun by accident. The record-owned `assay_batch_index` is stamped
onto the synthesis manifest; OPAL supplies `model_as_of_round` and selected
candidate rows, but it does not own the physical batch number.

## Operator Fetch Path

Generate the current batch-0 files with:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff \
  --handoff-id stress-opal-batch0-sfxi-v1 \
  --write \
  --json
```

Default generated output paths:

- ethanol:
  `src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/outputs/synthesis_handoff/stress-opal-batch0-sfxi-v1/`
- ciprofloxacin:
  `src/dnadesign/opal/campaigns/stress_eth_cip_cipro_rf_sfxi_topn/outputs/synthesis_handoff/stress-opal-batch0-sfxi-v1/`
- AND:
  `src/dnadesign/opal/campaigns/stress_eth_cip_and_rf_sfxi_topn/outputs/synthesis_handoff/stress-opal-batch0-sfxi-v1/`

Each campaign folder contains `synthesis_manifest.csv` and
`azenta_gene_synthesis.xlsx`. Fetch the workbook for vendor upload, but keep
the manifest beside it for canonical IDs, provenance, core/final hashes, and
readback validation evidence.

Preview the same handoff without writing:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff \
  --handoff-id stress-opal-batch0-sfxi-v1 \
  --json
```

That preview is the clean human fetch check: it returns the three expected
campaign rows in `handoff_record.expected_artifacts` and the live generated-file
state in `handoff_record.artifact_status`, including manifest/workbook
existence, SHA-256 values, manifest row counts, and workbook readback status.

## Measured Round Lifecycle Harness

After physical assay labels exist, the happy path for each campaign round is:

1. Ingest measured labels into the shared observed-label sidecar:

   ```bash
   uv run opal ingest-y \
     -c src/dnadesign/opal/campaigns/<campaign_slug>/configs/campaign.yaml \
     --round <observed_round> \
     --in <labels.parquet-or-csv> \
     --unknown-sequences error \
     --apply \
     --json
   ```

2. Run OPAL for the same `as_of_round`:

   ```bash
   uv run opal run \
     -c src/dnadesign/opal/campaigns/<campaign_slug>/configs/campaign.yaml \
     --round <as_of_round> \
     --json
   ```

3. Verify the OPAL selection artifact against the ledger:

   ```bash
   uv run opal verify-outputs \
     -c src/dnadesign/opal/campaigns/<campaign_slug>/configs/campaign.yaml \
     --round <as_of_round> \
     --json
   ```

4. Inspect or export the canonical OPAL selected set:

   ```bash
   uv run opal selection-set show \
     -c src/dnadesign/opal/campaigns/<campaign_slug>/configs/campaign.yaml \
     --round <as_of_round> \
     --json
   ```

5. Add or update a measured-round lifecycle row in
   `docs/studies/stress_ethanol_cipro_growth/record/synthesis_handoffs.yaml`.
   The record should declare:

   - `source_authority=opal_selection_set`
   - `selection_epoch=opal_model_round`
   - `model_as_of_round=<as_of_round>`
   - `assay_batch_index=<physical_batch_index>`
   - top-level `run_id` if all campaigns share one OPAL run ID, otherwise
     `expected_campaigns[].run_id` for every campaign

6. Generate synthesis handoff files from the same OPAL selection-set contract:

   ```bash
   uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff \
     --handoff-id <measured_round_handoff_id> \
     --write \
     --json
   ```

By default this last command reads the three stress SFXI campaign configs and
writes one campaign-scoped folder per campaign. If a round has multiple run IDs
because it was rerun, the lifecycle record must carry campaign-scoped
`expected_campaigns[].run_id` values before the order files are trusted.

Raw `--source opal-round --round <as_of_round>` remains available while
drafting a record or in fixtures, but it is not the production physical-order
handoff path.

Measured-round default output folders use
`outputs/synthesis_handoff/stress-opal-r<round>-sfxi-v1/`. The manifest records
`selection_source=opal_ledger`, `selection_epoch=opal_model_round`,
`model_as_of_round=<as_of_round>`, the resolved `run_id`, and the exact
`as_of_round`.

## Ordered Action Checklist

1. Persist this study dev spec and the active execution tracker before code.
2. Add RED tests for strategy validation, manifest construction, duplicate
   alias rejection, workbook rendering, and workbook readback validation.
3. Add a minimal `synthesis_handoff` package with contracts, transform logic,
   manifest builders, and an Azenta/GeneWiz renderer.
4. Add a CLI with dry-run validation by default and `--write` for generated
   artifacts.
5. Add route docs and package README examples that use explicit campaign,
   round, run/source, strategy inputs, and campaign-local output paths.
6. Validate with targeted tests and repo gates, then update the execution plan
   with evidence.

## First Slice Contract

Goal: prove the handoff ontology and first renderer with fixtures, without
depending on a completed live OPAL run.

In scope:

- Build manifests from an in-memory or fixture selected-candidate table.
- Build measured-round manifests from OPAL ledger-selected rows once campaign
  ledgers exist.
- Apply one explicit strategy equivalent to a 15 nt left flank, 60 nt promoter
  core, and 15 nt right flank.
- Render and read back an Azenta/GeneWiz-style workbook with `Sequence Name`
  and `Sequence` columns.
- Fail fast on wrong case, wrong length, duplicate aliases, duplicate IDs, and
  readback mismatches.

Done criteria:

- Tests demonstrate manifest and workbook round trip.
- The package is importable without touching OPAL core.
- CLI help is available through
  `python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff --help`.
- Batch-0 dry-run reports campaign counts and default campaign-local output
  directories.
- Measured-round CLI rejects ambiguous same-round reruns unless a `run_id` is
  supplied.
- Measured-round CLI validates selected ledger sequences against the campaign
  records table before rendering vendor files.
- No generated `outputs/**` artifacts are committed.

## Validation And Risk Handling

Functional checks:

- `uv run pytest -q src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff`
- `uv run ruff check src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff`
- `uv run ruff format --check src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff`
- `uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .`
- `uv run python -m dnadesign.devtools.docs.checks`

Risk controls:

- Do not read only `selection_top_k.csv` when live campaign ledgers are
  available. The live source reads ledger-selected rows by `run_id` and
  `as_of_round`; use `opal verify-outputs` as the OPAL-side artifact check.
- Keep strategy transforms separate from vendor renderers so a new vendor does
  not change canonical manifests.
- Keep order aliases separate from canonical IDs so assay labels and future
  OPAL rounds can join by stable candidate identity.
- Keep DenseGen probe execution separate from physical synthesis. Probe
  selections can validate in-silico behavior, but they are not a vendor-order
  source.
- Treat generated workbook and manifest files as outputs. Review before any
  generated artifact is committed.
- For batch zero, fetch files from the OPAL campaign `outputs/synthesis_handoff`
  directories; do not add another checked-in source tree for vendor workbooks.

## Open Questions

- The production alias namespace should either continue historical
  `ES-promoter-N` numbering from the sibling cloning workbooks or switch to a
  campaign/round-prefixed namespace. Until that decision is made, tests should
  use an explicit fixture alias map rather than auto-assign production names.
- The final cloning strategy name and flanks should be confirmed before live
  ordering. The first implementation uses a checked-in example strategy only
  for contract validation.
