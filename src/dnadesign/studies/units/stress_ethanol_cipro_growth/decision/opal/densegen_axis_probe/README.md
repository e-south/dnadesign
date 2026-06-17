# OPAL Dense Array Construction-Metadata Probes

This package owns two study-local OPAL probes that use Dense Array construction
metadata as synthetic labels. The current narrative surface is the TFBS
synthetic-metadata probe; the older plan-logic suite remains historical
execution precedent. The root package is only the entrypoint layer (`cli.py` and
`__main__.py`). The `tfbs/` subpackage owns the strict TFBS review surface,
where each positive campaign is paired with a matched control so enrichment can
be evaluated from realized selected labels rather than inferred from predicted
scores alone.

The organizing rule is ownership, not historical arrival order: study-specific
Dense Array construction semantics stay here, while OPAL core remains
campaign-agnostic.

- `core/`: shared constants, run specs, artifact layouts, path policy, and
  source-column contracts.
- `plan_logic/`: construction-metadata parsing, four-channel plan-logic labels,
  active-target wiring, label-family registry, and legacy plan-logic nulls.
- `runtime/`: run-matrix planning, scratch USR materialization, OPAL round
  execution, run-root fingerprinting, and sweep guards.
- `evaluation/`: prediction ledger checks, prediction scoring, round dynamics,
  trajectory metrics, and decision rendering.
- `reporting/`: status/progress surfaces, plot generation, suite manifests,
  suite notebooks, seed-replicate summaries, and review documents.
- `cli.py` and `__main__.py`: command-line entrypoints only.

## TFBS Learnability Subpackage

`tfbs/` answers a narrower question than the root plan-logic probe: can the
OPAL harness enrich literal Dense Array TFBS construction labels from the
sequence feature surface without label leakage, and does that enrichment exceed
a matched control?
The reader-facing ontology is intentionally small:

- **Composition:** can active selection enrich sequences with more of a target
  TFBS family? Count-fraction labels use `target_count / 3`, so larger values
  mean more copies of the target family among the three variable TFBS entries.
- **Placement:** can active selection enrich sequences where a target TFBS
  family is in a requested left, middle, or right slot? Placement labels are
  binary: `1` when the requested slot contains the target family and `0`
  otherwise.
- **Diagnostics:** older or weaker controls remain visible only when they help
  explain why the current design is stricter. They are not the main evidence
  surface.

The current completed synthetic-metadata review profile is
`tfbs_count_fraction_probe_v1`, which contains exactly
`lexA_count_fraction`, `cpxR_count_fraction`, and `baeR_count_fraction`. A
learning-loop baseline review now accompanies this profile as an offline
ablation: it retrains no new campaigns, but asks whether the initial X-based
model ranking already explains the count-fraction enrichment, whether iterative
OPAL retraining adds cumulative enrichment over the same acquired budget, and
what fraction of the same-budget known-label ranking gain the active loop
recovers.
The slim named diagnostic profile
`tfbs_slot_position_sentinel_probe_v1` contains `lexA_in_slot0` and
`cpxR_or_baeR_in_slot2` for a 2-label, 3-seed count-matched positional screen.
That screen is diagnostic because its control preserves row-level motif counts.
The stricter boundary profile is
`tfbs_slot_position_count_fixed_sentinel_probe_v1`, which uses the same two
placement labels but restricts each label's candidate universe to rows with
exactly one target-family TFBS construction entry and compares against a
count-fixed shuffled-slot negative control. In that profile, `lexA_in_slot0`
uses `lexA_count == 1`, and
`cpxR_or_baeR_in_slot2` uses `cpxR_or_baeR_count == 1`.
`tfbs_slot_position_count_fixed_baer_middle_probe_v1` adds the minimal middle-slot
placement extension `baeR_in_slot1`, scoped to `baeR_count == 1`, so the review
can show left, middle, and right placement checks without running every
regulator-by-slot combination.
The count-fixed placement profiles also have a learning-loop baseline review.
That surface is boundary-tier evidence: it asks whether placement enrichment was
already present in the initial ranker or benefited from adaptive retraining, but
it does not convert mixed placement outcomes into a general slot-geometry claim.
The broader `tfbs_slot_position_probe_v1` profile contains `lexA_in_slot0/1/2`
and `cpxR_or_baeR_in_slot0/1/2` when a full slot-by-family resolution map is
worth the additional campaign footprint. Presence labels remain valid ontology
members for custom diagnostics, but they are not part of the current canonical
count-fraction claim.

- `schema.py`: literal label ontology: `count`, `presence`,
  `count_fraction`, and `slot_family_presence`.
- `profiles.py`: first-class target-profile contracts that separate the
  completed count-fraction probe, the slim count-preserving slot-position
  diagnostic, the count-fixed two-label placement boundary probe, the BaeR
  middle-slot count-fixed extension, the broader slot-position resolution-map
  probe, and custom
  operator-selected label sets.
- `candidate_scopes/`: label-specific candidate-scope contracts, including the
  count-fixed slot-position rule that computes pool baselines on the filtered
  universe rather than the full construction-metadata candidate pool.
- `contracts.py`: strict row parser, final-coordinate slot contract, and
  passive sigma-core validation.
- `oracle.py`, `manifests.py`: positive-label construction and replay
  manifests.
- `nulls/`: matched-null construction split into contracts, exchangeability
  strata, validators, report/provenance generation, and public builders.
- `null_artifacts.py`: null artifact writing.
- `active_targets.py`: scalar expected-label targets for generic OPAL
  `vector_from_table_v1` and `vector_channel_v1` use.
- `retention.py`: preflight retention estimates for sentinel and full-matrix
  campaign footprints.
- `stage_a/materialization.py`, `stage_a/manifests.py`: Stage A label/null
  materialization, source fingerprints, pairings, and retention estimates.
- `stage_b/configs/`: Stage B campaign-set config generation split by
  contract dataclasses, fail-fast validation, and artifact materialization.
- `stage_b/layout.py`: Stage B filesystem ontology.
- `stage_b/io.py`: fail-fast filesystem and parquet/JSON contracts.
- `stage_b/commands.py`: OPAL validation and ingest command contracts.
- `stage_b/payloads.py`: OPAL YAML payload builders.
- `stage_b/seed.py`, `stage_b/semantics.py`: seed policy and campaign
  identity semantics.
- `stage_b/execution/`: campaign execution split into public contracts, OPAL
  command construction, label-input materialization, manifest validation,
  selection-budget contracts, and per-campaign orchestration.
- `stage_b/prune.py`: scoped artifact cleanup.
- `stage_b/review/`, `stage_b/claims.py`, `stage_b/slot_diagnostics/`, and
  `stage_b/notebook_visuals/`: realized-label review, claim gates, count
  confound diagnostics, and registry-backed plot/notebook-facing visual
  registration. `review/plots/` and `slot_diagnostics/plots/` keep manifest
  contracts, renderer registries, and materialization separated. The
  `notebook_visuals/specs.py` registry owns each Stage B visual's stable kind,
  plot filename, title, caption, alt text, metric contract, and tidy-source
  ownership so plot manifests and OPAL collection visual entries cannot drift.
  The `review/` package keeps artifact readers, trajectory frames, summary
  payloads, and materialization separate.
- `stage_b/learning_loop_baselines/`: offline active/frozen/known-label
  ranking reviews for completed Stage B campaigns. It keeps source loading,
  one-shot scoring, deterministic rank-chunk replay, cumulative-budget metric
  frames, plot contracts, notebook visual entries, and materialization in
  separate modules so the harness ablation does not become a new campaign
  runner.

## Probe Question Ontology

The reader-facing review surface is organized by probe question, not by storage
manifest. A probe question is the scientific comparison being reviewed, such as
count-fraction composition or count-fixed placement. A campaign set is the
positive/control active-learning run bundle for one target label and replicate
seed. A manifest is only the machine-readable provenance file that records or
regenerates those campaign sets. Notebook portfolio code therefore exposes
`probe_question_id` and `probe_question_label`; manifest paths remain
provenance, not the conceptual ontology.

## TFBS Stage B Paired-Start Contract

Stage B campaigns are paired active-learning probes. For each `label_name`, the
sequence-matched metadata campaign and its control campaign must ingest the same
initial labeled sequence IDs before round 0. The initial IDs are
label-value-stratified, not uniformly sampled from the candidate pool, so the
first model has both positive and negative evidence instead of an all-zero
start. The two campaign protocols then diverge only in the label table they read
from. This makes the sequence-matched-versus-control comparison about the
construction-label objective, not about a lucky or unlucky starting batch.

The shared-start seed context is recorded in campaign metadata and pair
manifests as `initial_seed_context`,
`initial_seed_pairing=shared_positive_null_starting_ids`, and an
`initial_label_ids_hash`. Older materializations that started positive and null
from different sampled IDs are valid historical artifacts but should not be used
as the modern paired-start evidence surface.

Replicate variation is a seed dimension, not a hidden plot option. Run the same
target profile with additional `--seed` values to produce seed-pair replicates.
The realized-label trajectory plot renders the observed line for single-seed
materializations; when replicate seed rows are present, it renders mean lines
with sample-standard-deviation bands and records that descriptive, non-CI
interval contract in the OPAL collection visual manifest.

The learning-loop baseline reuses the same shared-start campaign configs,
candidate scopes, label tables, model config, and selection budget. For each
campaign, it trains only on the round-0 seed labels, freezes that ranking, takes
the same `selection_k` chunks over 24 rounds, and evaluates cumulative
selected label lift against the same candidate-pool baseline. It also adds a
same-budget known-label ranking by selecting the highest known labels from the
same post-seed acquisition pool. This surface supports a harness-level statement
about adaptive gain only when the active trajectory beats the frozen replay at
the paired-seed level; the reference reports how much known-label ranking gain
was recovered. It does not broaden the Dense Array metadata claim.

## Review Plot And Outcome Contracts

Probe-level review artifacts stay under `reporting/review/`. Aggregate plots
are registry-backed in `reporting/review/aggregate_plots/`, with separate
contracts, normalized source frames, renderers, and writer orchestration. This
keeps OPAL campaign plots generic while allowing the study to add future
probe-specific review plots by registering a new `ProbeAggregatePlotSpec` and
renderer, rather than hardcoding plot order in the report builder.

All study-owned aggregate and TFBS review plots use the shared
`tfbs.plot_style` axis contract: styled ticks, visible left/bottom axes only,
review grid styling, and square axes where the data shape supports it. OPAL
campaign plots remain OPAL-owned primitives, but the study-generated campaign
configs request those primitives through `plots.yaml` instead of embedding
study rendering code in OPAL core.

For TFBS Stage B campaign-set notebooks, study-owned visuals are registered into
an OPAL `opal.collection_visual_manifest_index.v1` file. After running
`tfbs-stage-b-review`, regenerate the OPAL notebook with:

```bash
uv run opal notebook generate \
  --collection-visual-index <collection_visual_manifest.json> \
  --no-materialize-collection-visuals
```

This keeps OPAL campaign-set rendering generic and prevents generic collection
materialization from clobbering the registered realized-label review visuals.

The review manifest includes `outcome_summary`, a compact operator explanation
of the probe decision. Its interpretation boundary is deliberately narrow:
these are pre-assay construction-metadata learnability results, not growth,
stress-tolerance, TF-binding, regulatory-mechanism, wet-lab phenotype, or
biological-causality claims.

Run with:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.densegen_axis_probe run --gate source
```

Materialize the v1 TFBS learnability Stage A label/null/preflight artifacts with:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.densegen_axis_probe tfbs-stage-a \
  --run-id densegen_tfbs_learnability_stage_a_seed7 \
  --json
```

This writes the positive label table, sentinel matched-null label tables, null
viability reports, row-universe/source/label manifests, `pairing_manifest.json`,
`retention_estimate.json`, and `tfbs_stage_a_manifest.json`. It does not run OPAL
campaigns.

The default suite is `densegen_motif_qa_k12_s3_v1`: K12, initial labels 12,
seeds 7/17/29, 12 planned rounds, and active `densegen_plan_logic4` plus
`tf_family_count` campaign matrices. A single `run` invocation executes one
seed; repeat with `--seed 17` and `--seed 29` after the seed-7 burn-in is clean.

Use `--rounds N` to run a synthetic multi-round OPAL loop in scratch space.
Round 0 ingests the planned train IDs. Later rounds ingest labels for the
previous round's OPAL-selected candidates, using the study-owned construction
label table or permuted control for that scratch run only.

Applied runs write `probe_plan.json` at the run root and refuse to reuse a
nonempty root with a missing or mismatched plan. Use a new `--run-id` for normal
reruns; `--replace-run-root` intentionally deletes and rebuilds the scratch root.
Dry-run JSON reports `planned_plan_path` and `writes_artifacts: false`; it does
not claim that `probe_plan.json` already exists.
`progress --json` is compact by default; add `--full` to include the nested OPAL
campaign progress payloads.
`status --json` exits nonzero for materialized or partially scored roots; `ok`
means scored metrics, a decision, and expected final-round coverage exist for
the run plan.
Refresh configured OPAL plot indexes with `plot --run-root <run> --round all
--json`, then rerun `report --run-root <run> --plots --json` for the final
artifact review.
After seeds 7/17/29 are complete, write the cross-seed completion manifest with:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.densegen_axis_probe suite \
  --run-root .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_motif_qa_k12_s3_v1_seed7_all_r12 \
  --run-root .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_motif_qa_k12_s3_v1_seed17_all_r12 \
  --run-root .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_motif_qa_k12_s3_v1_seed29_all_r12 \
  --out-dir .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/_suite_reviews/densegen_motif_qa_k12_s3_v1_all_r12 \
  --json
```
