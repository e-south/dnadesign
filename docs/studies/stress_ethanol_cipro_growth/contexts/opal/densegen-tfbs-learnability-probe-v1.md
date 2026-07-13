---
id: stress-ethanol-cipro-growth-opal-densegen-tfbs-learnability-probe-v1
title: DenseGen TFBS learnability probe v1
owner: dnadesign-maintainers
status: active
last_verified: 2026-07-13
audience:
  - maintainer
  - agent
---

## OPAL DenseGen TFBS Learnability Probe v1

**Status:** study-owned v1 contract and implementation spec
**Owner:** `stress_ethanol_cipro_growth` study package
**Last verified:** 2026-07-13
**Target package:** `src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe`
**Test owner:** `src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe`
This design contract defines the study-owned implementation and is not a result
report. Realized profile boundaries are defined by the source
package README and profile registry under
`src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe`.
Its scope is the scientific contract, OPAL/study ownership boundary,
staged campaign plan, and runtime-retention gates.

Audience and outcome: this is written for maintainers and follow-on agents
working in `dnadesign`. Treat it as a contract and rationale record, not as the
current artifact inventory. It is not authorization to run placement or
full-matrix campaigns before the relevant gates pass.

Final-direction rule: later decisions in the source discussion supersede
earlier exploratory ideas. Do not reintroduce legacy load/amount terminology,
Tier 3 requirements, plan-string primitives, high-dimensional vector MSE
objectives, or DenseGen-specific OPAL-core logic as implementation
"completeness".

### 1. Executive Summary

The TFBS learnability probe is a study-owned synthetic-control benchmark. It
asks whether the OPAL harness can enrich DenseGen variable TFBS construction
features from the current `X` representation.

The v1 contract has two active surfaces:

1. **Family content:** which variable TFBS families are present, and how often?
2. **Slot geometry:** which TFBS family occupies which DenseGen left, middle,
   or right TFBS slot?

The active label ontology is limited to:

- `tf_family_count`
- `tf_family_presence`
- `tf_family_count_fraction`
- `tf_slot_family_presence`

DenseGen sigma-core fixed elements are parsed and retained only as passive
controls and null strata. They must not become active labels.

Active selection uses the predicted expected value of a declared scalar label:

```text
y = 0/1 or count/3
y_hat = E[y | X]
selection_score = y_hat
larger is better
```

Do not use negative MSE-to-vector, high-dimensional geometry reconstruction,
mixed grammar reconstruction, plan-string labels, equal-weight background
channels, or raw exact-offset reconstruction as production active objectives.

Artifact retention is part of the scientific contract. The default mode is
`production_review`, not "keep every intermediate table". OPAL should keep
deterministic replay inputs, selected rows, compact summaries, latest/final
predictions, PNGs, manifests, and collection visuals while avoiding every-round
full-pool ledgers and heavy all-row plot CSVs by default.

Review surfaces are also part of the contract. Study-owned aggregate plots are
registered under the probe review package and must use the shared review-axis
style: styled ticks, no top/right spine, readable axes, and square axes where
the plotted data shape supports them. OPAL campaign and campaign-set plots stay
generic OPAL primitives requested through configured plot manifests; DenseGen
TFBS interpretation remains in this study package. Every generated review
manifest should include an `outcome_summary` that explains PASS/DEBUG/STOP or
PENDING in pre-assay synthetic-oracle terms and repeats the non-claim boundary.

### 2. Scientific Claim And Non-Claims

#### Claim

> OPAL can actively enrich for DenseGen variable TFBS construction features
> from the current X representation, while fixed sigma-core elements are
> withheld from active labels and used only as passive controls/null strata.

This claim is about representation learnability and active selection against
deterministic DenseGen metadata-derived synthetic labels.

#### Non-Claims

The probe must not claim:

- measured growth
- ethanol tolerance
- ciprofloxacin response
- true TF binding
- regulatory mechanism
- wet-lab phenotype
- biological causality

DenseGen plan names such as `cipro`, `ethanol`, and `dual` are allowed only as
review lenses and audit context. They are not primitive label ontology for this
v1 probe.

### 3. Current-State Summary Of The Existing OPAL DenseGen Probe

The existing DenseGen probe established that the mechanics work: DenseGen
metadata can be converted into synthetic labels, OPAL can train on those
labels, campaigns can be run across seeds/splits/nulls, and suite-level
positive/null comparisons can be generated.

Historical suite behavior:

- Suite name: `densegen_motif_qa_k12_s3_v1`
- Seeds: `7`, `17`, `29`
- Rounds: `12`
- Initial labels per campaign: `12`
- Selection K per round: `12`
- Splits: `random_id`, `leave_sigma35_variant`
- Oracle roles: positive/intact and permuted/null
- Historical active families:
  - `densegen_plan_logic4`
  - `tf_family_count`
- Historical matrix size: `72` campaigns
  - 2 label families
  - 3 targets
  - 2 oracle roles
  - 2 splits
  - 3 seeds

The historical suite remains useful as execution evidence and as a
synthetic-control precedent. The current TFBS probe must not copy its active
label strategy.

In particular:

- `densegen_plan_logic4` compresses metadata into plan-shaped vectors and uses
  negative MSE to a target vector.
- The old count targets used plan-shaped composite target names.
- Existing notebook and collection visuals emphasize predicted score curves
  more strongly than raw true-label enrichment.
- Full-pool prediction ledgers and all-row plot CSVs scale poorly with
  campaigns, rounds, seeds, and label dimension.

The v1 probe keeps the good parts:

- study-owned DenseGen parsing
- campaign-agnostic OPAL execution
- positive/null paired comparisons
- seed/split replication
- source-surface validation
- X streaming where already present

It replaces the active probe with scalar, literal, construction-term labels.

### 4. Superseded Ideas

Treat the following as historical alignment drift, not production
requirements.

| Superseded idea | v1 decision | Reason |
|---|---|---|
| `densegen_plan_logic4` as a headline active probe | Audit/history only | Plan names confound TF family content, co-occurrence, and review semantics. |
| Plan strings as primitive labels | Rejected | `cipro`, `ethanol`, and `dual` are review lenses, not active label ontology. |
| 40-channel family x orientation x bin geometry MSE | Rejected as an active target | Sparse vector scoring can reward zeros and is harder to explain than scalar slot labels. |
| 39-channel mixed grammar MSE | Rejected as an active target | It mixes one-hot, binary, continuous, spacing, and score fields into one loss. |
| Equal-weight background channels | Rejected as active targets | Background is mostly complementary to LexA/CpxR/BaeR when exactly three TFBS entries exist. |
| Raw exact-offset reconstruction | Rejected as headline claim | The production geometry claim is slot-family enrichment, not exact coordinate reconstruction. |
| Orientation, motif score, adjacent pairs, pair spacing, full grammar | Deferred to audit/future work | These may be useful later, but they are not required for v1. |
| Negative MSE-to-vector as active objective | Rejected for new probes | Selection should maximize predicted expected scalar label value. |
| DenseGen-specific logic in OPAL core | Rejected | OPAL remains generic. DenseGen semantics stay in the study package. |
| Aliases for retired names | Rejected | Use strict v1 names and fail fast. |

Canonical wording must use literal construction terms: `count`, `presence`,
`count_fraction`, and `slot_family_presence`.

### 5. Final Label Ontology

#### 5.1 `tf_family_count`

Plain question: how many variable TFBS entries from a target family/group are
present?

Examples:

```text
lexA_count = count(LexA)
cpxR_count = count(CpxR)
baeR_count = count(BaeR)
cpxR_or_baeR_count = count(CpxR) + count(BaeR)
```

Allowed range: integer `0..3`.

Use this as a source for `tf_family_count_fraction` and optionally as an
integer diagnostic. For active selection, prefer `presence` or
`count_fraction` unless there is a specific reason to rank on raw count scale.

#### 5.2 `tf_family_presence`

Plain question: is the target family/group present anywhere among the three
variable TFBS entries?

Examples:

```text
lexA_present = 1[lexA_count > 0]
cpxR_present = 1[cpxR_count > 0]
baeR_present = 1[baeR_count > 0]
cpxR_or_baeR_present = 1[cpxR_count + baeR_count > 0]
```

Allowed range: binary `0/1`.

This selects for candidates containing the target TF family/group, regardless
of slot.

#### 5.3 `tf_family_count_fraction`

Plain question: what fraction of the three variable TFBS entries belong to the
target family/group?

Examples:

```text
lexA_count_fraction = count(LexA) / 3
cpxR_count_fraction = count(CpxR) / 3
baeR_count_fraction = count(BaeR) / 3
cpxR_or_baeR_count_fraction = (count(CpxR) + count(BaeR)) / 3
```

Allowed values: `0`, `1/3`, `2/3`, `1`.

This selects for higher target-family representation among the three DenseGen
variable TFBS entries.

#### 5.4 `tf_slot_family_presence`

Plain question: does a target family/group occupy a declared DenseGen TFBS
slot?

Examples:

```text
lexA_in_slot0 = 1[family(slot0) == LexA]
lexA_in_slot1 = 1[family(slot1) == LexA]
lexA_in_slot2 = 1[family(slot2) == LexA]
baeR_in_slot1 = 1[family(slot1) == BaeR]
cpxR_or_baeR_in_slot0 = 1[family(slot0) in {CpxR, BaeR}]
cpxR_or_baeR_in_slot1 = 1[family(slot1) in {CpxR, BaeR}]
cpxR_or_baeR_in_slot2 = 1[family(slot2) in {CpxR, BaeR}]
```

Allowed range: binary `0/1`.

This selects for candidates where the target TF family/group is in a declared
left, middle, or right DenseGen TFBS slot.

#### 5.5 Canonical Target Labels

Minimum production target set:

```text
lexA_present
cpxR_present
baeR_present
cpxR_or_baeR_present
lexA_count_fraction
cpxR_or_baeR_count_fraction
lexA_in_slot0
lexA_in_slot1
lexA_in_slot2
cpxR_or_baeR_in_slot0
cpxR_or_baeR_in_slot1
cpxR_or_baeR_in_slot2
```

Recommended sentinel subset:

```text
lexA_present
cpxR_or_baeR_present
lexA_count_fraction
lexA_in_slot0
cpxR_or_baeR_in_slot2
```

#### 5.6 Observed Label-Rate Sanity Checks

The implementation should reproduce approximate rates on the candidate-ID
intersection / quality-ok row universe:

```text
LexA present:                  about 0.488
CpxR present:                  about 0.392
BaeR present:                  about 0.370
CpxR-or-BaeR present:          about 0.491
LexA slot0/slot1/slot2:        about 0.204 / 0.188 / 0.193
CpxR-or-BaeR slot0/slot1/slot2: about 0.316 / 0.323 / 0.306
```

Use tolerances rather than exact equality. A good default gate is absolute
deviation `<= 0.005` after confirming row universe and parser version. If the
row universe changes, update the manifest and require explicit review.

### 6. Exact DenseGen Field-To-Label Mapping

#### 6.1 Source Inputs

Candidate records provide:

- stable candidate ID
- sequence
- OPAL X vector column/contract
- any source metadata required by existing OPAL ingestion

DenseGen sidecar provides:

- stable candidate ID
- `densegen__used_tfbs_detail`
- optional plan/provenance fields for audit only
- fixed-element detail embedded in `densegen__used_tfbs_detail`

Active labels must be derived only from entries with:

```text
part_kind == "tfbs"
```

Fixed-element entries with:

```text
part_kind == "fixed_element"
```

are passive controls/null strata only.

#### 6.2 Required Parsed Entry Fields

For active labels:

| Field | Required for | Contract |
|---|---|---|
| `part_kind` | all labels | Must be `tfbs` for active entries. |
| `regulator` or equivalent regulator/family field | all labels | Normalize to `LexA`, `CpxR`, `BaeR`, or `background` audit class. |
| `offset_raw` | slot labels | Final 60 bp sequence coordinate. Required for slot sorting. |
| `length` | coordinate validation | Used to compute `end_raw = offset_raw + length`. |

For passive controls/null strata:

| Field | Use |
|---|---|
| `part_kind` | Identify fixed elements. |
| `role` | Map upstream sigma70 core element to sigma35 and downstream sigma70 core element to sigma10. |
| `variant_id` or equivalent | sigma35 variant passive stratum/control. |
| `sequence` or consensus identity | sigma10 consensus identity/control. |
| `spacer_length` | sigma35/sigma10 spacer stratum/control. |
| `offset_raw` | final-sequence coordinate. |
| `length` | coordinate validation. |
| `end_raw` if present | validate against computed end. |

#### 6.3 Family Normalization

Normalize regulator text case-insensitively.

Recommended mapping:

| Normalized text contains | Family |
|---|---|
| `lexa` | `LexA` |
| `cpxr` | `CpxR` |
| `baer` | `BaeR` |
| `background`, empty configured background marker, or known non-target placeholder | `background` audit class |

Fail fast on unknown non-empty regulator values unless an explicit v1 parser
config maps them. Do not silently map unknown regulators to background.

#### 6.4 Count, Presence, And Count-Fraction Formulas

Let `T(id)` be the three parsed `tfbs` entries for a candidate ID.

```text
lexA_count(id) = sum_{e in T(id)} 1[family(e) == LexA]
cpxR_count(id) = sum_{e in T(id)} 1[family(e) == CpxR]
baeR_count(id) = sum_{e in T(id)} 1[family(e) == BaeR]
cpxR_or_baeR_count(id) = cpxR_count(id) + baeR_count(id)
```

```text
lexA_present(id) = 1[lexA_count(id) > 0]
cpxR_present(id) = 1[cpxR_count(id) > 0]
baeR_present(id) = 1[baeR_count(id) > 0]
cpxR_or_baeR_present(id) = 1[cpxR_or_baeR_count(id) > 0]
```

```text
lexA_count_fraction(id) = lexA_count(id) / 3
cpxR_count_fraction(id) = cpxR_count(id) / 3
baeR_count_fraction(id) = baeR_count(id) / 3
cpxR_or_baeR_count_fraction(id) = cpxR_or_baeR_count(id) / 3
```

#### 6.5 Slot Formulas

Sort the three `tfbs` entries by `offset_raw` ascending:

```text
slot0 = leftmost TFBS
slot1 = middle TFBS
slot2 = rightmost TFBS
```

Then derive:

```text
lexA_in_slot0(id) = 1[family(slot0) == LexA]
lexA_in_slot1(id) = 1[family(slot1) == LexA]
lexA_in_slot2(id) = 1[family(slot2) == LexA]

cpxR_or_baeR_in_slot0(id) = 1[family(slot0) in {CpxR, BaeR}]
cpxR_or_baeR_in_slot1(id) = 1[family(slot1) in {CpxR, BaeR}]
cpxR_or_baeR_in_slot2(id) = 1[family(slot2) in {CpxR, BaeR}]
```

Do not use `offset` for slot labels. It is padded-coordinate metadata and does
not define final-sequence slot geometry.

### 7. Coordinate And Slot Contract Using `offset_raw`

The implementation must enforce these facts before generating active labels:

```text
candidate row count:       157,160 OPAL rows
candidate sequence length: exactly 60 bp for every candidate
DenseGen sidecar rows:     157,183 rows
sidecar-only/outlier rows: 23 rows
active row universe:       candidate-ID intersection / quality-ok rows
```

Each active row must have exactly:

```text
3 entries where part_kind == "tfbs"
2 entries where part_kind == "fixed_element"
```

Fixed elements must be:

```text
1 upstream sigma70 core element used as sigma35
1 downstream sigma70 core element used as sigma10
```

Fixed-element coordinate sanity:

```text
sigma35 start spans: 0-32
sigma10 start spans: 22-54
sigma10_start - sigma35_start = 6 + spacer_length
spacer_length in {16, 17, 18, 19, 20}
```

Coordinate rules:

```text
offset_raw maps to actual final 60 bp candidate sequence
offset must not define active slot labels
end_raw = offset_raw + length
0 <= offset_raw < 60
0 < end_raw <= 60
```

Slot rules:

```text
sort the three TFBS entries by offset_raw ascending
slot0 = leftmost TFBS
slot1 = middle TFBS
slot2 = rightmost TFBS
```

Fail fast if:

- sequence length is not 60
- candidate ID is duplicated
- sidecar ID is duplicated after applying the chosen quality filter
- a candidate-table ID is missing required DenseGen sidecar metadata after
  applying the active row-universe filter
- a row has other than exactly three `tfbs` entries
- a row has other than exactly two `fixed_element` entries
- a row lacks `offset_raw` for any active TFBS entry
- `offset_raw` and `length` produce out-of-range coordinates
- slot order is ambiguous because two TFBS entries have the same `offset_raw`
- fixed-element roles cannot be mapped to one sigma35 and one sigma10 element
- `offset` is used by active label code

If any of these failures appear in real data, stop and write an explicit issue
note. Do not add silent fallbacks.

The 23 sidecar-only/outlier rows are expected source-surface residue for the
current data snapshot. Report and exclude them from the active row universe;
do not treat their exclusion as a parser failure.

### 8. Positive Oracle Construction

#### 8.1 Oracle ID

Use an explicit versioned oracle identifier:

```text
densegen_tfbs_learnability_positive_v1
```

The exact string can follow existing repo naming conventions, but it must be
versioned and appear in manifests, configs, label tables, and reports.

#### 8.2 Construction Steps

1. Load candidate records with only required source columns.
2. Load DenseGen sidecar.
3. Hash and schema-record both inputs.
4. Join on stable candidate ID.
5. Restrict to candidate-ID intersection / quality-ok rows.
6. Validate the source and DenseGen contracts from sections 6 and 7.
7. Parse `densegen__used_tfbs_detail` into typed entries.
8. Split entries into active TFBS entries and passive fixed-element entries.
9. Normalize active TFBS families.
10. Generate count, presence, count-fraction, and slot-family labels.
11. Generate passive sigma-core audit fields.
12. Write a compact label table keyed by candidate ID.
13. Write a label manifest with source hashes, parser config hash, oracle
    version, row counts, label schema, and label-rate summaries.

#### 8.3 Leakage Boundaries

Label generation may read DenseGen construction metadata. Training/selection
must not read label-only or audit-only fields through X.

Reject source columns for label construction, split/null strata, target
configuration, or report-side active labels if they create an unintended
shortcut, including:

```text
latentdna__*
infer__*
opal__*
umap_x
umap_y
cluster
opal_prediction
opal_selection
prior campaign outputs
```

`latentdna__*` columns may still be named in the OPAL X contract and used by
OPAL as the feature surface. The v1 oracle builder must not read them when
constructing labels, nulls, strata, or audit fields.

Plan metadata may be retained in audit manifests, but not used as an active
label primitive.

#### 8.4 Label Table Schema

Minimum required columns:

```text
id
quality_flag
lexA_count
cpxR_count
baeR_count
cpxR_or_baeR_count
lexA_present
cpxR_present
baeR_present
cpxR_or_baeR_present
lexA_count_fraction
cpxR_count_fraction
baeR_count_fraction
cpxR_or_baeR_count_fraction
lexA_in_slot0
lexA_in_slot1
lexA_in_slot2
cpxR_or_baeR_in_slot0
cpxR_or_baeR_in_slot1
cpxR_or_baeR_in_slot2
slot0_family
slot1_family
slot2_family
sigma35_variant
sigma10_consensus_identity
spacer_length
sigma35_offset_raw
sigma10_offset_raw
sigma35_end_raw
sigma10_end_raw
oracle_version
label_recipe_hash
```

Store tabular artifacts as Parquet with Zstandard compression unless a small
CSV is explicitly needed for human review.

#### 8.5 Manifest Schemas

The positive oracle build must write three compact manifests before any OPAL
campaign configs are materialized.

`row_universe_manifest.json`:

```text
candidate_records_path
candidate_records_hash
candidate_records_row_count
candidate_records_schema_hash
densegen_sidecar_path
densegen_sidecar_hash
densegen_sidecar_row_count
densegen_sidecar_schema_hash
candidate_id_count
sidecar_id_count
candidate_sidecar_intersection_count
sidecar_only_id_count
candidate_only_id_count
quality_ok_count
active_row_count
excluded_row_count_by_reason
candidate_id_order_hash
```

`label_manifest.json`:

```text
oracle_version
label_recipe_hash
parser_config_hash
row_universe_manifest_hash
label_table_path
label_table_hash
label_table_row_count
label_table_schema
active_label_families
active_label_names
passive_control_names
observed_label_rate_summary
algebraic_consistency_summary
coordinate_contract_summary
known_deviations
```

`source_hash_manifest.json`:

```text
git_sha
uv_lock_hash
source_records_path_hash_row_schema
densegen_sidecar_path_hash_row_schema
x_contract
x_column
python_version
numpy_version
sklearn_version
pyarrow_version
thread_settings
```

These manifests are replay-critical. Retention tools must refuse to delete or
compact them unless a replacement manifest proves equivalent replay coverage.

### 9. Null Oracle Construction And Null Viability Report

#### 9.1 Null Principles

The null oracle must preserve label distribution while breaking ID-to-label
alignment.

The null is not random noise. It is a matched negative control for the
question:

> Can OPAL exploit the relationship between X and DenseGen-derived labels
> beyond marginal label prevalence and sigma-core strata?

Null labels must be deterministic under the configured seed and must write
provenance sufficient for replay.

#### 9.2 Family Content Null

For `tf_family_count`, `tf_family_presence`, and
`tf_family_count_fraction`, prefer:

```text
permute label vectors within sigma35/spacer strata
```

Minimum stratum key:

```text
sigma35_variant + spacer_length
```

If the stratum key is not viable, coarsen in a declared way, for example:

```text
sigma35_variant
```

If no matched stratum scheme is viable, use a global permutation only as a
clearly marked fallback and report the weakness.

Preserve jointly:

- count labels
- presence labels
- count-fraction labels
- composite CpxR-or-BaeR labels
- row universe
- label multiset

Do not independently permute derived labels. Count, presence, and
count-fraction labels must remain algebraically consistent after permutation.

#### 9.3 Slot-Position Controls

For `tf_slot_family_presence`, the control must separate slot placement from
family content. The original count-preserving slot control is useful as a
confound diagnostic, but it is not sufficient as the negative-control evidence
for a slot-position selection claim. The current claim-oriented control fixes
the target-family count in the candidate universe first, then uses a
count-fixed shuffled-slot control for supported placement labels.

Preserve at least:

```text
target-family count
sigma35_variant
spacer_length
```

For a target such as `lexA_in_slot0`, use strata like:

```text
sigma35_variant + spacer_length + lexA_count
```

For a target such as `cpxR_or_baeR_in_slot2`, use strata like:

```text
sigma35_variant + spacer_length + cpxR_or_baeR_count
```

Within each viable stratum, permute the slot-family mapping or the relevant
slot-event labels across matched rows. Preserve row-level slot constraints
where possible by shuffling the whole slot-family vector instead of each slot
independently.

Required slot-control reporting:

- Does the DenseGen-vs-control effect survive when target-family count cannot
  drive selection?
- Does the control preserve the candidate scope and label marginal?
- Does the older count-preserving diagnostic show target-count confounding?
- What fraction of rows kept the same slot-event label after permutation?

#### 9.4 Null Viability Report

Every null build must write a `null_viability_report` with at least:

```text
oracle_version
null_version
null_control_role
preserved_signal
disrupted_signal
negative_control_claim_status
seed
row_count
label_name
stratum_key
stratum_count
min_rows_per_stratum
median_rows_per_stratum
max_rows_per_stratum
fraction_rows_in_singleton_strata
fraction_rows_in_tiny_strata
configured_tiny_stratum_threshold
unchanged_label_fraction_after_permutation
label_leakage_assessment
label_marginal_before
label_marginal_after
label_joint_summary_before
label_joint_summary_after
permutation_entropy
estimated_effective_permutation_count
coarsening_steps_applied
viability_status
warnings
```

Recommended defaults:

```text
tiny_stratum_threshold: 3
fail_if_fraction_rows_in_singleton_strata_gt: 0.01
fail_if_fraction_rows_in_tiny_strata_gt: 0.05
fail_if_label_marginal_changes: true
fail_if_count_distribution_changes_for_slot_null: true
warn_if_unchanged_label_fraction_ge: 0.50
fail_if_unchanged_label_fraction_ge: 0.75
```

These thresholds are configurable, but deviations must be recorded in the
manifest.

The `slot_geometry_count_matched_null` is a **count-preserving confound
control**, not a clean negative control for slot learnability. It preserves
row-level LexA/CpxR/BaeR counts before permuting slot-family assignments. A
model can therefore score well on this control by learning target-family count
rather than slot position. Reports must label this control with
`null_control_role=count_preserving_slot_confound_control` and
`negative_control_claim_status=CONFOUND_CONTROL_ONLY`.

For manuscript claims, a positive-vs-null result is considered a primary
negative-control comparison only when
`negative_control_claim_status=VALID_AS_NEGATIVE_CONTROL`. Count-preserving slot
controls may still be useful, but only as evidence that count structure remains
a confound that must be separated from slot-position learnability.

#### 9.5 Viability Status

Use explicit status values:

```text
PASS
PASS_WITH_COARSENING
FAIL_WEAK_EXCHANGEABILITY
FAIL_LABEL_DISTRIBUTION_CHANGED
FAIL_COUNT_MATCHING_CHANGED
```

Do not proceed to full matrix campaigns unless all sentinel nulls are `PASS`
or explicitly reviewed `PASS_WITH_COARSENING`.

#### 9.6 Null IDs And Pairing Manifest

Use explicit versioned null identifiers. Recommended names:

```text
densegen_tfbs_learnability_family_content_matched_null_v1
densegen_tfbs_learnability_slot_geometry_count_matched_null_v1
```

Every null label table must include:

```text
null_version
null_seed
positive_oracle_version
positive_label_table_hash
null_recipe_hash
stratum_key
coarsening_steps_applied
viability_status
```

Every campaign set must also write a positive/null pairing manifest keyed by:

```text
label_name
label_family
split
seed
oracle_role
positive_oracle_version
null_version
campaign_config_hash
retention_policy_hash
```

The pairing manifest is the source of truth for paired AUC delta, final
positive-minus-null lift, collection visuals, and notebook campaign-set
selectors. Do not infer positive/null relationships from directory names.

### 10. Campaign Matrix Proposal

Campaign execution must be staged. Do not start with the full matrix.

#### 10.1 Stage A: Label/Null/Preflight Only

No OPAL campaigns yet.

Required outputs:

- positive label table
- positive label manifest
- null label tables for sentinel labels
- null viability reports
- row-universe manifest
- source hash manifest
- retention estimate for sentinel and full matrix

Gate to proceed:

- DenseGen contract passes
- label rates match expected sanity ranges
- sentinel labels have sufficient variance for the declared objective
- null viability passes or has explicitly reviewed coarsening
- estimated artifact footprint under configured budget

#### 10.2 Stage B: Sentinel Campaigns

Use a small set that exercises both probes and both label types.

Recommended sentinel labels:

```text
lexA_present
cpxR_or_baeR_present
lexA_count_fraction
lexA_in_slot0
cpxR_or_baeR_in_slot2
```

Recommended sentinel matrix:

```text
labels: sentinel labels above
oracle roles: positive, matched null
split: random_id
seeds: 7 initially, then 7/17/29 after first pass
rounds: 12 unless a shorter smoke setting exists for config validation only
selection_k: current study default unless changed deliberately
retention: production_review
```

Gate to proceed:

- OPAL config validation passes
- run manifests complete
- no forbidden source columns enter X/Y config
- retention mode obeyed
- plots and collection manifests render
- raw true-label enrichment is visible beside predicted-score curves
- positive/null paired metrics compute without special-case code
- no obvious sigma-core drift explains the selected set without being reported

Do not require a positive scientific result as an implementation gate. If the
effect is absent, report it accurately.

#### 10.3 Stage C: Production Review Matrix

Recommended v1 matrix after gates pass:

Family content:

```text
lexA_present
cpxR_present
baeR_present
cpxR_or_baeR_present
lexA_count_fraction
cpxR_or_baeR_count_fraction
```

Slot geometry:

```text
lexA_in_slot0
lexA_in_slot1
lexA_in_slot2
cpxR_or_baeR_in_slot0
cpxR_or_baeR_in_slot1
cpxR_or_baeR_in_slot2
```

Recommended dimensions:

```text
oracle roles: positive, matched null
splits: random_id, leave_sigma35_variant
seeds: 7, 17, 29
rounds: 12 default; longer only after retention estimate passes
```

This matrix must not be run until Stage A and Stage B gates pass.

#### 10.4 Longer Campaigns

Longer runs, additional seeds, or extra labels are allowed only when:

- `artifact_retention.max_estimated_bytes` remains under budget
- null viability remains acceptable for new labels
- labels have sufficient variance
- the report UI can show raw true-label endpoints, not only predicted scores
- replay-critical artifacts are recorded

### 11. Active Objective And Scoring Semantics

#### 11.1 Binary Labels

For labels such as `lexA_present` or `lexA_in_slot0`:

```text
y in {0, 1}
model predicts y_hat = E[y | X]
selection_score = y_hat
larger is better
```

Plot labels should say probability-style expected event labels, for example:

```text
Predicted P(LexA present)
Predicted P(LexA in leftmost TFBS slot)
Predicted P(CpxR or BaeR in rightmost TFBS slot)
```

If the model is an uncalibrated regressor, captions should say predictions are
expected-label estimates for ranking, not calibrated probabilities.

#### 11.2 Count-Fraction Labels

For labels such as `lexA_count_fraction`:

```text
y in {0, 1/3, 2/3, 1}
model predicts y_hat = E[count / 3 | X]
selection_score = y_hat
larger is better
```

Plot labels should say:

```text
Predicted E[LexA count / 3]
Selected true LexA count / 3
```

#### 11.3 Diagnostics

MSE or Brier-like loss can remain a model diagnostic.

Diagnostics must not be presented as the active selection objective for this v1
probe. The active objective is the predicted expected value of the declared
scalar label.

#### 11.4 OPAL Implementation Surface

Preferred OPAL behavior:

- Use existing generic scalar/vector transform and objective mechanisms where
  possible.
- If labels are stored in one-column vector tables, the objective should be a
  selected channel equal to the declared scalar.
- Do not introduce DenseGen-specific objective code in OPAL core.
- Do not add plan-shaped target-vector similarity for these v1 labels.

### 12. Primary Endpoints And Plots

#### 12.1 Primary Endpoints

For each label, split, oracle role, seed, and campaign set, report:

1. **Selected true-label mean over rounds**
   - Binary labels: selected true event rate.
   - Count-fraction labels: selected mean true count fraction.
2. **Pool baseline**
   - Same true-label statistic over the eligible pool.
   - If using matched strata, also show matched-pool baseline where applicable.
3. **Selected true-label lift vs pool baseline**
   - Report both difference and ratio when ratio is well-defined.
4. **Positive-vs-null paired trajectory delta**
   - Report both mean-round lift and normalized trapezoid AUC over the selected
     true-label trajectory for positive and matched null/control campaigns.
   - Keep the column names explicit: `mean_round_*` is a simple round average;
     `trapezoid_auc_*` is the normalized AUC surface.
   - Pair by label, split, seed, oracle comparison, and campaign settings.
5. **Final positive-minus-null lift**
   - Difference between positive and null final-round selected true-label lift.
6. **Seed replicate mean and interval**
   - Use mean and a simple interval across seeds where there are at least three
     seeds.
   - Caption the interval method.
7. **Selected sigma-core balance diagnostics**
   - Compare selected-vs-pool distributions for sigma35 variant, sigma10
     consensus identity, spacer length, sigma35 final coordinate, and sigma10
     final coordinate.

The predicted selected score is an acquisition diagnostic, not the endpoint.
Because OPAL explicitly selects rows with high predicted score, a plot of
`pred__score_selected` can make a null/control campaign look successful by
construction. The peer-review-facing endpoint is the realized selected label
trajectory computed from `selection_top_k.csv` joined to the positive or null
label table.

Stage B must write a realized-label review bundle:

```text
review/realized_labels/tfbs_stage_b_realized_label_trajectory.csv
review/realized_labels/tfbs_stage_b_positive_null_pair_summary.csv
review/realized_labels/tfbs_stage_b_claim_assessment.csv
review/realized_labels/plots/tfbs_stage_b_realized_label_plot_manifest.json
review/realized_labels/tfbs_stage_b_realized_label_review.json
```

When an OPAL campaign-set notebook visual index already exists at:

```text
notebooks/collection_visuals/collection_visual_manifest.json
```

the Stage B review step must register the realized-label plots into that index
as `study_realized_label_review` collection visuals under comparison set
`stage_b_realized_label_review`. This keeps the generic OPAL notebook as the
operator-facing surface while preserving the study-owned semantics for realized
oracle-label review. If an explicit visual-index path is provided and does not
match the expected `opal.collection_visual_manifest_index.v1` schema, the review
step must fail rather than silently writing an incompatible notebook surface.

The pair summary must include `peer_review_claim_status`, with at least:

```text
positive_exceeds_null
not_separated_from_null
null_is_confound_control_only
```

The last status is not a failed run; it is a failed primary negative-control
claim. In a manuscript, those rows should be reported as confound-control
evidence rather than positive-vs-null learnability evidence.

The claim assessment must convert pair metrics into manuscript-facing claim
readiness:

```text
READY_AS_VALID_NULL_LEARNABILITY_SIGNAL
LIMITED_TO_CONFOUND_CONTROL_DIAGNOSTIC
BLOCKED_NOT_SEPARATED_FROM_NULL
BLOCKED_NONPOSITIVE_TRAJECTORY_DELTA
```

Only `READY_AS_VALID_NULL_LEARNABILITY_SIGNAL` rows are eligible for the
conservative ML deliverable claim that OPAL learned a DenseGen TFBS oracle above
a matched valid null. Slot-position controls with preserved row-level TF-family
counts remain useful diagnostics, but their claim-readiness status must prevent
them from being counted as valid-null learnability evidence.

#### 12.2 Required Plot Families

At minimum:

- selected true label vs round, positive and null, with pool baseline
- selected true-label lift vs round, positive and null
- predicted expected label of selected set vs round
- positive-vs-null paired mean-round and normalized-AUC delta summary
- final positive-minus-null lift summary
- seed replicate mean/interval summary
- selected-vs-pool sigma-core balance plots
- label-rate summary for generated positive/null label tables
- null viability summary plots or tables

#### 12.3 Plot Labeling Rules

Use explicit y-axis labels. Examples:

```text
Predicted P(LexA present)
Predicted E[LexA count / 3]
Predicted P(LexA in leftmost TFBS slot)
Selected true LexA presence rate
Selected true LexA-in-slot0 rate
Selected true CpxR-or-BaeR-in-slot2 rate
Positive-minus-null selected true-label lift
```

Do not use vague labels such as:

```text
selected score
objective score
biology score
```

unless the exact mathematical expression is printed in the caption and axis
label.

### 13. Notebook And Reporting Requirements

The OPAL review notebook/report must support campaign-set comparisons as
first-class objects.

Required report surfaces:

- campaign set selector by label, split, seed, oracle role, and positive/null
  relationship
- positive/null paired visual panels
- raw true-label enrichment and lift beside predicted-score curves
- realized-label review plots in the campaign-set notebook visual picker
- pool baseline overlays
- final lift summaries
- seed replicate summaries
- sigma-core balance diagnostics
- null viability summaries
- source/label/oracle manifest display
- retention manifest display

Required captions:

- State that labels are synthetic DenseGen construction labels.
- State that sigma-core fields are passive controls/strata only.
- State that plan names are review lenses only.
- State that predictions are expected-label estimates for ranking.
- State the non-claims from section 2.

Manifest requirements:

- Every generated PNG must have a manifest entry.
- Every compact plot data table must identify its source campaign(s), row
  filters, label name, oracle role, and units.
- Plot-derived data should default to compact summaries in `production_review`
  mode.
- Full all-row plot CSVs are allowed only in `audit_full` mode unless
  explicitly overridden.

### 14. Runtime, Performance, And Retention Requirements

#### 14.1 Known Scaling Risks

The existing probe avoids copying the large candidate records table, but
several risks remain:

1. OPAL streams X but materializes full-pool predictions for each round.
2. Prediction ledgers can write every candidate for every campaign/round.
3. Plot config can retain heavy all-row CSVs.
4. Prediction and plot artifacts scale with:

```text
campaigns x rounds x seeds x label_dim x candidate_rows
```

5. Feature importance over 8192 X dimensions is costly and not central
   evidence.
6. Seed determinism is insufficient without source and environment hashes.
7. JSON/list-heavy prediction schemas are expensive at this scale.

#### 14.2 Required Retention Policy Config

Add a probe-level block equivalent to:

```yaml
artifact_retention:
  mode: production_review
  prediction_ledger: latest_full_plus_selected_history
  plot_tidy_data: compact
  model_artifacts: latest
  tabular_format: parquet_zstd
  max_estimated_bytes: 50000000000
  fail_if_estimate_exceeds: true
```

#### 14.3 Retention Modes

`audit_full` is for first validation/debug of a new oracle or OPAL artifact
path.

Keep:

- all configs
- all labels
- all splits
- all model artifacts
- every-round full prediction ledgers
- selected rows
- full plot data
- compact plot data
- PNGs
- manifests
- collection visuals
- source fingerprints
- environment fingerprints

`production_review` is the default scientific review mode.

Keep:

- deterministic replay inputs
- campaign configs
- source fingerprints
- DenseGen sidecar fingerprints
- label tables or deterministic label recipe plus hash
- split IDs
- selected IDs per round
- observed labels for selected rows
- per-round metrics
- compact summaries
- latest/final full prediction ledger only
- selected-row prediction history
- latest model artifact
- PNGs
- plot manifests
- collection visuals
- retention manifest

Do not keep by default:

- every-round full-pool prediction ledgers
- heavy all-row plot CSVs
- repeated sequence columns inside ledgers
- full feature-importance tidy tables over all 8192 X dimensions

`ephemeral_selection` is for large sweeps or long runs where selection is the
product and replay is handled by manifests.

Keep:

- deterministic replay inputs
- source fingerprints
- label recipe/hash or label table hash
- split IDs
- campaign config
- seed
- selected IDs per round
- selected-row observed labels
- per-round metrics
- thresholds
- score quantiles/histograms
- model hyperparameters
- environment fingerprints
- retention manifest

Discard after use:

- full-pool predictions for intermediate rounds
- full plot data
- non-latest model artifacts

#### 14.4 Replay-Critical Artifacts

All retention modes must preserve or reference:

```text
source records path/hash/row count/schema
DenseGen sidecar path/hash/row count/schema
X contract and X column
oracle version and config hash
generated label table or deterministic label recipe
label table hash and row count
split IDs
campaign config
seed
selected IDs per round
observed labels
metrics
software version
git SHA
uv.lock hash
sklearn/numpy/pyarrow versions
model hyperparameters
thread settings / n_jobs
candidate ID order hash
retention policy and pruning manifest
```

#### 14.5 Preflight Size Estimator

Before execution, estimate:

```text
planned campaign count
round count
candidate row count
label dimension
full prediction rows
full prediction y_hat cells
selected-row ledger rows
plot-derived table rows
expected prediction ledger bytes
expected plot data bytes
expected model artifact bytes
expected total bytes
```

Fail closed if:

```text
expected_total_bytes > artifact_retention.max_estimated_bytes
```

and:

```text
artifact_retention.fail_if_estimate_exceeds == true
```

The estimator must run for sentinel and full matrix modes.

#### 14.6 Streaming Selection Requirement

Add or use an OPAL generic selection path that can:

1. Load candidate X in batches.
2. Predict `y_hat` for a batch.
3. Score the batch.
4. Update a top-K heap or equivalent streaming selector.
5. Update quantile/histogram summaries.
6. Persist selected rows and replay-critical summaries.
7. Optionally persist latest/final full predictions according to retention
   mode.

This should be generic OPAL functionality, not DenseGen-specific code.

#### 14.7 Ledger Schema Requirements

For `production_review`, prediction rows should be keyed by candidate ID and
avoid repeated heavy fields.

Preferred schema principles:

- Do not repeat sequence strings in every prediction row.
- Join sequence lazily from source records when needed.
- Use float32 where scientifically acceptable for prediction arrays.
- Prefer Parquet/Zstandard over CSV for large tables.
- Store selected-row histories separately from full-pool latest/final
  predictions.
- Avoid JSON/list-valued prediction columns for large full-pool ledgers when a
  compact typed representation is available.

#### 14.8 Prune/Compact Tool

Implement a manifest-aware retention tool that:

- reads a campaign or suite root
- applies `artifact_retention`
- writes `retention_manifest.json`
- records deleted, compacted, and retained artifacts
- never deletes replay-critical artifacts
- fails if it cannot prove an artifact is noncritical

### 15. File And Package Ownership Plan

DenseGen-specific logic belongs in:

```text
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe
```

This package owns:

- DenseGen sidecar parsing
- TF family normalization
- coordinate/slot validation
- label-family definitions
- positive oracle construction
- null oracle construction
- split/seed semantics
- label/null manifests
- study-specific plot captions
- biological-language boundaries and non-claims

Existing study package surfaces to inspect before adding new files:

| Surface | Expected v1 use |
|---|---|
| `axis_oracle.py` | Current DenseGen sidecar parsing and label construction precedent. Refactor or extend here only if the v1 parser stays readable and fail-fast; otherwise split v1 parser/oracle modules under the same package. |
| `source_contract.py` | Candidate/sidecar source validation. Add the 60 bp, row-count, `offset_raw`, part-count, and fixed-element contract checks here or in a narrow v1 contract module called from here. |
| `label_families.py` | Study-owned label-family registry. Add strict v1 family names only: `tf_family_count`, `tf_family_presence`, `tf_family_count_fraction`, and `tf_slot_family_presence`. |
| `active_targets.py` | Study-owned OPAL target declarations. Add scalar expected-label targets that rank by the declared label channel. Do not add plan-vector target similarity for v1 labels. |
| `tfbs/nulls/` | Matched-null construction, exchangeability strata, viability reports, and explicit `PASS`/`FAIL_*` status values. |
| `tfbs/stage_a/materialization.py` | Stage A label/null/preflight materialization. Write positive labels, sentinel nulls, source-file hash manifests, pairing manifest, retention estimate, and Stage A summary without running OPAL campaigns. |
| `tfbs/retention.py` | Stage A retention estimator for sentinel and full-matrix campaign modes. Fail closed when estimated retained artifacts exceed the configured budget. |
| `tfbs/stage_a/manifests.py` | Source-file fingerprint, positive/null pairing, and Stage A summary manifest builders. Keep manifest semantics study-owned and replay-safe. |
| `runtime/plan.py`, `reporting/suite_manifest.py`, `runtime/scratch.py`, `tfbs/stage_b/configs/` | Campaign/suite materialization. Generate sentinel configs after Stage A gates pass and before full-matrix configs; preserve retention/preflight manifests. |
| `trajectory_metrics.py`, `suite_replicates.py`, `suite_review.py` | Positive/null paired metrics and replicate summaries. Add selected true-label lift, paired AUC delta, final lift, and seed intervals here. |
| `reporting/plotting.py`, `reporting/suite_notebook.py`, `reporting/review/`, `tfbs/stage_b/review/` | Manifest-backed review surfaces. Add raw true-label enrichment, null viability, and sigma-core balance visuals without making OPAL notebooks DenseGen-specific. |

Tests belong in:

```text
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe
```

Durable docs:

```text
docs/studies/stress_ethanol_cipro_growth/contexts/opal/densegen-tfbs-learnability-probe-v1.md
```

Route link:

```text
docs/studies/stress_ethanol_cipro_growth/routes/decision/opal/README.md
```

If implementation starts, create an execution tracker under:

```text
docs/exec-plans/active
```

OPAL core owns only generic machinery:

- scalar/vector transforms
- objectives
- model training
- prediction
- selection
- ledgers
- plot manifests
- campaign-set comparison mechanics
- retention modes if implemented generically
- streaming top-K selection if implemented generically

OPAL package surfaces to respect:

| Surface | Boundary |
|---|---|
| `src/dnadesign/opal/src/transforms_y/vector_from_table_v1.py` | Accepts generic numeric scalar/vector Y surfaces. Do not add DenseGen parsing here. |
| `src/dnadesign/opal/src/objectives/vector_channel_v1.py` | Preferred existing objective shape for selecting a declared scalar channel. |
| `src/dnadesign/opal/src/objectives/vector_target_similarity_v1.py` | Historical support for plan-vector similarity. Do not use this as the v1 active objective. |
| `src/dnadesign/opal/src/runtime/round/stages/prediction.py` | Generic prediction path. Any streaming-memory improvement must remain campaign-agnostic. |
| `src/dnadesign/opal/src/runtime/round/writebacks.py` | Generic ledger writeback path. Retention and compact ledger changes must not depend on DenseGen labels. |
| `src/dnadesign/opal/src/analysis/plots/` | Generic plot primitives. DenseGen captions and study-specific aggregate plots stay in the study package unless promoted through the generic plot manifest contract. |

Do not put DenseGen-specific parsing, TF family names, slot contracts,
sigma-core fields, or plan-specific captions into OPAL core.

Do not add aliases for retired DenseGen target names.

### 16. Implementation Phases

#### Phase 0: Land The Spec And Execution Tracker

Tasks:

- Add this spec to the target docs path.
- Link it from the OPAL decision README.
- Create an execution tracker under `docs/exec-plans/active` once
  implementation begins.
- Record any deviations from this spec in the tracker.

Recommended tracker filename:

```text
docs/exec-plans/active/YYYY-MM-DD-densegen-tfbs-learnability-probe-v1.md
```

Minimum tracker contents:

- implementation owner and branch
- linked spec path and current spec commit
- planned file/module changes by phase
- Stage A/B/C gate status
- validation commands and latest result
- known blockers and explicit deviations from this spec
- final artifact locations for label tables, null reports, campaign configs,
  retention manifests, plots, notebooks, and suite review

Exit criteria:

- Docs link resolves.
- Tracker identifies implementation owner, task list, validation commands, and
  known blockers once code implementation begins.

#### Phase 1: Source Contract And Parser

Tasks:

- Add/extend source contract validation for candidate records and DenseGen
  sidecar.
- Parse `densegen__used_tfbs_detail` into typed entries.
- Enforce row count, sequence length, part count, fixed-element, and coordinate
  contracts.
- Normalize TF family labels.
- Extract passive sigma-core fields.

Exit criteria:

- Unit tests cover valid rows and failure modes.
- Parser rejects `offset` use for active slots.
- Parser uses `offset_raw` for slot order.

#### Phase 2: Positive Oracle Labels

Tasks:

- Generate `tf_family_count` labels.
- Generate `tf_family_presence` labels.
- Generate `tf_family_count_fraction` labels.
- Generate `tf_slot_family_presence` labels.
- Write label table and label manifest.
- Add observed label-rate sanity checks.

Exit criteria:

- Label table schema matches section 8.4 or a documented superset.
- Label rates match expected ranges.
- Algebraic consistency tests pass:
  - `present == count > 0`
  - `count_fraction == count / 3`
  - `cpxR_or_baeR == cpxR + baeR` for count/fraction variants where
    applicable.

#### Phase 3: Null Construction

Tasks:

- Implement family-content matched permutation nulls.
- Implement slot-geometry count-matched permutation nulls.
- Preserve derived-label consistency.
- Write null label tables and null viability reports.
- Add coarsening/fail logic for weak strata.

Exit criteria:

- Marginal distributions preserved exactly for required labels.
- Count matching preserved for slot nulls.
- Null viability status is explicit.
- Unit tests verify deterministic seed behavior.

#### Phase 4: Active Target And Campaign Materialization

Tasks:

- Add v1 target specs for scalar expected-label selection.
- Generate sentinel configs first.
- Ensure OPAL config validation passes.
- Ensure no DenseGen-specific objective logic is added to OPAL core.

Exit criteria:

- Sentinel campaign configs validate.
- Active score is predicted scalar expected label.
- Plot/report labels use explicit mathematical names.

#### Phase 5: Runtime And Retention

Tasks:

- Add retention config handling.
- Add preflight size estimator.
- Add or wire generic streaming top-K selection if needed.
- Add manifest-aware pruning/compaction.
- Store tables as Parquet/Zstandard by default.
- Suppress heavy plot CSVs in `production_review` unless explicitly
  overridden.

Exit criteria:

- Preflight estimate runs before campaign execution.
- Runs fail if estimated size exceeds budget.
- `production_review` produces a retention manifest.
- Replay-critical artifacts are retained.

#### Phase 6: Plots, Notebook, And Suite Review

Tasks:

- Add true-label enrichment plots.
- Add positive/null paired AUC delta summaries.
- Add final positive-minus-null lift summaries.
- Add sigma-core selected-vs-pool diagnostics.
- Add null viability display.
- Ensure campaign-set comparison IA supports these surfaces.

Exit criteria:

- Notebook/report shows raw true-label enrichment beside predicted expected
  label curves.
- No vague selected-score labels appear.
- Collection visual manifests are complete.

#### Phase 7: Staged Execution

Tasks:

- Run Stage A preflight.
- Run Stage B sentinel campaigns.
- Review label/null/runtime outputs.
- Run Stage C only after gates pass.

Exit criteria:

- Suite review contains all primary endpoints.
- Results are interpreted as synthetic-control learnability only.
- No wet-lab or mechanism language appears in conclusions.

### 17. Test And Validation Plan

#### 17.1 Parser Tests

Cover:

- valid row with exactly three TFBS and two fixed elements
- missing `offset_raw`
- attempted use of `offset` instead of `offset_raw`
- duplicate candidate IDs
- duplicate sidecar IDs
- wrong sequence length
- wrong number of TFBS entries
- wrong number of fixed elements
- unknown regulator text
- ambiguous slot order from tied `offset_raw`
- invalid coordinate range
- fixed elements missing upstream/downstream roles
- invalid spacer relationship

#### 17.2 Label Tests

Cover:

- count formulas
- presence formulas
- count-fraction formulas
- CpxR-or-BaeR composite formulas
- slot-family formulas
- algebraic consistency
- background audit class does not enter required active targets
- label schema manifest completeness
- approximate label-rate sanity checks on fixture or sampled real data

#### 17.3 Null Tests

Cover:

- deterministic output for fixed seed
- different output for different seed where viable
- exact preservation of marginal label distributions
- preservation of count distribution for slot nulls
- preservation of sigma35/spacer strata for matched nulls
- unchanged-label fraction reported
- singleton/tiny strata reported
- coarsening status reported
- failure when exchangeability is too weak

#### 17.4 OPAL Config Tests

Cover:

- generated campaign configs validate
- active objective uses declared scalar expected label
- positive and null campaigns pair correctly
- split IDs are stable and recorded
- model hyperparameters recorded
- no DenseGen parsing import appears in OPAL core

#### 17.5 Runtime/Retention Tests

Cover:

- preflight estimator computes expected bytes
- fail-if-estimate-exceeds behavior
- `audit_full` keeps full artifacts
- `production_review` keeps replay-critical artifacts and compact summaries
- `ephemeral_selection` avoids intermediate full-pool ledgers
- prune/compact tool refuses to delete replay-critical artifacts
- retention manifest records actions
- large tables use Parquet/Zstandard by default

#### 17.6 Plot/Report Tests

Cover:

- true-label enrichment plots render
- predicted expected-label plots render
- positive/null paired AUC delta plots render
- final lift plots render
- sigma-core balance plots render
- null viability tables render
- manifest entries exist for all generated visuals
- captions include non-claims
- y-axis labels are explicit
- no vague selected-score labels appear

#### 17.7 Repo Hygiene Checks

Run the repo's standard checks. At minimum include:

```text
uv run ruff check .
uv run ruff format --check .
uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .
uv run pytest -q src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe
uv run python -m dnadesign.devtools.docs.checks
git diff --check
```

If the full repo check fails because of unrelated dirty state, report the exact
blocker and keep DenseGen-scoped validation separate.

### 18. Risks And Fail-Fast Gates

#### 18.1 Scientific Risks

| Risk | Gate |
|---|---|
| Labels accidentally encode sigma-core signal | Fixed elements are passive only; active label code reads only `part_kind == "tfbs"`; sigma-core balance diagnostics required. |
| Slot geometry is only family content in disguise | Slot null must preserve target-family count and sigma35/spacer strata; report whether positive-vs-null survives count matching. |
| Null strata are too small to permute | Null viability report must pass or coarsen explicitly. |
| Plan names leak into active labels | Plan metadata is audit only; generated active schema must not include plan-string primitives. |
| Scalar predictions are misread as calibrated probabilities | Captions must call them expected-label estimates for ranking unless calibration is performed. |
| Selected sets drift by sigma-core | Report selected-vs-pool sigma35, sigma10, spacer, and coordinate distributions. |

#### 18.2 Engineering Risks

| Risk | Gate |
|---|---|
| Artifact footprint becomes prohibitive | Preflight size estimator with hard byte budget. |
| Full prediction ledgers dominate storage | `production_review` keeps latest/final full predictions plus selected history, not every round. |
| Heavy plot CSVs dominate storage | `plot_tidy_data: compact` by default. |
| Feature importance over 8192 X dimensions burns runtime | Disable or summarize in production mode. |
| Seeds are recorded but environment is not replayable | Record git SHA, lockfile hash, package versions, thread settings, source hashes, and ID-order hash. |
| DenseGen code leaks into OPAL core | Tests/static checks should enforce ownership boundary. |
| Silent fallback hides contract failures | Parser/null/retention paths fail fast with explicit status and manifest entries. |

#### 18.3 Full-Matrix Stop Conditions

Do not run the full matrix if any of these are true:

- source contract fails
- label-rate sanity checks fail without explanation
- any sentinel null has `FAIL_*` viability status
- retention estimate exceeds budget
- sentinel configs do not validate
- plot/report surfaces cannot show raw true-label enrichment
- replay-critical artifact manifest is incomplete
- active target implementation uses negative vector MSE
- active labels read fixed-element fields
- active labels use plan-string primitives

### 19. Concrete Acceptance Criteria

#### 19.1 Documentation

- `docs/studies/stress_ethanol_cipro_growth/contexts/opal/densegen-tfbs-learnability-probe-v1.md` exists.
- It is linked from
  `docs/studies/stress_ethanol_cipro_growth/routes/decision/opal/README.md`.
- If code implementation starts, an execution tracker exists under
  `docs/exec-plans/active`.
- The doc states the final claim and non-claims.
- The doc defines only the v1 active ontology listed in section 5.
- The doc marks plan logic, high-dimensional geometry/grammar MSE,
  exact-offset reconstruction, pair spacing, orientation, and motif score as
  non-production active objectives.
- The doc and UI labels use literal construction terms: `count`, `presence`,
  `count_fraction`, and `slot_family_presence`.
- The generated configs, reports, and UI labels use those same literal terms
  rather than legacy load/amount wording for probe labels.

#### 19.2 Source And Oracle

- Positive oracle uses candidate-ID intersection / quality-ok rows.
- Candidate row count, sidecar row count, and sidecar-only row count are
  reported.
- Every active row has 60 bp sequence length.
- Every active row has exactly three `tfbs` entries and two `fixed_element`
  entries.
- Active slot labels use `offset_raw` only.
- `offset` is not used for active labels.
- Fixed elements are parsed only for passive controls/null strata.
- Label table contains required count, presence, count-fraction, slot-family,
  and passive sigma-core columns.
- Label-rate sanity values match expected ranges or deviations are documented.

#### 19.3 Nulls

- Family content null preserves marginal label distributions while breaking
  ID-to-label alignment.
- Preferred family content null permutes within viable sigma35/spacer strata.
- Slot geometry null preserves target-family count and sigma35/spacer strata at
  least by sigma35 variant plus spacer length.
- Slot geometry null reports whether the slot effect survives count matching.
- Null viability reports include all required fields from section 9.4.
- Weak exchangeability causes fail or explicit coarsening, not silent success.

#### 19.4 OPAL Campaigns

- Sentinel configs are generated and validated before full matrix configs.
- Full matrix generation is gated by Stage A and Stage B outputs.
- Active score is `y_hat`, the predicted expected scalar label.
- MSE/Brier-like loss is diagnostic only.
- Positive/null campaign pairs are manifest-backed.
- Splits, seeds, selected IDs, observed labels, and metrics are retained.

#### 19.5 Plots And Reports

- Reports include selected true-label lift vs pool baseline.
- Reports include positive-vs-null paired AUC delta.
- Reports include final positive-minus-null lift.
- Reports include seed replicate mean/interval where appropriate.
- Reports include selected sigma-core balance diagnostics.
- Raw true-label enrichment appears beside predicted expected-label curves.
- Plot labels are explicit, for example:
  - `Predicted P(LexA present)`
  - `Predicted E[LexA count / 3]`
  - `Predicted P(LexA in leftmost TFBS slot)`
  - `Selected true LexA presence rate`
  - `Selected true LexA-in-slot0 rate`

#### 19.6 Runtime And Retention

- `artifact_retention` config supports `audit_full`, `production_review`, and
  `ephemeral_selection`.
- Default config is equivalent to:

```yaml
artifact_retention:
  mode: production_review
  prediction_ledger: latest_full_plus_selected_history
  plot_tidy_data: compact
  model_artifacts: latest
  tabular_format: parquet_zstd
  max_estimated_bytes: 50000000000
  fail_if_estimate_exceeds: true
```

- Preflight size estimator runs before campaigns.
- Runs fail when estimated bytes exceed budget and `fail_if_estimate_exceeds`
  is true.
- Replay-critical artifacts from section 14.4 are retained.
- Retention manifest records all prune/compact actions.
- Large tables use Parquet/Zstandard by default.
- Heavy all-row plot CSVs are not retained in `production_review` unless
  explicitly overridden.
- Every-round full-pool prediction ledgers are not retained in
  `production_review` unless explicitly overridden.

#### 19.7 Ownership And Architecture

- DenseGen parsing and label construction live only in the study package.
- OPAL core remains campaign-agnostic.
- No DenseGen-specific TF family names, slot contracts, or sigma-core fields
  are added to OPAL core.
- No aliases for retired target names are added.
- Repo checks pass, including lint, tests, config validation, plot/report
  checks, docs checks where available, and `git diff --check`.

### Appendix A: Minimal Config Sketch

This sketch is illustrative. Match the repo's actual config schema.

```yaml
probe:
  name: densegen_tfbs_learnability_probe_v1
  oracle_version: densegen_tfbs_learnability_positive_v1
  row_universe: candidate_intersection_quality_ok
  coordinate_contract:
    sequence_length: 60
    active_coordinate_field: offset_raw
    forbidden_active_coordinate_field: offset
    required_tfbs_entries: 3
    required_fixed_element_entries: 2
  active_label_families:
    - tf_family_count
    - tf_family_presence
    - tf_family_count_fraction
    - tf_slot_family_presence
  passive_controls:
    - sigma35_variant
    - sigma10_consensus_identity
    - spacer_length
    - sigma35_offset_raw
    - sigma10_offset_raw
  nulls:
    family_content:
      preferred_strata:
        - sigma35_variant
        - spacer_length
      preserve_joint_labels: true
    slot_geometry:
      preferred_strata:
        - sigma35_variant
        - spacer_length
        - target_family_count
      preserve_target_family_count: true
      permute_slot_family_mapping: true
  artifact_retention:
    mode: production_review
    prediction_ledger: latest_full_plus_selected_history
    plot_tidy_data: compact
    model_artifacts: latest
    tabular_format: parquet_zstd
    max_estimated_bytes: 50000000000
    fail_if_estimate_exceeds: true
```

### Appendix B: Recommended First Sentinel Labels

```text
lexA_present
cpxR_or_baeR_present
lexA_count_fraction
lexA_in_slot0
cpxR_or_baeR_in_slot2
```

These labels exercise:

- binary family content
- composite family content
- count-fraction ranking
- LexA slot geometry
- CpxR/BaeR slot geometry

### Appendix C: Required Wording In Reports

Use wording like:

> This is a DenseGen metadata-derived synthetic-control benchmark. It tests
> whether OPAL can enrich for declared variable TFBS construction labels from
> X. Fixed sigma-core elements are withheld from active labels and used only
> for passive controls and null strata. Results do not imply measured growth,
> ciprofloxacin response, ethanol tolerance, true TF binding, or mechanism.
