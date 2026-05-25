## DenseGen Motif QA K12/S3 v1

`densegen_motif_qa_k12_s3_v1` is the current planned study-owned OPAL QA
suite for the pre-assay stress ethanol/cipro campaign. It replaces the stale
K6/single-seed probe framing with a K12, three-seed, trajectory-based benchmark.

The suite asks a bounded methods question: can the configured OPAL campaign
loop, using the current LatentDNA/Evo2 X column, recover known
motif-composition structure that was intentionally introduced by DenseGen? It
does not claim growth phenotype, ethanol tolerance, ciprofloxacin response, or
biological mechanism.

### Boundary

OPAL stays campaign-agnostic. It owns candidate-table validation, label-source
ingest, model fitting, scoring, selection, ledgers, configured plots, review
manifests, and generated notebooks. The study probe owns DenseGen-specific
synthetic labels, suite manifests, positive/null pairing, passive label-family
readouts, and aggregate benchmark prose.

Study-specific aggregate plots remain in the probe report under `.var/...` and
`reports/`. They enter canonical OPAL notebooks only if promoted through the
generic OPAL plot registry contract: `PlotMeta`, `PlotContext`, configured
`plots.yaml`, media/tidy outputs, and `opal.plot_artifact.v1` manifests.

### Suite Contract

- Suite id: `densegen_motif_qa_k12_s3_v1`
- Selection K: 12
- Initial labels: 12
- Seeds: 7, 17, 29
- Rounds: 12
- Splits: `random_id`, `leave_sigma35_variant`
- Active label families:
  - `densegen_plan_logic4`
  - `tf_family_count`
- Passive readout label families:
  - `tf_family_presence`
  - `densegen_plan_class`
- Primary null: global quality-ok joint permutation of the active label-family
  values across sequence IDs

The full suite prepares two active campaign matrices:

- DenseGen plan logic4: `3 targets x 2 oracle kinds x 2 splits x 3 seeds = 36`
  campaigns.
- Compact TF count: `3 count objectives x 2 oracle kinds x 2 splits x 3
  seeds = 36` campaigns.

Per seed, `--gate all` therefore prepares 24 OPAL campaigns: 12 plan-logic4
campaigns and 12 TF-count campaigns. Across seeds 7, 17, and 29, the full
planned suite is 72 campaigns.

The next expansion promotes selected passive families into study-owned active
campaign variants only through OPAL's generic label/objective contract. OPAL
must see them as numeric scalar or vector targets with declared score channels;
the study keeps the DenseGen oracle, null construction, biological names, and
decision interpretation.

### Runtime Footgun Controls

The source candidate table is a shared 157,160-row, 8192-dimensional
`records.parquet` artifact and is roughly 5 GiB on disk. Probe materialization
must not physically copy that table per split, seed, or campaign. Each split
dataset instead carries:

- a `records.parquet` symlink to the shared candidate table;
- a `candidate_scope_ids.parquet` file with the scoped train/eval candidate IDs;
- campaign-local observed-label sidecars and OPAL outputs.

Generated campaign configs declare `data.candidate_scope` so OPAL scores the
split-specific ID universe while streaming X from the shared records table.
This keeps the review endpoints unchanged while avoiding tens of GiB of
duplicated input parquet across the three-seed suite.

Configured selected/top-k plots should rely on OPAL's lazy ledger filters. In
particular, selected score trajectories and selected vector heatmaps must not
collect all candidate-round vector predictions only to discard non-selected
rows in pandas.

### Label Families

`densegen_plan_logic4` is the primary synthetic control label family. It is
derived from `densegen__used_tfbs_detail`: LexA contributes the cipro axis,
CpxR/BaeR contribute the ethanol axis, and both axes form the dual/AND target.
The label is a four-channel DenseGen plan-logic vector in state order
`v00`, `v10`, `v01`, `v11`. It is not a measured SFXI assay, not an SFXI
projection diagnostic, and not a separate biological claim from the DenseGen
plan class. OPAL sees it through the generic `vector_from_table_v1` transform
and the generic `vector_target_similarity_v1` objective.

`tf_family_count` is the second active probe family. It asks a distinct
learnability question: can OPAL recover motif-count structure from the current
X representation? The compact active objectives are:

- `tf_count__lexA`: cipro-like motif-count target.
- `tf_count__cpxR_plus_baeR`: ethanol-like motif-count target.
- `tf_count__lexA_plus_cpxR_plus_baeR`: dual-like motif-count target.

The active OPAL target is a three-channel numeric vector produced by
`vector_from_table_v1`, with selection controlled by the declared
`vector_channel_v1` score channel for the current campaign target.

`tf_family_presence` remains a passive readout for now. It is useful as a
thresholded count audit, but count is the richer first active probe.

`densegen_plan_class` is a part-derived plan-class proxy. The raw
`densegen__plan` string remains audit metadata and sigma35 split metadata; it
is not the primary label source. In this synthetic setup it is a categorical
compression of the same DenseGen plan-logic state represented by
`densegen_plan_logic4`, so it stays passive unless the study later asks a
specific coarse-class learnability question. If promoted, it must first be
encoded as explicit one-vs-rest numeric columns.

### Targets, Oracles, Seeds, And Splits

For DenseGen plan logic4, the three targets are the intended axis objectives:

- `cipro`: recover ciprofloxacin-axis structure, mainly LexA-associated.
- `ethanol`: recover ethanol-axis structure, mainly CpxR/BaeR-associated.
- `dual`: recover the joint/AND-like axis structure.

For TF counts, the analogous compact objectives are:

- `tf_count__lexA`: cipro-like motif-count target.
- `tf_count__cpxR_plus_baeR`: ethanol-like motif-count target.
- `tf_count__lexA_plus_cpxR_plus_baeR`: dual-like motif-count target.

The positive/intact oracle is a deterministic synthetic label derived from
DenseGen TFBS metadata for the same sequence ID. The null/permuted oracle keeps
the same label distributions but scrambles the label values relative to
sequence IDs. The null is not random labels from nowhere; it preserves marginal
structure while breaking the representation-to-label relationship.

Seeds provide deterministic replicate fidelity. They change deterministic
choices such as initial labeled batch and permutation order. Positive-vs-null
separation across seeds is more credible than one lucky trajectory.

Splits test generalization regime, not replication:

- `random_id` is the easier interpolation-like learnability question: can OPAL
  learn the synthetic label structure when train/evaluation candidates come
  from the same broad design distribution?
- `leave_sigma35_variant` is the harder structured split: can OPAL recover the
  motif/axis signal when evaluation candidates differ along a held-out DenseGen
  sigma35 design stratum?

If positives separate from nulls on `random_id` but collapse on
`leave_sigma35_variant`, the interpretation is that OPAL can exploit local
representation signal, but that signal may not generalize across the held-out
DenseGen design axis.

### Active Label-Family Expansion

The expansion is a campaign-matrix decision, not a change to OPAL's campaign
identity. Each active variant is still one OPAL campaign: one config, one label
source, one numeric Y contract, one model/objective/selector chain, and one set
of ledgers and plot manifests.

Initial supported mappings:

- `tf_family_presence`: LexA, CpxR, and BaeR binary columns can be run as
  scalar one-channel campaigns or as a finite vector target with a declared
  channel objective.
- `tf_family_count`: the default active variant runs the compact finite count
  vector with declared objective channels for LexA, CpxR+BaeR, and
  LexA+CpxR+BaeR.
- `densegen_plan_class`: plan-class activity must be encoded explicitly, for
  example one-vs-rest numeric columns. OPAL should not ingest raw DenseGen plan
  strings as objective labels.

Both `densegen_plan_logic4` and `tf_family_count` use generic OPAL plots, such
as score over rounds, score versus rank, feature importance, vector summary
heatmaps, selected-label enrichment, and scalar/vector distribution diagnostics.
SFXI-specific diagnostics are intentionally omitted from this synthetic control
probe. Real SFXI assay campaigns can enable SFXI objective and plot plugins
later, when the labels are measured SFXI values rather than DenseGen-derived
control labels. Any study-specific aggregate interpretation belongs in the
probe review unless it is promoted through the OPAL plot registry and emits
normal plot manifests.

### Campaign Collection Semantics

Positive/null review is collection-level, not a campaign-local appendix. The
study emits an optional `opal.campaign_collection.v1` manifest with
relationship roles and match dimensions. OPAL renders generic paired comparison
surfaces from that manifest and does not infer controls from campaign names.

For multi-seed interval bands, seed must be declared as replicate structure in
the collection relationship. Until then, notebook comparison bands are labeled
as IQR across validated campaign-set comparison units, not statistical
confidence intervals.

### QA Metrics

The old `null_lift > 1.25` STOP gate is removed. Null lift is still reported,
and null spike/drop behavior remains a non-blocking diagnostic, but the primary
QA read is paired positive-vs-null trajectory separation:

- positive lift AUC
- null lift AUC
- paired AUC delta
- final positive-minus-null lift delta
- seed-level mean/min/max summaries, with `n=3` stated plainly

A successful QA result is positive trajectories exceeding paired null
trajectories by AUC and final lift across the scoped campaign/split/seed pairs.
A debug result means the active-learning harness or representation space needs
more review before using this probe as a methods figure.

### Commands

Source/materialization dry run:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe run \
  --gate source --selection-k 12 --initial-labels 12 --seed 7 --json
```

Apply the source gate and write the suite, label-family, and null provenance
manifests:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe run \
  --gate source --selection-k 12 --initial-labels 12 --seed 7 --apply --json
```

Validate the full per-seed campaign matrix without scoring the candidate pool:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe run \
  --gate all --selection-k 12 --initial-labels 12 --rounds 12 --seed 7 \
  --score-batch-size 512 --stop-after validate --apply --json
```

This should plan 24 campaigns for one seed: 12 plan-logic4 and 12 TF-count.

One-seed scored burn-in:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe run \
  --gate all --selection-k 12 --initial-labels 12 --rounds 12 --seed 7 \
  --score-batch-size 512 --apply --json
```

Repeat the scored burn-in command for seeds 17 and 29 only after seed 7
artifact freshness, configured plots, trajectory QA, and null diagnostics are
mechanically clean.

To run a lower-cost scope before the full 72-campaign suite, constrain the
active families explicitly:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe run \
  --gate all --active-label-families densegen_plan_logic4 --selection-k 12 \
  --initial-labels 12 --rounds 12 --seed 7 --score-batch-size 512 --apply --json
```

or:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe run \
  --gate random-all --active-label-families tf_family_count --selection-k 12 \
  --initial-labels 12 --rounds 12 --seed 7 --score-batch-size 512 --apply --json
```

Completion requires one mechanically clean `--gate all --rounds 12` root for
each suite seed. `status --json` is intentionally strict for scored plans: a
materialized root without metrics, decision, and final-round coverage is
`attention`, not a completed probe.

After configured plots and per-root reports are refreshed, write the suite
aggregate:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe suite \
  --run-root .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_motif_qa_k12_s3_v1_seed7_all_r12 \
  --run-root .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_motif_qa_k12_s3_v1_seed17_all_r12 \
  --run-root .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/densegen_motif_qa_k12_s3_v1_seed29_all_r12 \
  --out-dir .var/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe/_suite_reviews/densegen_motif_qa_k12_s3_v1_all_r12 \
  --json
```

The suite review verifies seed coverage, 24 scored campaign/control/split/family
rows per seed, final round 11, 288 round metrics per seed, positive/null pair
coverage, configured plot quality, nested stale review warnings, null-spike
diagnostics, and trajectory min/mean/max summaries.
