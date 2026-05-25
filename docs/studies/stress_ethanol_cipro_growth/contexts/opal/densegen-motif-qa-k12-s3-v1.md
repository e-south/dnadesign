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
- Active label family: `sfxi_axis_vec8`
- Passive readout label families:
  - `tf_family_presence`
  - `tf_family_count`
  - `densegen_plan_class`
- Primary null: global quality-ok permutation of the active label family

The first active suite remains the SFXI vec8 campaign matrix over cipro,
ethanol, and dual objectives. TF-family presence/count and DenseGen plan class
are manifest-backed passive readouts over the same selected IDs. They are not
promoted to independent active OPAL campaigns until the K12/S3 SFXI suite shows
that extra active runs would add decision value.

### Label Families

`sfxi_axis_vec8` is the active synthetic label family. It is derived from
`densegen__used_tfbs_detail`: LexA contributes the cipro axis, CpxR/BaeR
contribute the ethanol axis, and both axes form the dual/AND target. The vec8
is a binary SFXI-compatible proxy, not a measured effect size.

`tf_family_presence` and `tf_family_count` are passive readouts derived from
the same TFBS detail. They expose LexA, CpxR, and BaeR presence and counts so
the selected candidates can be audited against the underlying motif composition
without adding study semantics to OPAL core.

`densegen_plan_class` is a part-derived plan-class proxy. The raw
`densegen__plan` string remains audit metadata and sigma35 split metadata; it
is not the primary label source.

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

One-seed burn-in:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe run \
  --gate all --selection-k 12 --initial-labels 12 --rounds 12 --seed 7 \
  --score-batch-size 512 --apply --json
```

Repeat the burn-in command for seeds 17 and 29 only after seed 7 artifact and
trajectory QA review is mechanically clean.

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

The suite review verifies seed coverage, 12 scored campaign/control/split rows
per seed, final round 11, 144 round metrics per seed, positive/null pair
coverage, configured plot quality, nested stale review warnings, null-spike
diagnostics, and trajectory min/mean/max summaries.
