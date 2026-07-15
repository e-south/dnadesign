---
id: stress-response-model-evidence-trajectory
title: Stress response model-evidence trajectory
owner: stress_ethanol_cipro_growth
status: active
last_verified: 2026-07-15
---

# Model-Evidence Trajectory

This harness records how well the stress-study sequence models explain or
predict measured response-window outcomes as the candidate corpus grows. It
preserves each scientific checkpoint without treating OPAL execution state as
model evidence.

## Scientific boundary

Each checkpoint comes from a verified response-metastudy bundle. The snapshot
records:

- label-truth readiness and repeat-aggregation state;
- the configured campaign model, the best fixed challenger, and the baseline;
- all prespecified model-screen outcomes, channel-level X-to-Y performance,
  and per-selection-view ordering;
- configured-campaign greedy-support evidence kept distinct from descriptive
  best-challenger support;
- response-window corpus size and repeated-candidate disagreement;
- retrospective or prospective evidence timing;
- Reader, candidate-binding, measurement-selection, campaign-configuration,
  X-matrix, observed-label, and metastudy digests when present; and
- the four separate decision gates for label truth, model support, selection
  policy, and synthesis authorization.

OPAL initialization, run IDs, rounds, predictions, selections, and ledgers are
operational progress. OPAL owns those records. They are intentionally excluded
from this trajectory so a runnable campaign cannot be mistaken for a supported
scientific decision.

## Comparability contract

The frozen protocol contains the label-truth and repeat-aggregation contract,
target masks, grouped-validation design, metrics, decision thresholds,
calibration estimator settings, model definitions, configured campaign model,
and digests of the evaluator sources.
Its canonical JSON content determines the protocol digest. Fitted review
calibration values are checkpoint results, so the snapshot records them without
mistaking them for protocol definitions.

The evaluator fingerprint is limited to scientific `core/` and `evaluation/`
modules plus the exact runtime and selection-config sources that determine the
screen. Reporting, notebook, publication, and trajectory-storage code cannot
start a new scientific series merely because its presentation or persistence
changed.

Any protocol change creates a new digest and therefore a new series. Results
from different protocol series may be reviewed together, but they are not one
continuous performance curve. A protocol must not be edited in place to make
new evidence appear comparable with prior evidence.

The current response screen is retrospective and nonpromoted. It uses the
screen-only measurement selection and leave-one-Reader-experiment-out
evaluation. A prospective checkpoint requires predictions fixed before the
new assay results are observed. A repeat-aware, candidate-purged evaluation
protocol adopted after observed-label promotion must begin a new series.

## Storage contract

```text
model_evidence/
├── protocols/<protocol_digest>/protocol.json
├── series/<protocol_digest>/checkpoints/
│   └── <evidence_id>__<checkpoint_digest>/checkpoint.json
├── latest.json
└── catalog.json
```

Protocol and checkpoint directories are create-only. Recording the same
evidence ID with identical content is idempotent. Reusing an evidence ID with
different content in the same protocol series fails closed. `latest.json` is a
replaceable convenience pointer. `catalog.json` is a replaceable index rebuilt
only from verified immutable records; neither file is scientific authority.
The catalog exposes the model-screen candidate count and compact campaign and
challenger summaries so progress can be inspected without reopening every
checkpoint. Those values are observations, not a promise of monotonic model
improvement.

## Commands

From the `dnadesign/` repository:

```bash
uv run python -m \
  dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.model_evidence \
  record \
  --metastudy-bundle /path/to/verified/response_metastudy \
  --trajectory-root /path/to/model_evidence \
  --evidence-id pre_batch0_retrospective \
  --json

uv run python -m \
  dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.model_evidence \
  verify --trajectory-root /path/to/model_evidence --json

uv run python -m \
  dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.model_evidence \
  rebuild-catalog --trajectory-root /path/to/model_evidence --json
```

Record a checkpoint only after the metastudy bundle is complete. The recorder
verifies every artifact named by that bundle before projecting the checkpoint.
Passing trajectory verification establishes record integrity; it does not
promote a model, authorize selection, or authorize synthesis.

## Per-batch review

Use one evidence ID per eligible measured corpus, for example
`pre_batch0_retrospective`, `batch0_prospective`, and `batch1_prospective`.
Before recording, verify that predictions for a prospective checkpoint were
fixed before its measurements were observed. After recording, compare within a
single protocol series:

1. candidate and held-out group support;
2. median and limiting channel-level X-to-Y Spearman correlation;
3. response-magnitude error;
4. weakest ordering for each required selection view; and
5. configured campaign-model greedy support.

Treat these as a trajectory, not a requirement that every batch improve every
number. Low-sample estimates can move sharply as new response regimes enter the
corpus. A durable learning claim needs prospective checkpoints with improving
or stable held-out behavior, adequate group support, and no degradation hidden
by switching protocols or models.
