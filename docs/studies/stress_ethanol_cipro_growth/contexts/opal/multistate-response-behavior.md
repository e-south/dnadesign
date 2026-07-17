---
id: stress-ethanol-cipro-growth-multistate-response-behavior
title: Multistate Response Behavior shadow protocol
owner: stress_ethanol_cipro_growth
status: evaluation
last_verified: 2026-07-17
audience:
  - scientist
  - maintainer
  - operator
  - agent
---

## Multistate Response Behavior

### Decision status

`multistate_response_behavior_v1` is a shadow objective. It is available for
read-only scoring and comparison, but no checked-in campaign may use it while
the study protocol remains `shadow_only`. It does not authorize synthesis.

The persisted study contract is
`src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/config/multistate_response_behavior_shadow_v1.yaml`.
That file fixes the assay binding, target masks, normalization derivation,
evidence roles, and activation boundary. The pure objective mathematics live
in OPAL's canonical objective source. The study does not duplicate those
equations or implement a second scorer.

This evaluation does not alter or reinterpret
`response_magnitude_feasibility_v1`. RMF remains the signed feasibility-margin
contract for the completed round-0 audit record. A later promotion decision
must leave exactly one executable study campaign.

### Biological specification

For ordered assay states `i`, Reader supplies two values:

```text
r_i = reduced log2(YFP/CFP) response
b_i = same-state pDual-10-relative log2(YFP/OD600) fluorescence
```

For a binary target mask, `O` contains intended ON states and `F` contains
intended OFF states. The behavior objective has three requirement families:

1. **Response ordering:** `(r_i - r_j) / s_R` for every `i` in `O` and `j` in
   `F`. Higher values mean that an intended ON response exceeds an intended
   OFF response.
2. **ON signal:** `b_i / s_signal` for every `i` in `O`. Higher values mean
   stronger YFP/OD600 fluorescence signal relative to pDual-10 measured in that
   same state.
3. **OFF signal suppression:** `-b_j / s_signal` for every `j` in `F`. Higher
   values mean lower OFF-state YFP/OD600 fluorescence signal relative to
   pDual-10 measured in that same state.

Every desired directional change improves one of these coordinates. The
objective does not add a scored `b_ON - b_OFF` term because ON signal and OFF
signal suppression already contain that information. A displayed ON-to-OFF
fluorescence contrast may help interpretation, but scoring it again would
double-count both fluorescence requirements.

The stress study fixes state order and masks as follows:

| View | Mask `[00, 10, 01, 11]` | Intended ON states |
| --- | --- | --- |
| ethanol | `[0, 1, 0, 1]` | ethanol; both stresses |
| ciprofloxacin | `[0, 0, 1, 1]` | ciprofloxacin; both stresses |
| AND | `[0, 0, 0, 1]` | both stresses only |

The OPAL objective accepts any ordered set of at least two states with at least
one ON and one OFF state. Factor names and a complete factorial design are not
part of its ontology. The four stress states and their biological names remain
study-owned.

### Selector score and diagnostics

The score takes a smooth bottleneck within each family, then gives the three
families equal one-third prior weight:

```text
behavior_score = -log[
    (mean(exp(-response_clearance))
     + mean(exp(-on_signal_clearance))
     + mean(exp(-off_signal_suppression_clearance))) / 3
]
```

The normalized temperature is fixed at one assay-resolution unit. It is not a
campaign tuning parameter.

The derivative with respect to every clearance is positive. Raising any ON
response, lowering any OFF response, raising any ON fluorescence value, or
lowering any OFF fluorescence value therefore increases the score when the
other measurements stay fixed. Poor coordinates receive exponentially larger
bottleneck weights, so one weak behavior still matters more than an already
strong behavior.

Strict monotonicity is a real-arithmetic property. At finite-precision
extremes, already favorable coordinates can saturate or receive a numerically
zero bottleneck weight. Runtime pressure tests therefore require a finite,
nondecreasing score at extremes rather than a distinguishable increase after
every arbitrarily large separation.

The score permits bounded compensation. A large improvement in one coordinate
can partly offset a smaller deficit elsewhere. No continuous scalar can be
both strictly increasing in every desired coordinate and completely
noncompensatory. The evidence surface therefore also reports:

- the hard minimum across all coordinates;
- response, ON-signal, and OFF-signal-suppression family scores;
- every state-level clearance and its bottleneck weight;
- the limiting state or state pair; and
- whether every reference direction is nonnegative.

The hard minimum is a diagnostic, not a selector. A positive
`behavior_score` does not mean feasible, and
`all_reference_directions_met=true` is a natural-zero diagnostic rather than a
biological acceptance claim.

### What pDual-10 establishes

Reader defines fluorescence relative to pDual-10 in the same experiment and
assay state:

```text
b_i = log2[(YFP/OD600)_design,i / (YFP/OD600)_pDual-10,i]
```

Thus `b_i = -3` means eightfold lower YFP/OD600 fluorescence signal than
pDual-10 in that state.
The behavior score continues to reward `-4`, `-5`, and lower OFF values without
introducing an acceptance threshold.

pDual-10 does not measure reporter background or prove absence of expression.
The strongest supported phrase is **OFF suppression relative to same-state
pDual-10**. Documentation, plots, and tables must not translate this evidence
into “no expression,” “absolute OFF,” or a background-level claim.

### Assay-resolution normalization

The shadow protocol derives two scales from Reader's joint well-bootstrap
draws over the existing exact primary candidate-experiment cohort:

- `response_scale` is the declared 0.90 quantile of bootstrap standard
  deviations for the union of state pairs that appear as ON-versus-OFF pairs
  in any declared view;
- `signal_scale` is the same quantile of bootstrap standard deviations for
  each state-level reference-relative `b_i` value.

The 0.90 quantile is a conservative resolution convention frozen in the
protocol before shadow ranking and prospective use. It was formulated after
the existing assay corpus was available, so it is not described as prospective
preregistration. The median would describe a typical, relatively clean
component and can make
noisier but valid assay states look artificially far apart after scaling. The
maximum would let one unstable component or experiment set the scale for the
entire study. The 0.90 quantile represents the noisier part of the verified
assay without giving a single extreme unlimited influence. It is not inferred
from which candidates rank well. Alternative quantiles belong in a declared
sensitivity analysis and must not be tuned to preserve preferred nominations.

State-pair direction does not change its standard deviation. A pair used by
more than one view is therefore counted once per candidate-experiment unit,
preventing duplicated masks from silently reweighting the normalization. The
quantile method is fixed to `linear`, every unit must provide the same number
of draws, at least 100 draws are required, and both derived scales must be
finite and positive.

These are measurement-resolution conventions, not biological thresholds. The
derivation does not reuse RMF's view-specific extrema scales. A shadow
normalization record pins the protocol digest, Reader bundle and request
digests, candidate-binding and observation-policy digests, source-row digest,
cohort identity, draw count, quantile, pair rule, and both resulting values.
Scoring fails when observed rows or bootstrap draws do not reproduce the exact
cohort that fixed those scales.

### Uncertainty and data-quality evidence

Uncertainty does not become another coordinate in the response-window vector
or the behavior selector. Four evidence layers remain distinct:

- **Within-unit joint bootstrap:** derives the two resolution scales and supports
  candidate-experiment-unit rank sensitivity analysis. It does not support a
  candidate-level Top-K statistic before repeated experiments have a declared
  aggregation policy. Each Reader draw preserves the multichannel dependence
  within one design/state reduction. Draw indices aligned across different
  candidates form an independent-product Monte Carlo sample: they do not share
  pDual-10 resampling indices and are not an experiment-level shared-anchor
  bootstrap. Rank intervals may therefore be conservative when common-reference
  variation would otherwise cancel.
- **Event-time sensitivity:** midpoint-centered component half-ranges construct
  desirable-direction worst and best score envelopes. These are conservative
  componentwise bounds, not jointly observed event draws, confidence
  intervals, or probabilistic standard deviations. Central, worst-envelope,
  and best-envelope candidate-experiment-unit ranks are reported with their
  span.
- **Repeated experiments:** candidate/view score minima, maxima, ranges,
  limiting coordinates, and contributing experiment IDs are reported without
  aggregating labels or choosing a source.
- **Censoring:** bounded or overflow-derived components remain visible and are
  excluded from the exact normalization cohort. A finite clipped value must
  not look like exact phenotype evidence.

The shadow scorer applies one frozen protocol to observed rows, all Reader
joint-bootstrap draws, and a fixed prediction matrix. It emits score tables,
state-level coordinate tables where requested, event envelopes, repeat
agreement, observed candidate-experiment-unit rank sensitivity, and aligned
rank/Top-K comparisons against an explicitly named hard score on the fixed
prediction surface. Prediction ranking uses descending score followed by
ascending candidate ID, with an ordinal ID tie-break. A read-only allocation
preview calls OPAL's public round-robin next-best-unallocated runtime and
deduplicates by exact sequence. The study does not duplicate the allocator and
does not mutate a campaign.

The registered random forest is trained against raw eight-component
response-window Y. Both RMF and the behavior objective score the same predicted
Y afterward. Changing the scalar objective can change ranking and allocation;
it cannot, by itself, make the sequence-to-Y relationship easier to predict.
Any claim that the behavior objective hill-climbs better therefore requires
predictions frozen before new measurements over prospective rounds.

Family means prevent a target mask with more response pairs from receiving
more prior weight solely because it has more coordinates. They also create a
cardinality boundary. Numeric behavior scores are view-local: comparison
requires the same ordered state IDs, target mask, normalization record, and
protocol. A shared state count or state list is insufficient because different
masks create different coordinate families and priors. As the number of
otherwise strong coordinates grows, one fixed weak coordinate can receive less
total soft-bottleneck weight. The persisted pressure table reports balanced
`K=2,4,8,16` analytic maximum soft-over-hard gaps of 1.10, 2.48, 3.87, and 5.26
assay-resolution units. In this study, ethanol and ciprofloxacin response
coordinates have prior `1/12`, giving `log(12)=2.485`; AND response and OFF
coordinates have prior `1/9`, giving `log(9)=2.197`. The hard bottleneck,
limiting coordinate, family scores, and coordinate weights disclose this
compensation but do not constrain selection. A materially negative coordinate
can coexist with a positive selector score. The semantic GO is limited to
within-view ranking under the fixed four-state protocol; it is a NO-GO for
conformance, feasibility, cross-view scalar comparison, or cross-state-space
comparison.

The objective does not encode growth, viability, or expression burden.
YFP/OD600 denominator safeguards do not turn a high behavior score into growth
compatibility evidence. Reader trajectories and study QC retain that evidence;
growth is not added as a fourth score family.

### Persisted evidence bundle

The study publisher writes
`stress_ethanol_cipro_growth.multistate_response_behavior_shadow_bundle.v1`
atomically. Its manifest binds the protocol, exact Reader cohort, candidate
bindings, observation policy, fixed OPAL prediction run, and every table by
path, byte count, SHA-256 digest, row count, and ordered columns. The bundle
contains:

- both normalization-resolution tables and the complete bootstrap score table;
- observed scores and state-level coordinates;
- bootstrap rank support, event envelopes, repeat disagreement, and censor
  exclusions;
- fixed-prediction scores; and
- aligned RMF-versus-behavior raw rank evidence.

The completion gate adds the smallest evidence set needed for adjudication:

- `normalization_sensitivity.parquet`: q50, q75, q90, q95, q99 and
  leave-one-source-experiment-out scale sensitivity on one fixed prediction
  matrix;
- `grouped_objective_validation.parquet`: five-seed,
  leave-one-label-source-experiment-out raw-Y random-forest predictions, with
  fold-local behavior and RMF parameters and explicit rank-undefined groups;
- `rmf_replay_calibration.parquet`: zero-threshold RMF scales replayed from the
  corrected Reader uncertainty evidence rather than stale campaign scales;
- `allocation_comparison.parquet`: sequence-unique public-runtime allocation
  previews for both objectives;
- `observed_control_face_validity.parquet`: SpyP ethanol and sulAp
  ciprofloxacin score/rank diagnostics, with no positive AND control claim;
- `family_cardinality_pressure.parquet`: analytic and realizable family
  dilution pressure at `K=2,4,8,16`;
- `grouped_rmf_resolution.parquet` and `source_equivalence.json`: portable
  corrected-Reader calibration and exact point-label reuse receipts;
- `decision.json`: a typed split verdict for semantic fit, implementation,
  normalization robustness, predictive support, prospective efficacy,
  technical readiness, campaign disposition, and synthesis;
- `report.md`, one independent adversarial implementation-audit record, and
  three minimal plots for normalization robustness, grouped validation, and
  allocation-preview family decomposition.

The audit record names the stable Codex subagent auditor, UTC completion time,
full reviewed source commit, and SHA-256 digest of the preliminary bundle that
the independent pass inspected. Publication and verification reject any drift
in those four fields. This provenance identifies the exact automated review
snapshot; it is not a human or external peer review, a cryptographic signature,
or evidence of prospective hill-climb efficacy.

The verifier rejects unregistered files, path escape, duplicate JSON keys,
non-finite required values, row or schema drift, duplicate semantic keys,
prediction-run mismatch, evidence-digest drift, non-replayable fold parameters
or scores, allocator divergence, report drift, and any change from the
shadow-only activation boundary.

### Source correction and label reuse

The shadow protocol binds Reader manifest
`bdc6f7afc8b00a7960eac7bf402be2632f9815c7fedc88cae4c27d47e9d09418`.
It requires pDual-10 compared with the same resample to have exactly zero
central `b_i`, bootstrap standard deviation, and every joint-bootstrap `b_i`
draw. The immutable observation and label artifacts remain bound to their
prior Reader manifest and remain independently verifiable.

The shadow loader does not overwrite or reinterpret those artifacts. It selects
the same candidate and Reader-experiment identities from the corrected bundle
and requires bit-exact equality of all eight central values before reusing a
promoted label. The bundle records a central-equivalence digest. Because the
correction changes reference-bootstrap uncertainty but not central labels, a
new observation version is not required for point-label reuse. The shadow
evidence is a new source-bound publication because its uncertainty scales and
source digests changed.

The immutable label directory retains a digest-identical copy of the source
observation manifest as a provenance receipt; it does not duplicate the source
Parquet records. The shadow protocol therefore declares the study-owned source
observation bundle path explicitly. The loader requires that bundle's manifest
digest to equal the immutable receipt before it reads any record, then confines
every record path and verifies its digest within the declared study bundle.

Preview without writing:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.multistate_behavior_cli \
  preview \
  --reader-bundle ../reader/outputs/reviews/stress_response_window/latest \
  --candidate-bindings src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/promoter_candidate_bindings/latest \
  --prediction-run-id <exact-opal-run-id>
```

Replace `preview` with `publish` to materialize the default study output, or
use `verify --bundle <path>` to verify an existing publication. Publishing
evidence does not activate a campaign or authorize synthesis.

### Promotion gate

Promotion requires an explicit study adjudication after the digest-bound
shadow evidence is materialized. Review must cover:

1. strict directional monotonicity, Pareto dominance, state-permutation
   equivariance, extreme finite inputs, and family-cardinality pressure tests;
2. observed, bootstrap, event-bound, repeated-candidate, and censor evidence;
3. fixed-prediction objective disagreement and current-versus-proposed Top-K
   changes, kept distinct from grouped prediction-to-truth evidence;
4. normalization sensitivity as a robustness test, without tuning scales to
   preserve current nominations; and
5. an independent adversarial implementation audit with no unresolved blockers;
6. one deliberate campaign disposition that leaves a single executable route.

The persisted decision is deliberately split. Semantic fit and shadow
implementation may be `go` while predictive support remains insufficient,
prospective hill-climb efficacy remains unproven, campaign activation remains
`no_go`, and synthesis remains prohibited. A better-behaved scalar is not, by
itself, evidence for a better predictor or a better prospective campaign.

While the protocol remains `shadow_only`, an architecture test rejects any
checked-in campaign config that names `multistate_response_behavior_v1`.
Changing that guard, the protocol status, and a campaign config must be one
reviewed promotion decision. Synthesis authorization remains separate.

### Code and contract map

- Study protocol loader:
  `response_metastudy/evaluation/multistate_behavior_protocol.py`
- Study scale derivation and digest-bearing record:
  `response_metastudy/evaluation/multistate_behavior_normalization.py`
- Observed, bootstrap, event, repeat, prediction, and rank evidence:
  `response_metastudy/evaluation/multistate_behavior_shadow.py`
- Fixed OPAL prediction-run verification:
  `response_metastudy/runtime/multistate_behavior_prediction.py`
- Atomic publication and verification:
  `response_metastudy/runtime/multistate_behavior_publication.py`
- Operator entrypoint:
  `response_metastudy/multistate_behavior_cli.py`
- Canonical OPAL objective:
  `src/dnadesign/opal/docs/plugins/objectives/multistate-response-behavior.md`
