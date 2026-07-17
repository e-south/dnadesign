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
2. **ON expression:** `b_i / s_B` for every `i` in `O`. Higher values mean
   stronger fluorescence relative to pDual-10 measured in that same state.
3. **OFF suppression:** `-b_j / s_B` for every `j` in `F`. Higher values mean
   lower OFF-state fluorescence relative to pDual-10 measured in that same
   state.

Every desired directional change improves one of these coordinates. The
objective does not add a scored `b_ON - b_OFF` term because ON expression and
OFF suppression already contain that information. A displayed ON-to-OFF
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
     + mean(exp(-on_expression_clearance))
     + mean(exp(-off_suppression_clearance))) / 3
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

The score permits bounded compensation. A large improvement in one coordinate
can partly offset a smaller deficit elsewhere. No continuous scalar can be
both strictly increasing in every desired coordinate and completely
noncompensatory. The evidence surface therefore also reports:

- the hard minimum across all coordinates;
- response, ON-expression, and OFF-suppression family scores;
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

Thus `b_i = -3` means eightfold lower fluorescence than pDual-10 in that state.
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
- `fluorescence_scale` is the same quantile of bootstrap standard deviations
  for each state-level `b_i` value.

The 0.90 quantile is a prespecified conservative resolution convention. The
median would describe a typical, relatively clean component and can make
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
ascending candidate ID, with an ordinal ID tie-break. The builder does not
allocate a campaign batch or duplicate OPAL's sequence allocator.

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

The verifier rejects unregistered files, path escape, duplicate JSON keys,
non-finite required values, row or schema drift, duplicate semantic keys,
prediction-run mismatch, and any change from the shadow-only activation
boundary.

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
3. fixed-prediction rank sensitivity and current-versus-proposed Top-K changes;
4. normalization sensitivity as a robustness test, without tuning scales to
   preserve current nominations; and
5. one deliberate campaign disposition that leaves a single executable route.

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
