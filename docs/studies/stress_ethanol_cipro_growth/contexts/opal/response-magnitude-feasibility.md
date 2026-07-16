---
id: stress-ethanol-cipro-growth-response-magnitude-feasibility
title: Response-Magnitude Feasibility study decision
owner: dnadesign-maintainers
status: active
last_verified: 2026-07-15
audience: [scientist, operator, agent]
---

## Response-Magnitude Feasibility

### Decision

Treat `sfxi_v1` and `response_magnitude_feasibility_v1` as distinct objective
contracts. This study applies **RMF** to the promoted Reader reduction under
one repeated-candidate label-source rule and one typed OPAL label contract.
SFXI round-0 records retain their own y-space and are not RMF inputs.

The round-0 policy coordinates six sequence-unique slots for each of three
named selection views evaluated from one shared model fit. A deterministic
round-robin allocator advances a view to its next-best unallocated sequence
when preferred lists overlap. This is a deliberate, time-bounded test of
whether the fixed X representation can hill-climb the declared multistate
objectives. It is not a claim that the fitted model is well calibrated or that
predicted feasibility is measured promoter behavior.

### Premise

A stress-responsive promoter is useful when its intended ON states exceed its
intended OFF states, every intended ON state retains reference-relative
fluorescence, and every intended OFF state stays below the declared
reference-relative fluorescence boundary.

This is the study's operational form of high dynamic range without allowing a
single favorable contrast to hide leak. Response separation rewards the target
state contrast, the OFF ceiling directly penalizes leaky fluorescence, and the
ON floor prevents a uniformly dark promoter from looking selective. The
maximin score is controlled by the weakest requirement.

### Why this study selects RMF rather than SFXI

Canonical SFXI has three relevant mathematical properties:

- its product score lets high effect compensate for weak setpoint fidelity;
- per-design min-max scaling can make a small response span look
  shape-consistent;
- its effect term does not constrain absolute target-OFF fluorescence.

Its round-fitted intensity scaling also prevents direct score comparison across
rounds without preserving the fitted denominator. These properties are
documented in the canonical [SFXI objective](../../../../../src/dnadesign/opal/docs/plugins/objectives/sfxi.md)
and explain the correlated SFXI source selections. They are not software
defects, and exponent tuning alone does not repair the study mismatch.

RMF instead exposes three signed requirements and selects by their minimum. One
strong component cannot compensate for one failed component. The canonical
equations and invariants live in the [RMF objective](../../../../../src/dnadesign/opal/docs/plugins/objectives/response-magnitude-feasibility.md).

### Assay binding

Reader owns the event-relative trajectory reduction. For each assay state
`i`, the promoted handoff provides:

```text
r_i = study-selected experiment value of log2(4-8 h post-event geometric time mean of YFP/CFP)
b_i = the corresponding log2 geometric time mean of YFP/OD600 minus same-state pDual-10
```

The ordered response-window Y is:

```text
[r00, r10, r01, r11, b00, b10, b01, b11]
```

| State | Condition |
| --- | --- |
| `00` | no stress |
| `10` | ethanol |
| `01` | ciprofloxacin |
| `11` | ethanol plus ciprofloxacin |

Here `b_i` is reference-relative fluorescence, not luminance or a general
fluorophore brightness property.

For a binary target mask `p`, RMF computes:

```text
response_separation   = min(r_i where p_i=1) - max(r_i where p_i=0)
on_magnitude_floor    = min(b_i where p_i=1)
off_magnitude_ceiling = max(b_i where p_i=0)
feasibility_margin    = min(q_response, q_on, q_off)
```

The `q` values are signed decision margins around declared boundaries divided
by positive assay-derived scales; they are not classical z-scores. The campaign
scales come from the declared

`exact_primary_reader_candidate_experiments_v1` cohort: 41 exact
candidate-experiment units covering 32 candidates across eight Reader
experiments. This cohort is independent of the retrospective model-screen rows
and repeated-candidate label decisions. Zero is the requirement boundary. Increasing an
ON response, decreasing an OFF response, increasing an ON fluorescence value,
or decreasing an OFF fluorescence value cannot reduce the score for a fixed
mask.

### Selection views

One OPAL campaign owns X, Y, labels, model fitting, predictions, and round
history. Three selection views apply different masks to the same predicted
eight-component phenotype:

| Selection view | Mask `[00,10,01,11]` | Intended ON states |
| --- | --- | --- |
| `ethanol` | `[0,1,0,1]` | ethanol; both stresses |
| `ciprofloxacin` | `[0,0,1,1]` | ciprofloxacin; both stresses |
| `and` | `[0,0,0,1]` | both stresses only |

OR `[0,1,1,1]` remains a pressure-test mask, not an active view. Changing a
mask does not change Reader data or retrain the model. It changes which states
define the ON minima and OFF maxima.

### Evidence and risk

The 35-row retrospective Reader screen is sufficient to probe signed RMF
ordering, but not to establish a reliable phenotype predictor. The approved
exact campaign label corpus contains 27 candidates after repeated-source and
bounded-value exclusions. The working X has 8,192
columns, experiment-held-out ordering is weak, and ethanol and AND have little
observed positive separation. Ciprofloxacin has the strongest retrospective
support. A positive archetype is not required for a negative margin to improve,
but scarce support makes exact top-six identities uncertain.

The first prospective RMF round tests the full method:

```text
fixed sequence X -> shared eight-output model -> view-specific RMF -> coordinated six-slot allocation
```

No numerical probability of success is supported. The credible outcome is a
directional test: whether selected constructs improve measured RMF relative to
the 35-row source corpus and whether each view's nominations outperform the
constructs nominated by the other views under that same mask.

### Frozen round-0 contract

Round 0 froze and recorded:

1. Reader reduction `event_logmean_4_8h_post`.
2. One explicit label-source rule for repeated candidates.
3. The typed eight-column response-window Y schema and Reader bundle digest.
4. RMF thresholds, positive scales, calibration cohort, state order, and three masks.
5. The candidate table, eligibility rules, X column, RF parameters, and seed.
6. Six slots per view, sequence deduplication, round-robin
   next-best-unallocated allocation, and exact expected batch size 18.

The config `src/dnadesign/opal/campaigns/secg_rmf_greedy/configs/campaign.yaml`
accepts labels only through the manifest-pinned study publication. Do not run it
against an ad hoc or reconstructed label table.

The verified round-0 run pinned in the campaign README used 27 exact labels
and one 100-tree RF lineage. It produced six allocations per view and 18
sequence-unique candidates; one preferred overlap required one AND
next-best-unallocated replacement. `model_support_ready` remains false. The
run is a frozen learning probe, and synthesis authorization is a separate study
decision.

### Prospective evidence

Every selected construct is measured in all four assay states, so all 18
constructs update every view. Report:

- predicted versus measured RMF across all 18 constructs for each mask;
- the 27-row promoted source distribution versus the prospective measured round;
- response separation, ON fluorescence floor, OFF fluorescence ceiling, and
  the limiting requirement for each construct;
- whether each view's six outperform the other twelve under that view's mask;
- rank changes and model performance after the prospective response-window Y
  is ingested.

TFBS composition is a provenance and diagnostic surface, not a campaign-specific
eligibility constraint. Predicted RMF alone is not evidence of a responsive
promoter, and successful architecture migration does not authorize synthesis.
