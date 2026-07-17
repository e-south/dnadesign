---
id: stress-ethanol-cipro-growth-response-magnitude-feasibility
title: Response-Magnitude Feasibility study decision
owner: dnadesign-maintainers
status: active
last_verified: 2026-07-16
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

This is the study's operational direction toward high dynamic range without
allowing a single favorable contrast to hide leak. Response separation rewards
the target-state contrast, the OFF ceiling directly penalizes leaky
fluorescence, and the ON floor prevents a uniformly dark promoter from looking
selective. The maximin score is controlled by the weakest requirement. A
high-dynamic-range feasibility claim additionally requires biologically
meaningful nonzero separation and tight-OFF thresholds; the frozen round-0 zero
boundaries do not establish those effect sizes by themselves.

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

### What the pDual-10 boundary means

The reference subtraction is experiment-local and state-matched:

```text
b_i = log2[(YFP/OD600)_design,i / (YFP/OD600)_pDual-10,i]
```

Therefore `b_i = 0` means that the design and pDual-10 have equal YFP/OD600 in
the same assay condition. It does not mean zero fluorescence, background-like
fluorescence, or absence of expression. State matching remains necessary
because a nominally constitutive reference can change under stress; it prevents
a stressed design from being compared with the reference measured in a
different condition.

The frozen round-0 campaign sets `off_magnitude_max: 0.0`. Its exact OFF claim
is consequently **no brighter than same-state pDual-10**. Lower OFF output
improves `q_off`, but the maximin score stops rewarding that improvement while
response separation or ON fluorescence is the weaker requirement. The current
boundary must not be described as a no-expression or tight-OFF gate.

The selector ranks every candidate by the minimum margin and requires six
allocations per view; it does not filter selections to `S_RMF >= 0`. When a view
has no feasible prediction, OPAL still returns the six least-negative scores.
Therefore `selected` means nominated by the declared policy, not feasible or
biologically ON/OFF.

The generic RMF objective already accepts a stricter study-declared ceiling.
If the allowed OFF output is a fraction `f` of same-state pDual-10, the ceiling
is:

```text
off_magnitude_max = log2(f)
```

For example, `-1`, `-2`, and approximately `-3.322` mean at most one-half,
one-quarter, and one-tenth of same-state pDual-10, respectively. These remain
reference-relative requirements; they do not establish biological background.
The verified round-0 Reader bundle has no annotated promoterless or
reporter-negative cellular control, so it cannot calibrate a background-like
OFF boundary. A media blank is not equivalent to that control.

A future tight-OFF policy must be study-owned and prespecified from the desired
biological claim plus same-host, same-backbone negative-control evidence. Keep
one scalar ceiling only if that boundary is stable across the relevant states;
otherwise use a separately versioned per-state contract instead of collapsing
different boundaries into one number. Changing the ceiling changes RMF scores
and candidate rankings. It therefore requires a new frozen selection contract
and a new model-evidence protocol series; it must not reinterpret the existing
round-0 predictions after measurement.

The same decision discipline applies to response separation. The round-0
`response_separation_min: 0.0` boundary requires no prespecified minimum effect
size. Top-K ranking rewards larger values, but feasibility at the boundary does
not mean high dynamic range. A future requirements policy can state the intended
phenotype in interpretable ratios and compile it into the existing RMF fields:

```text
response_separation_min = log2(minimum ON/OFF YFP/CFP ratio)
on_magnitude_min        = log2(minimum ON output / same-state pDual-10)
off_magnitude_max       = log2(maximum OFF output / same-state pDual-10)
```

Biological thresholds and assay-resolution scales are separate decisions.
Thresholds state what phenotype is acceptable; Reader bootstrap and event-time
evidence estimate how finely the assay resolves each requirement. Do not choose
thresholds by searching for values that preserve the current candidate list.

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
the 27-row promoted pre-round corpus and whether each view's nominations
outperform the constructs nominated by the other views under that same mask.
The 35-row retrospective screen remains diagnostic context, not the campaign's
prospective baseline.

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
constructs update every view. For round 0, and after every eligible measured
batch that retains the same six-per-view, 18-unique-candidate contract, report:

- frozen predicted versus measured response separation, ON fluorescence floor,
  OFF fluorescence ceiling, `q_response`, `q_on`, `q_off`, scalar RMF, and the
  limiting requirement for all 18 constructs under every mask;
- feasibility counts (`S_RMF >= 0`) for the view's nominated six, the other
  twelve, the complete 18, and the promoted pre-round corpus;
- the promoted pre-round distribution versus the prospective measured round;
- whether each view's six outperform the other twelve under that view's mask;
- rank changes and model performance after the prospective response-window Y
  is ingested.

The other twelve are contemporaneous constructs nominated for different views,
not untreated or randomly sampled controls. Their comparison with the nominated
six tests prospective view-specific enrichment within the coordinated batch;
it does not by itself estimate causal treatment effects or population-wide hit
rates.

Before the first measured-batch checkpoint is recorded, the study needs one
digest-bound `stress_ethanol_cipro_growth.prospective_round_evidence.v1`
publication built from the frozen OPAL prediction and allocation records plus
subsequently promoted observations. For the current contract its
candidate-by-view table has 54 rows: 18 measured candidates evaluated under
three masks. Preserve final allocation view and rank, predicted and measured
components and margins, feasibility, limiting requirement, Reader and
label-publication provenance, and campaign prediction and selection digests.
The immutable model-evidence checkpoint references this publication; mutable
OPAL runtime state is not the scientific record.

Across batches, compare checkpoints only within one protocol series. Candidate
membership and measured-corpus size are checkpoint evidence and may grow.
State order, masks, RMF thresholds and scales, inclusion rules, evaluation
methods, and model roles belong to the frozen protocol. Changing any of those
starts a new series rather than silently extending the prior trajectory.

TFBS composition is a provenance and diagnostic surface, not a campaign-specific
eligibility constraint. Predicted RMF alone is not evidence of a responsive
promoter, and successful architecture migration does not authorize synthesis.
