---
doc_id: study-rt-lnrna-sponging-construct-triage-reporter-response-evidence
surface: study-context
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-08-01
measurement_readiness: ready
descriptive_visualization_readiness: ready
reduction_recommendation_status: ready
objective_readiness_status: blocked
---

## Reporter-response evidence

Reader owns experiment-local ingestion, treatment metadata, recorded time,
persisted observation coordinates, explicit replicate declarations, and
generic measurement records. The RT-lnRNA study owns the scientific policy
that selects controls, declares pairing, chooses an endpoint or time window,
and decides whether evidence is comparable enough to support a later
preference objective.

The handoff publishes exactly one descriptive profile variant. When reference
normalization is unavailable, `rt_lnrna_reporter_measurement_profile.v1`
retains raw time-reduced RFP, OD600, and RFP/OD600 summaries plus a typed reason.
When normalization is available, `rt_lnrna_reporter_response_profile.v3`
retains those same raw measurements and adds dose-wise reporter response,
`relative_od`, and supported biological-replicate uncertainty. Both carry
exact Reader record and evidence-binding provenance, declared condition roles,
the ordered dose grid, and within-acquisition observation counts. Neither emits
a score, scalar objective, rank, or OPAL label. There is no active scalar
objective.

The evidence-binding artifact is not a caller-authored metadata bag. Its
builder accepts only a source-closed Reader record, publication derives a
canonical artifact id and digest and never overwrites an existing target, and
the strict loader restores source closure only after exact-field, count,
identity, and digest validation. Downstream profiles must select an exact bound
row from that loaded artifact rather than accept free-form provenance strings.

For an endpoint reduction, `relative_od` is the endpoint OD ratio. It is not
viability or a growth rate and may be interpreted as relative biomass only
after OD linearity and handling effects are validated. No endpoint or growth-
rate objective is currently named or published.

### Comparability and uncertainty

Profiles may be compared or aggregated only when their comparability keys
match and their contract variants support the requested operation. Every
profile embeds a typed
`rt_lnrna_reporter_response_observation_policy.v3`; its digest is derived from
fixed formulas, clipping policy, pairing kind, and reduction semantics rather
than supplied as provenance. The comparability key binds that policy digest,
within-acquisition and biological-replicate
reduction statistics, time-window summary and ratio-reduction order,
uncertainty method, reduction interval or endpoint, and ordered dose grid.
Each Reader experiment is an acquisition. Persisted positions identify
observations within that acquisition and never imply biological or technical
replication. A declared replicate field supplies the only biological-replicate
identifier; absent that field, identity remains unknown. Repeated labels are
scoped by subject and condition. Uncertainty is estimated only over complete,
declared biological-replicate identities.

Profile construction and parsing both require a source-closed Reader
evidence-binding set. The profile selects exactly one bound subject row and
derives the complete Reader protocol, record kind, record path, revision, and
binding-artifact provenance from it.

Profiles without sufficient declared biological-replicate support remain descriptive
and carry an explicit not-estimable reason. A large normalized reporter value
does not override low relative biomass, missing uncertainty, or a mismatched
dose grid.

### Reduction calibration and objective gate

The retrospective meta-study recommends the inclusive 6-10 h interval for
consistent descriptive comparison of the current kinetic cohort. It owns:

- the endpoint or time-window choice and its relationship to recorded Reader
  time, including whether any window-derived slope is valid;
- the tested dose grid and cross-experiment grid-comparability rules;
- control separation, control assignment, and plate/block-effect policy;
- endpoint-OD linearity checks before interpreting `relative_od` as relative
  biomass;
- biological-replicate uncertainty over declared identities and minimum support;
- rank-stability analysis across defensible endpoint/window, dose-grid,
  control, and plate/block choices;
- the distinction between a descriptive recommendation, its limitations, and
  any later objective claim.

The reduction recommendation is ready at evidence grade
`provisional_descriptive`. Objective readiness remains independently blocked:
there is no constrained objective, biological-replicate uncertainty is not
estimable, and OD linearity is not validated. Reader evidence may therefore
support descriptive review, assay QC, canonical Reader plots, and aggregate
study plots without becoming an optimization target.

### Proposed objective, not an active contract

The current objective hypothesis is **Reporter Response Feasibility (RRF)**,
with the prospective identifier `reporter_response_feasibility_v1`. RRF would
replace the retired endpoint-only SPOP formulation. For one predeclared dose
and reduction, it would retain two visible margins:

$$
m_r = \frac{r-r_{\min}}{s_r}, \qquad
m_{OD} = \frac{OD_{rel}-OD_{\min}}{s_{OD}}, \qquad
RRF = \min(m_r, m_{OD}).
$$

Here, $r$ is the study-normalized reporter response and $OD_{rel}$ is the
descriptive relative-OD measurement. The response floor, OD floor, and positive
scales must be fixed before model fitting. Feasibility additionally requires
both raw margins to be nonnegative. A strong reporter response therefore cannot
compensate for a failed OD requirement, and the component margins remain
available for review instead of being hidden inside one score.

RRF is intentionally not registered in OPAL yet. Activation requires a
source-closed response normalization policy, declared floors and scales,
validated OD linearity for the relevant range, and supported biological-
replicate uncertainty. Positive controls may calibrate a declared policy when
present, but no construct alias or inducer name is part of the objective
contract. Until those gates pass, `reporter_response_feasibility_v1` is a
study-owned proposal rather than a label, rank, or optimization surface.
