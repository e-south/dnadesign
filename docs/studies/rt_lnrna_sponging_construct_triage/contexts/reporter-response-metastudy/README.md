---
doc_id: study-rt-lnrna-reporter-response-metastudy
surface: study-context
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-08-01
parent_route: ../../routes/reporter-response-evidence.md
bridge_route: rt_lnrna_reporter_response_metastudy
data_plane_role: retrospective-descriptive-reduction-selection
current_state: provisional-descriptive-reduction-selected
measurement_readiness: ready
descriptive_visualization_readiness: ready
reduction_recommendation_status: ready
objective_readiness_status: blocked
entry_artifacts:
  - source-closed-reporter-response-profiles
  - derivation-closed-profile-audits
  - owner-bound-live-readiness-receipt
exit_artifacts:
  - protocol.yaml
  - metastudy-state.yaml
---

## Reporter-response window meta-study

This surface selects one standard time-series reduction for comparing the
currently measured RT-lnRNA variants. It is a retrospective calibration over
the available kinetic experiments, not a prospective efficacy claim and not an
optimization objective.

The current recommendation is the inclusive **6–10 h window**, graded
`provisional_descriptive`. It places the reduction after early growth while the
cohort is still transitioning from active to decelerating growth. Earlier
windows end during stronger growth; later windows approach the plate-run tail,
where plateau and handling artifacts are more plausible. The recommendation is
therefore suitable for consistent descriptive comparisons and aggregate plots,
with its limitations retained explicitly.

### Four independent states

| State | Current result | Meaning |
| --- | --- | --- |
| Measurement readiness | `ready` | All eight selected Reader experiments resolve to verified measurement records. |
| Descriptive visualization readiness | `ready` | Reader can render growth, reporter, reporter/OD, and the selected interval through its canonical diagnostic path. |
| Reduction recommendation | `ready` | The study recommends 6–10 h for descriptive variant comparison. |
| Objective readiness | `blocked` | No constrained objective is defined; biological-replicate uncertainty coverage is incomplete; OD linearity is not validated. |

### Cohort boundary

The workspace bridge catalogs 19 retron-adjacent Reader experiments; that is
the navigation census, not the calibration cohort. This meta-study selects
eight comparable kinetic, single-reporter RT-lnRNA dose-series acquisitions:
three from 2025 and five from 2026. The remaining routes stay available for
their own questions without silently broadening this protocol.

- `20250512_retron_panel_M9_glu` and `20250618_sensor_panel_M9_glu` are
  dual-reporter panels with different subject and condition systems.
- `20251105_retron_Eco1_RT_variants` is `snapshot_only`; it can provide
  descriptive context but cannot select a time window.
- `20260727_retron_Eco1_26_D01_D02_P01_P03_DP01_DP03_benchmark` tests a
  separate competence subject set and is the validation route for explicitly
  declared biological-replicate IDs.
- Cytometry and construct-design experiments remain program context rather
  than plate-reader window evidence.

Adding an experiment requires an explicit protocol and bridge-registry change;
calendar year, a retron label, or one shared control is not sufficient
admission evidence.

A missing subject/window coordinate is recorded as a typed omission. It does
not discard other valid subjects from the same experiment. Missing retron26 or
another named reference limits reference-panel analysis; it does not make the
remaining measurements unusable. Positive-control separation is evaluated
where the declared condition roles provide it and is not inferred from a
particular inducer name or construct alias.

### Acquisition projection

Per-acquisition profiles remain the immutable evidence layer. After selecting a
window, the meta-study adds one descriptive projection for that reduction,
keyed by subject, dose condition, and exact Reader acquisition ID.
Candidate-window profiles stay
in decision evidence; they are not published as co-equal downstream results.
Reader experiments, plates, sheets, wells, and positions remain acquisition
provenance. They are never promoted to biological replicates. The projection
reports the median across acquisitions and deterministic leave-one-acquisition-
out estimates. It publishes no confidence interval. Biological-replicate
uncertainty is available only inside a profile whose source declares complete
condition-scoped replicate identities. Repeated labels in different conditions
do not establish pairing, so this meta-study uses pooled controls rather than
inferring a paired design. Otherwise the profile reports
`biological_replicate_identity_unknown`.

### Reader visualization path

Reader already supplies the study-neutral execution path:

```text
reader/v8
  -> plate_reader/single_reporter_screen
  -> sample_measurements/df
  -> plot/single_reporter_diagnostic
  -> registered file bundle
  -> canonical notebook dropdown and single viewport
```

The diagnostic shows four coordinated panels: OD trajectory, reporter
trajectory, reporter/OD trajectory, and the interval reduction. Reader owns the
mechanics and provenance of that plot. This study owns why 6–10 h is the
comparison interval and how RT-lnRNA subjects and condition roles are bound.
Adding another plot does not create another notebook or execution lifecycle.

### Selection policy

The machine-readable contract is [`protocol.yaml`](protocol.yaml). For each
candidate four-hour interval, the materializer derives growth-phase summaries
from the observed OD values in the same verified Reader records. It does not
claim a separate study-side blank correction:

- median OD over within-acquisition observations at each recorded time;
- one-hour ordinary least-squares slopes of log OD, with at least four points;
- each stratum scaled by its 90th percentile of positive integer-hour slopes;
- median normalized slopes within each acquisition, followed by the cohort
  median across equally weighted acquisitions, at the interval start and at one
  hour before the interval end.

An interval is growth-phase admissible when its normalized start slope is at
least `0.5` and its normalized end slope is between `0.1` and `0.6`. Admissible
windows are then ordered lexicographically by worst-experiment control
separation, repeated-reference drift, within-acquisition observation range, and
earlier end. Within-acquisition observation range, incomplete reference panels, and
leave-one-acquisition-out instability remain visible limitations; they do not
erase an otherwise useful descriptive recommendation.

The primary reporter-response cohort uses 500 uM. The 5 uM and 50 uM cohorts,
single endpoints, and alternate centered-window widths are typed sensitivity
results and cannot select the primary reduction. There is no weighted score,
clipping, capping, label, or objective in this meta-study.

The later objective candidate is Reporter Response Feasibility (RRF), documented
in the parent [reporter-response evidence context](../reporter-response-evidence.md).
It is not computed here: this meta-study selects a descriptive reduction, while
the proposed objective separately needs predeclared response and OD margins,
validated OD linearity, and biological-replicate support.

### Evidence and publication

The operator accepts only source-closed Reader dataframe records, exact
study-owned evidence bindings, the pinned condition ontology, and
`rt_lnrna_reporter_response_profile.v3`. Raw workbooks, notebook cells,
screenshots, and caller-supplied digests are not evidence inputs.

`metastudy-state.yaml` is generated state. Do not edit it by hand. Its decision,
readiness, objective readiness, acquisition projection, sensitivity summaries,
and compact coverage receipts are one atomic generation. Immutable publications
contain `manifest.json`, `report.md`, `sensitivity.json`, and, for evaluated
decisions, `evidence.json`. A selected decision also contains
`acquisition.json` for its selected reduction. Offline verification recomputes
both the decision and any selected acquisition projection from the bundled typed
evidence; live status additionally verifies the external Reader records and
study bindings.

Run from the repository root:

The operator reads study-owned inputs from the Dnadesign checkout executing the
command. `--phd-root` remains the sibling-workspace root for Reader and bridge
routing. Use `--dnadesign-root` only to assert that the active checkout is the
one you intended; a different or missing checkout is rejected.

```bash
uv run rt-lnrna-reporter-metastudy regenerate \
  --phd-root /path/to/phd \
  --state-dir docs/studies/rt_lnrna_sponging_construct_triage/contexts/reporter-response-metastudy

uv run rt-lnrna-reporter-metastudy status \
  --phd-root /path/to/phd \
  --state-dir docs/studies/rt_lnrna_sponging_construct_triage/contexts/reporter-response-metastudy

uv run rt-lnrna-reporter-metastudy regenerate \
  --phd-root /path/to/phd \
  --publication /new/create-only/publication

uv run rt-lnrna-reporter-metastudy verify \
  --publication /path/to/publication
```

Implementation and verification live under
`src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/`.
