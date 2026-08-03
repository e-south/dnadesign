---
doc_id: study-rt-lnrna-sponging-construct-triage-reporter-response-route
surface: study-route
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-08-02
parent_route: README.md
bridge_route: rt_lnrna_reporter_response_evidence
data_plane_role: descriptive-evidence-binding
current_state: provisional-descriptive-reduction-selected
measurement_readiness: ready
descriptive_visualization_readiness: ready
reduction_recommendation_status: ready
objective_readiness_status: blocked
entry_artifacts:
  - source-closed-reader-record
  - source-closed-reader-evidence-bindings
exit_artifacts:
  - rt_lnrna_reporter_measurement_profile.v2
  - rt_lnrna_reporter_response_profile.v4
  - reporter-response-metastudy-route
---

## Reporter-response evidence route

Use this route for RT-lnRNA assay evidence, the canonical Reader view, and the
study-owned reduction recommendation.

1. Resolve exact biological subjects through the study-owned subject-binding
   registry. Reader labels are join keys, not biological authority.
2. Require exact, digest-verified Reader records and a separate evidence
   binding. Profile construction and parsing require the source-closed binding
   set and derive provenance from exactly one bound subject row. Do not copy
   measurements into the identity registry.
3. Publish exactly one descriptive profile variant. A positive control is
   optional and must be declared by role; its name is never inferred. When
   reference normalization is unavailable, use
   `rt_lnrna_reporter_measurement_profile.v2` and record the typed reason.
   Otherwise use the stricter `rt_lnrna_reporter_response_profile.v4`, which
   retains the recorded interval, dose grid, condition roles, raw reporter,
   OD600, and reporter/OD600 measurements and adds normalized response,
   relative OD, and supported biological-replicate uncertainty.
4. Declare a typed observation policy whose fixed formulas and reduction
   semantics derive its digest. Never supply a policy digest as provenance.
5. Require exact comparability-key equality for direct aggregation within one
   profile variant. The meta-study may compare the shared raw estimand across
   raw and reference-normalized variants only when observation-policy digest,
   reduction, and dose grid match; normalized response remains a separate
   compatible subset and is never imputed for a raw-only profile. Stop on any
   mismatch in that base identity.
6. View each admitted time series through Reader's one public lifecycle:
   `reader/v8 -> plate_reader/single_reporter_screen ->
   sample_measurements/df -> plot/single_reporter_diagnostic -> registered
   file bundle -> canonical notebook viewport`. The diagnostic coordinates OD,
   reporter, reporter/OD, and interval-reduction panels without importing
   RT-lnRNA interpretation into Reader.
7. Route interval selection through bridge route
   `rt_lnrna_reporter_response_metastudy`, then to the study's
   [retrospective calibration meta-study](../contexts/reporter-response-metastudy/README.md).
   The current 6-10 h recommendation is `provisional_descriptive`; valid
   subject/window coordinates are retained when another coordinate is omitted.
   The selected calibration cohort contains eight kinetic acquisitions
   with the required assay layout: three from 2025 and five from 2026. Each
   experiment is one acquisition. Persisted positions are observations, not
   replicate identities. Historical acquisitions with no declared replicate
   field remain explicitly unknown; the 2026-07-27 acquisition's declared
   replicate IDs are accepted without changing that rule. The 2025-11-05
   snapshot is descriptive context. Neither the competence experiment nor the
   related snapshot participates in tuning or blocks the recommendation.
8. Stop before OPAL. Objective readiness is independently blocked until the
   study defines a constrained objective, comparable profiles carry supported
   biological-replicate uncertainty, and OD linearity is validated.

The four readiness questions are deliberately separate: measurements are
ready, descriptive visualization is ready, the 6-10 h reduction recommendation
is ready, and objective readiness is blocked. There is no active scalar
objective. The proposed successor is Reporter Response Feasibility (RRF), not
SPOP; its formula and activation gates live in the
[reporter-response evidence context](../contexts/reporter-response-evidence.md).
