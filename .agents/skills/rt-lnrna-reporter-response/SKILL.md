---
name: rt-lnrna-reporter-response
description: Route admitted RT-lnRNA reporter-response profiles, interval recommendations, and objective readiness. Use for verified Reader-to-study evidence handoffs; not for generic Reader plots or MSD design.
metadata:
  version: 1.3.1
  category: workflow-automation
  tags: [rt-lnrna, reader, reporter-response, evidence, study]
  owner: rt_lnrna_sponging_construct_triage
  routing_priority: specific
---

# RT-lnRNA reporter response

## Purpose

Route assay evidence to its owner without turning a historical metric or a
bridge into scientific authority.

## Scope

Use this skill after the workspace bridge has admitted RT-lnRNA Reader evidence
into the study. It owns descriptive response profiles, window meta-study state,
sensitivity publications, and objective-readiness routing. Cross-repository
Reader intake, generic Reader plots, and MSD design remain outside this skill.

## Success Criteria

- Exact Reader and subject-binding identities are named.
- Primary selection and sibling sensitivity evidence stay distinct.
- Measurement, visualization, reduction, and objective readiness are reported
  independently.
- The response stops at a verified study-owned artifact or explicit blocker.

## Workflow

1. For any mixed Reader/dnadesign request, delegate intake and manifest
   readiness to the workspace `retron-assay-study-bridge`. Continue here only
   with its exact admitted Reader record identity.
2. Open `docs/studies/rt_lnrna_sponging_construct_triage/routes/reporter-response-evidence.md`.
3. Resolve exact RT-lnRNA identity through the study subject-binding registry.
4. Require a source-closed Reader record and a separately source-closed,
   canonically digested evidence-binding artifact. Never accept caller-supplied
   provenance lookalikes.
5. Publish exactly one descriptive profile variant. Use
   `rt_lnrna_reporter_measurement_profile.v1` when reference normalization is
   unavailable; otherwise use `rt_lnrna_reporter_response_profile.v3`, which
   retains the raw measurements and adds normalized response. Both derive
   exact provenance from one bound subject row and derive their typed
   observation-policy digest.
6. Compare only profiles with the same comparability key.
   Biological-replicate identity comes only from Reader's declared replicate
   field and is scoped to its source condition. Reusing one label under two
   conditions does not declare a pair. Acquisition, plate, sheet, well, and
   position stay provenance; an absent declaration remains unknown. The
   canonical eight-acquisition meta-study therefore pools controls by design;
   it does not invent cross-condition pairing.
7. Route interval choice to the study meta-study. The current canonical
   descriptive recommendation is the inclusive 6-10 h window with evidence
   grade `provisional_descriptive`.
8. For the assay view, use Reader's canonical
   `plate_reader/single_reporter_screen` path and registered
   `single_reporter_diagnostic`. Do not add a study notebook or a parallel
   execution command. Reader owns growth, reporter, reporter/OD, and interval
   panels; the study owns subject identity, condition roles, and the reason for
   the selected interval.
9. Use `regenerate` for live reconstruction and create-only publication,
   `status` for live parity with checked source state, and `verify` for an
   immutable offline publication. The canonical
   operator routes through the public Reader resolver and exact binding ledger.
   The source state keeps strict non-selectable sensitivity summaries and
   compact coverage receipts as siblings of the decision. An immutable
   publication carries the full profile/audit and exact coverage-ledger
   projection in `sensitivity.json` for offline recomputation.
10. Report four states separately: measurement readiness, descriptive
    visualization readiness, reduction-recommendation status, and objective
    readiness. Stop before OPAL while the objective remains blocked by an
    undefined constrained objective, unsupported biological-replicate uncertainty,
    or unvalidated OD linearity.

## Guardrails

- Reader owns generic measurements and recorded time; this study owns
  scientific selection and interpretation.
- The bridge routes evidence; it owns no formulas, controls, treatments, or
  objective semantics.
- Endpoint relative OD is an endpoint OD ratio, not viability or growth rate.
  Call it relative biomass only after the meta-study validates OD linearity
  and handling effects.
- Do not name or publish an objective until its formula, constraint, uncertainty
  unit, and validation claim are explicit. The 6-10 h recommendation is a
  descriptive reduction only.
- Sensitivity evaluations may diagnose endpoint, centered-window, or dose
  behavior, but never enter `MetastudyDecision`, window selection, or OPAL.
- A subject/window omission limits that coordinate. It does not invalidate
  other valid subjects from the same experiment.
- Treat the selected acquisition projection as descriptive aggregation with
  leave-one-acquisition-out robustness only. It does not provide a biological
  confidence interval or license acquisition-level bootstrap claims.
- Do not require a named inducer or a particular construct such as retron26.
  Resolve condition and control roles from the study-owned ontology.

## Required Deliverables

- Owner and exact evidence identity.
- Profile comparability and uncertainty state.
- Four independent readiness states, the meta-study recommendation, sibling
  sensitivity state, and stop condition.

## Trigger Tests

Use for “verify the RT-lnRNA response-window meta-study” or “route these Reader
records into the retron reporter-response evidence.” Do not use for “add a
generic Reader plot” or “design an MSD hairpin.”

## Output Contract

Return the owner, exact evidence identity, profile comparability, measurement
readiness, descriptive-visualization readiness, reduction recommendation,
objective readiness, sensitivity state, and stop condition.

## Progressive Disclosure Resources

- Owner source map: `references/external-sources.md`
- Routing test matrix: `references/test-matrix.md`
