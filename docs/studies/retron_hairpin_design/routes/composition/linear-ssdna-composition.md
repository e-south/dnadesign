---
doc_id: study-retron-hairpin-design-route-composition-linear-ssdna-composition
surface: study-route-detail
study_id: retron_hairpin_design
owner: dnadesign-maintainers
last_verified: 2026-05-18
parent_route: ../README.md
type: route
plane: data-plane
owner_boundary: construct/folding/baserender
surface_role: composition-service-handoff
current_state: materialization-ready-after-sequences
entry_artifact: selected_msd_parts_plus_payload_and_cap_sequences
exit_artifact: linear_ssdna_composition_v1_bundle
---

## Linear ssDNA Composition Route

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Use this route when the Retron task shifts from solving Snapback or scar-nick
primitives to composing a whole sequence artifact from selected parts.

### Route Contract

- Type: `route`
- Plane: `data-plane`
- Surface role: `composition-service-handoff`
- Owner-boundary: `construct/folding/baserender`
- Current state: `materialization-ready-after-sequences`
- Entry artifact: selected MSD parts plus concrete payload and cap sequences
- Exit artifact: `linear_ssdna_composition_v1` bundle with curated plots

### Boundary

- Retron study records own selected variants, rationale, labels, and display
  semantics.
- Construct owns generic `linear_ssdna_composition_v1` assembly.
- Folding owns secondary-structure prediction from explicit files or producer
  bundles.
- BaseRender renders sequence evidence maps.
- The Retron MSD compiler invokes these services for one selected MSD unit per
  design after parts and sequences are complete.

### Primary References

- Study handoff: `docs/studies/retron_hairpin_design/contexts/composition/linear-ssdna-composition.md`
- Generic authority: `src/dnadesign/construct/docs/reference/linear-ssdna-composition.md`
- Dev spec: `docs/dev/plans/cross-tool/linear-ssdna-composition/2026-05-13-generic-linear-ssdna-composition.md`
- Implementation record: `docs/exec-plans/completed/2026-05-13-generic-linear-ssdna-composition.md`
- Follow-ups: `docs/exec-plans/active/2026-05-14-linear-ssdna-composition-hardening-followups.md`

### Route Rule

For complete Retron MSD labels plus concrete payload and cap sequences, use the
study compiler `materialize` command from `routes/compiler/msd-design-references.md`.
For generic composition behavior, use the Construct reference. Do not create one
Construct or Folding workspace per Retron MSD ID.
