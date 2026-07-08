---
doc_id: study-retron-hairpin-design-workbench-design-sets
surface: study-workbench-design-sets
study_id: retron_hairpin_design
owner: dnadesign-maintainers
last_verified: 2026-07-08
plane: intent-plane
surface_role: durable-study-cohorts
---

## Retron Workbench Design Sets

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-22

Stores durable study cohorts. A design set is the authoritative answer for
which variants belong to an experimental effort and which workbench ontology
terms they test.

### Records

- `scar_nick_profile_panel_v1.yaml`: pES-retron-177 through pES-retron-194,
  with expected MSD design ids, direction ids, effect tags, nickase posture, and
  rationale.
- `teto_pwm_trim_rescue_v1.yaml`: nine-variant bidirectional TetR
  cargo-shortening pilot across retron26 control, retron43 target scaffold,
  and the pES-retron-180 C172/AGTG/CATG/XWMM precedent under WT Eco1 RT.
- `teto_payload_trim_retest_v1.yaml`: four-variant tetO payload-prior retest
  that keeps the 15 nt and 13 nt retained-span extents from the TetR PWM pilot,
  but applies them to the retron26/retron43 tetO payload prior in retron26 and
  retron180 scaffold contexts.

### Boundary

Do not put generated compiler catalogs here. Generated catalogs belong in
caller-chosen transient output directories or a future owning Reader snapshot.
Convenience label inputs remain under `../../compiler/`.
Hypothesis-specific expectations for PWM panels, sequence-review stills/videos,
GenBank handoff files, and future outcome overlays belong in
`../deliverables/`.
