## Retron Workbench Design Sets

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-20

This lane holds durable study cohorts. A design set is the authoritative answer
to which variants belong to an experimental effort and which workbench ontology
terms they test.

### Records

- `scar_nick_profile_panel_v1.yaml`: pES-retron-177 through pES-retron-194,
  with expected MSD design ids, direction ids, effect tags, nickase posture, and
  rationale.
- `teto_pwm_trim_rescue_v1.yaml`: nine-variant bidirectional TetR cargo-shortening pilot
  across retron26 control, retron43 target scaffold, and one selected
  DE033-compatible cap/stem-base context under WT Eco1 RT.

### Boundary

Do not put generated compiler catalogs here. Generated catalogs belong in
caller-chosen transient output directories or a future owning Reader snapshot.
Convenience label inputs remain under `../../compiler/`.
Hypothesis-specific expectations for PWM panels, sequence-review stills/videos,
GenBank handoff files, and future outcome overlays belong in
`../deliverables/`.
