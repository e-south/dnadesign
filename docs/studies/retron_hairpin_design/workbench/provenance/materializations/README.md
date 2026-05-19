## Retron Materializations

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Materialization records state whether a workbench design set was turned into
sequence, GenBank, and plot deliverables.

### Records

- `2026-05-18-msd-177-194.single-unit.yaml`: single-unit materialize posture
  for the `scar_nick_profile_panel_v1` cohort. The checked-in cohort is routed
  through a typed spec that selects TetR plus explicit 5'->3' C172 and C26
  cap/foldback segments; C26 materializes without subsection labels because no
  topology is supplied.
- `2026-05-18-msd-177-194.non-ligatable-s0-control.yaml`: materialization
  record for the explicit C172/LCGGG/RACAG/MXMX control. It is not default
  scar-nick-compatible output; the spec declares the S0 exception and the
  emitted reference marks `s0_match_required=false`.

### Boundary

Store required inputs, expected bundle contracts, and blocker status here. Do
not commit generated GenBank files or plot images under this workbench lane by
default.

Materialization records are allowed to be explicit blockers. A blocked record
is preferable to emitting GenBank or plots from fixture-only sequence evidence
or from a label whose declared mismatch profile does not validate. Missing
topology blocks only topology-specific claims, not literal cap/foldback segment
assembly.
