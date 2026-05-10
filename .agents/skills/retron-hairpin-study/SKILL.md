---
name: retron-hairpin-study
description: Recover the checked-in retron hairpin design effort across released-product Snapback cap/shortening, scar-nick base-junction design, and YIU boundary contrast. Use when the user asks about the TetO/retron hairpin effort, snapback shortening, scar-nick stem-base scars, profile-diverse S0=M scar_nick coverage, current phase, next route, or study-owned docs/bootstrap hardening. Pair with `harness-engineering` for study-surface hardening and with `code-change-discipline` for lane, ontology, contract, or fail-fast boundary changes. Do not use for generic Cruncher walkthroughs, unrelated Snapback feature work, generic retron biology, or bench-protocol advice.
metadata:
  version: 0.4.0
  category: workflow-automation
  tags: [cruncher, retron, hairpin, snapback, scar-nick, study, routes]
---

# Retron Hairpin Study

## Purpose

Answer `what is the checked-in retron hairpin design effort trying to do right
now?` from the study record. The checked-in study id is
`retron_hairpin_design`, and the ontology covers both cap/shortening work
through released-product Snapback and base-junction work through scar-nick.

## Scope

In scope:
- the checked-in `docs/studies/retron_hairpin_design/` record
- released-product Snapback for cap/shortening geometry
- scar-nick for upstream Type IIS scar plus terminal nick base-junction geometry
- YIU as a contrast-only boundary surface
- `cruncher-study-status` and `cruncher-study-preflight` for this study
- study-owned automation bootstrap, route maps, and progressive disclosure

Out of scope:
- generic Cruncher operator walkthroughs outside this tracked study
- turning YIU into the shortening or scar-nick topology engine
- treating retron/P4 biology as hidden solver scoring
- arbitrary released-product or scar-nick feature work with no tracked-study angle
- bench-level retron protocol advice

## Success Criteria

- answers come from the checked-in study record plus pinned status/preflight
- subtopic routing is explicit: Snapback for cap/shortening, scar-nick for
  base-junction processing, YIU for contrast
- released-product Snapback remains the active shortening lane
- scar-nick context preserves the strict terminal nick rule: top or bottom nick
  allowed, zero protected bases downstream, downstream `N` only, and `S0=M`
- YIU remains mismatch-centric and contrast-only
- the next route goes through `routes.md`; open `pipeline.yaml` only for
  machine-readable command-group or bootstrap confirmation
- harness or contract changes stay explicit and fail fast

## Workflow

1. Load the checked-in study surfaces.
- Read `docs/studies/README.md` and `docs/studies/index.yaml`.
- Read `docs/studies/retron_hairpin_design/status.md`.
- Use `docs/studies/retron_hairpin_design/routes.md` as the canonical
  next-command handoff.
- Open `docs/studies/retron_hairpin_design/pipeline.yaml` only when the
  task needs machine-readable command-group or automation bootstrap context.
- Use [study-surfaces.md](references/study-surfaces.md) for ownership
  boundaries.

2. Refresh the record-backed answer first.
- Run
  `uv run ops progress show cruncher.data-plane.cruncher-study-status --study-dir docs/studies/retron_hairpin_design --json`
  for current phase, command groups, and bootstrap context.
- Route blocker or next-run readiness questions to
  `uv run ops progress show cruncher.data-plane.cruncher-study-preflight --study-dir docs/studies/retron_hairpin_design --scope next --json`.
- Use [route-matrix.md](references/route-matrix.md) and
  [refresh-loop.md](references/refresh-loop.md) for cold-start routing.

3. Load subtopic detail only when needed.
- For origin-0/stem-3/cap-3 Snapback nickase questions, open
  [origin-033-hits.md](references/origin-033-hits.md).
- For scar-nick base, profile-diverse S0-matched coverage, top/bottom nick flexibility, or
  retained-scar sequence-space questions, open
  `docs/studies/retron_hairpin_design/scar-nick-base-junction.md`.

4. Pair with the right companion skill when the task widens.
- Pair with `harness-engineering` when the change touches study status,
  preflight, repo-local skill routing, or docs integrity.
- Pair with `code-change-discipline` when the change touches lane boundaries,
  ontologies, contracts, degraded modes, or fail-fast behavior.

## Guardrails

- `released-product Snapback` is the shortening architecture under test.
- `scar-nick` is the base-junction processing surface, not a retron phenotype
  predictor.
- Scar-nick current processing policy is exact terminal nick, top or bottom strand
  allowed, zero protected bases downstream, downstream degenerate `N` only, and
  `S0=M` for ligation.
- The current scar-nick context says exact supplied B26/B43 L/R pairs are not
  catalog-feasible under that policy; profile-diverse S0-matched analogs are
  the maintained route.
- Active scar-nick panels allow S3-edge double-hard buckets such as
  `XXMM` and `XMXM`, but keep middle-middle `MXXM` as reserve/not active.
- Default operational policy excludes `FREQUENT_CUTTER` nickases.
- `preserved-site Snapback` stays a separate contract.
- `YIU` stays mismatch-centric and contrast-only.
- Retron/P4 notes are framing context, not hidden scoring hooks.
- Use pinned study commands and paths; do not rebuild them from memory.

## Required Deliverables

- whether the answer came from snapshot posture or preflight readiness
- current phase and next owning surface
- requested subtopic route: Snapback, scar-nick, YIU, or study-harness
- current primary lane and contrast lane
- explicit note that YIU is contrast-only
- explicit pair-with guidance when harness or boundary work is requested

## Output

Return:
- study id and checked-in path note when relevant
- snapshot vs preflight posture
- current phase and next route
- subtopic route and the next file, workspace, or command group to open
- explicit blockers only when preflight was requested

## Trigger Tests

Should trigger:
- "Check the retron hairpin study."
- "Where does the retron hairpin design effort stand right now?"
- "Route the scar-nick base-junction context."
- "What profile-diverse S0=M scar_nick candidates can we generate?"
- "Harden the hairpin study status, preflight, or skill routing."
- "Which nicking endonucleases result in the 033 snapback?"

Should not trigger:
- "Run a generic Cruncher snapback search."
- "Explain retron biology broadly."
- "Design a new YIU payload."
- "Add a released-product feature with no study-record change."

## References

- [route-matrix.md](references/route-matrix.md)
- [refresh-loop.md](references/refresh-loop.md)
- [study-surfaces.md](references/study-surfaces.md)
- [origin-033-hits.md](references/origin-033-hits.md)
- [external-sources.md](references/external-sources.md)
