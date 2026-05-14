# Refresh Loop

Start with the user's parts and desired output. Use study status only when the
question is explicitly about progress, history, or blockers.

## Compiler Bootstrap

1. Parse what the user supplied: label, payload, cap, left base, right base,
   profile, repeat count, requested outputs.
2. Open the Study route for MSD design references in
   `docs/studies/retron_hairpin_design/routes.md`.
3. Open `docs/studies/retron_hairpin_design/msd_design_registry.yaml` for
   payload/cap/route metadata.
4. Use `msd-design-references.md` for lint/compile commands and output
   posture.
5. Open `linear-ssdna-composition.md` only when the requested output needs
   sequence assembly, visual QA, or GenBank sidecars.
6. Open `pipeline.yaml` only when a machine-readable command group is needed.

## Minimum Evidence By Question

| Question | Minimum evidence | Fail visibly when |
| --- | --- | --- |
| Can this ID compile? | normalized `msd_design_id`, profile, route metadata | label syntax, profile, `S0`, or registry lookup fails |
| What is missing? | missing payload/cap/base/profile/repeat/artifact fields | unknown fields are silently guessed |
| Which primitive should solve a missing part? | Snapback, scar-nick, or YIU route plus reason | the compiler tries to solve primitive search internally |
| Where did outputs go? | explicit out-dir and contract filenames | outputs are hidden in a new workspace |
| What is the old study status? | status command output | progress posture is mixed into a compile answer |

## Pair-With Rules

- Pair with `harness-engineering` when routing, skill audits, or deterministic
  command surfaces change.
- Pair with `code-change-discipline` when contracts, ontology, fail-fast
  behavior, or module boundaries change.

## Failure Routing

- Missing payload/cap registry entry: fix registry or ask for explicit metadata.
- Non-ligatable `S0`: stop; do not generate a catalog.
- Missing scar-nick feasibility: route to scar-nick.
- Missing cap/shortening geometry: route to Snapback.
- Output needs Reader integration: compile catalog first, then snapshot into
  the owning Reader experiment in a later handoff slice.
