---
name: snapback-hairpin-study
description: Recover the checked-in Snapback shortening study, its released-product Snapback execution lane, its YIU boundary role, and its study-owned command ladder. Use when the user asks about the snapback hairpin or shortening effort, wants the current phase or next route, or wants to harden the study-owned ops, docs, or native-agent bootstrap surfaces. Pair with `harness-engineering` for study-surface hardening and with `code-change-discipline` in the `pragmatic-programming-principles` lane for lane, contract, or fail-fast boundary changes. Do not use for generic Cruncher walkthroughs, unrelated Snapback design work, or bench-level retron advice.
metadata:
  version: 0.1.1
  category: workflow-automation
  tags: [cruncher, snapback, study, shortening, routes]
---

# Snapback Hairpin Study

## Purpose

Answer `what is the checked-in Snapback shortening effort trying to do right
now?` from the study record instead of reconstructing it from notes, workspaces,
and runbooks.

## Scope

In scope:
- the checked-in `docs/studies/snapback_shortening_effort/` record
- `cruncher-study-status` and `cruncher-study-preflight` for this study
- the released-product Snapback route, the YIU contrast route, and the
  study-owned native-agent bootstrap context
- pairing with `harness-engineering` when the work changes study status,
  preflight, skill routing, or docs integrity
- pairing with `code-change-discipline` in the `pragmatic-programming-principles` lane when the work changes lane
  boundaries, contracts, or failure behavior

Out of scope:
- generic Cruncher operator walkthroughs outside this tracked study
- turning YIU into the shortening topology engine
- treating the retron note as hidden solver scoring
- arbitrary released-product feature work with no tracked-study angle

## Success Criteria

- the answer comes from the checked-in study record plus the pinned ops status
  or preflight surface
- released-product Snapback remains the active shortening lane
- released-product Snapback means the BspQI-pinned retained-active geometry lane rebased so the nick boundary is origin `0` in final-geometry space for this study
- YIU remains a contrast-only boundary surface
- the next route goes through `routes.md`; open `pipeline.yaml` only for
  machine-readable command-group or bootstrap confirmation, not hand-built
  command guesses
- harness or contract changes stay explicit and fail fast

## Workflow

1. Load the checked-in study surfaces.
- Read `docs/studies/README.md` and `docs/studies/index.yaml`.
- Read `docs/studies/snapback_shortening_effort/status.md`.
- Use `docs/studies/snapback_shortening_effort/routes.md` as the canonical
  next-command handoff.
- Open `docs/studies/snapback_shortening_effort/pipeline.yaml` only when the
  task needs machine-readable command-group or native-agent bootstrap context.
- Use [study-surfaces.md](references/study-surfaces.md) for ownership
  boundaries.

2. Refresh the record-backed answer first.
- Run
  `uv run ops progress show cruncher.data-plane.cruncher-study-status --study-dir docs/studies/snapback_shortening_effort --json`
  for the current phase, command groups, and bootstrap context.
- Route blocker or next-run readiness questions to
  `uv run ops progress show cruncher.data-plane.cruncher-study-preflight --study-dir docs/studies/snapback_shortening_effort --scope next --json`.
- Use [route-matrix.md](references/route-matrix.md) and
  [refresh-loop.md](references/refresh-loop.md) for cold-start routing.
- If the user asks which nicking endonucleases can produce the origin-0,
  stem-3, cap-3 outcome, open [origin-033-hits.md](references/origin-033-hits.md)
  before answering; treat it as the concise study-owned answer and rerun the
  listed screen/solve command only when freshness is requested.

3. Pair with the right companion skill when the task widens.
- Pair with `harness-engineering` when the change touches study status,
  preflight, repo-local skill routing, or docs integrity. Keep the endpoint set
  to `knowledge-integrity`, `autonomy-capability`, and
  `architecture-invariants`.
- Pair with `pragmatic-programming-principles` via `code-change-discipline`
  when the change touches the
  released-product vs preserved-site boundary, YIU boundary language, explicit
  degraded modes, or fail-fast contract behavior.

## Guardrails

- `released-product Snapback` is the shortening architecture under test.
- in this study, released-product Snapback scores retained active top and bottom products; do not collapse the screen back to exposed-bottom-only semantics.
- outside-site type IIS nickases keep their downstream spacer geometry, but the
  released-product lane only permits left-of-origin nickase geometry when the
  omitted leading prefix is one contiguous fully degenerate `N` block in the
  oriented top-strand view; protected bases and all release-site geometry must
  still remain at or to the right of logical origin `0`.
- treat released-product exact hits as multi-invariant: the nick must come from
  a real nickase recognition site positioned at the resolved boundary, any
  top-prefix fragment that remains left of the nick must stay Watson-Crick
  paired to the exposed active bottom across that residual duplex overlap, and
  the active bottom stem and foldback return must also remain Watson-Crick
  paired.
- default released-product operational policy excludes nickases carrying `FREQUENT_CUTTER`; do not route operators to `Nt.CviPII` or a pinned explicit `de033` bundle unless the task is an explicit policy-comparison audit.
- `preserved-site Snapback` stays a separate contract.
- `YIU` stays mismatch-centric and contrast-only here.
- the retron/P4 note is framing context, not a hidden scoring hook
- use the pinned study commands and paths; do not rebuild them from memory
- do not require `pipeline.yaml` or `ops.study.yaml` to recover the next human
  step when `routes.md` already answers it

## Required Deliverables

- whether the answer came from snapshot posture or preflight readiness
- current phase and next owning surface
- current primary lane and its command group
- explicit note that YIU is contrast-only for this study
- explicit pair-with guidance when harness or boundary work is requested

## Output

Return:
- study id
- snapshot vs preflight posture
- current phase and next route
- current primary lane and contrast lane
- the next file, workspace, or command group to open
- explicit blockers only when preflight was requested

## Trigger Tests

Should trigger:
- "Check the snapback hairpin study."
- "Where does the snapback shortening effort stand right now?"
- "What should I open next for the shortening study?"
- "Refresh the tracked Snapback shortening context."
- "Harden the shortening study status, preflight, or skill routing."
- "Which nicking endonucleases result in the 033 snapback?"

Should not trigger:
- "Run a generic Cruncher snapback search."
- "Explain retron biology."
- "Design a new YIU payload."
- "Add a released-product feature with no study-record change."

## References

- [route-matrix.md](references/route-matrix.md)
- [refresh-loop.md](references/refresh-loop.md)
- [study-surfaces.md](references/study-surfaces.md)
- [origin-033-hits.md](references/origin-033-hits.md)
- [external-sources.md](references/external-sources.md)
