---
name: notify-ops
description: Operate dnadesign Notify setup, profile validation, watcher loops, and spool recovery. Use for Slack delivery or USR .events.log watches. Do not use for Slack admin, scheduler submits, or code changes.
metadata:
  version: 0.1.1
  category: workflow-automation
  tags: [notify, usr, operations, slack, dnadesign]
---

# Notify Ops

## Purpose

Run `notify` setup, watch, and recovery workflows through the repo's canonical
operator docs and fail-fast command contracts without depending on subtree
`AGENTS.md` injection.

## Scope

In scope:
- file-backed webhook setup for reusable watcher profiles
- resolver-mode and explicit-path Notify setup
- `notify profile doctor`, `notify usr-events watch`, and `notify spool drain`
- repo-rooted workspace shorthand for DenseGen, Infer, and Construct
- routing to scheduler-managed watcher flows when BU SCC batch is involved

Out of scope:
- generic Slack workspace administration
- non-Notify scheduler submission planning; use `.agents/skills/sge-hpc-ops/SKILL.md`
- changing Notify code with no operator-workflow question
- treating DenseGen runtime telemetry as Notify input

## Success Criteria

- the user is routed through exactly one canonical setup/watch/recover path
- the selected path uses repo-owned docs instead of hidden subtree-only
  instructions
- Notify input stays USR `.events.log` only
- reusable live delivery uses file-backed secret references instead of plaintext
  webhook URLs in profiles
- one watcher is planned per live lane or destination dataset when Notify and
  resume posture matter

## Workflow

1. Route through the canonical docs first.
- Start at `docs/notify/README.md` for workflow selection.
- Use `docs/notify/usr-events.md` for ordered setup, run, and recovery steps.
- Use `src/dnadesign/notify/docs/reference/command-contracts.md` when the
  question is about CLI contracts or fail-fast behavior.
- Use [workflow-router.md](references/workflow-router.md) to choose the next
  surface quickly.

2. Choose one operation family.
- Bootstrap or refresh profile: `uv run notify setup slack ...`
- Validate profile and wiring: `uv run notify profile doctor --profile <profile.json>`
- Resolve watcher inputs without writing profile artifacts:
  `uv run notify setup resolve-events --tool <tool> --workspace <workspace> --json`
- Start watcher loop: `uv run notify usr-events watch ...`
- Replay failed deliveries: `uv run notify spool drain --profile <profile.json>`

3. Apply repo-specific guardrails.
- Reusable Slack/profile workflows in this repo use
  `--secret-source file --secret-ref file://<abs-path-to-webhook-secret>`.
- Keep one watcher per live Infer lane config and one watcher per destination
  dataset; do not collapse multi-destination live lanes into one watcher when
  Notify routing and resume posture matter.
- Resolver shorthand is repo-rooted. Outside the repo checkout, use explicit
  `--config` or the runbook guidance for `DNADESIGN_REPO_ROOT`.
- Notify consumes `"<dataset>/.events.log"` only. DenseGen
  `outputs/meta/events.jsonl` is not Notify input.

4. Route scheduler-backed watcher work outward instead of expanding this skill.
- For BU SCC batch submits or long-running watcher jobs, use
  `docs/bu-scc/batch-notify.md`.
- When queue/session decisions are in scope, use
  `.agents/skills/sge-hpc-ops/SKILL.md`.

## Required Deliverables

- selected workflow family: setup, validate, watch, or recover
- canonical doc or command-contract surface used
- exact profile/events inputs or explicit missing-input failure
- secret-source posture for live delivery
- watcher topology note when Infer or multi-destination routing is involved
- next command or next owning doc to open

## Output

Return:
- workflow family
- source-of-truth doc or contract path
- exact command sequence or failure reason
- repo-root or config-path assumption
- watcher topology recommendation when relevant
- next verification step

## Trigger Tests

Should trigger:
- "Set up Notify for this workspace and validate the profile."
- "Start a Notify watcher for this USR event stream."
- "Recover failed Notify deliveries from spool."
- "Why does `notify profile doctor` fail on this repo?"
- "Which runbook should I use for Notify on BU SCC?"

Should not trigger:
- "Submit this BU SCC batch job."
- "Rotate a Slack workspace admin secret."
- "Explain DenseGen runtime telemetry."
- "Refactor Notify code."

## References

- [workflow-router.md](references/workflow-router.md)
- [external-sources.md](references/external-sources.md)
