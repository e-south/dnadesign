---
name: chimerax-structure-review
description: Collaboratively review protein or protein-complex structures in ChimeraX with visible-session handoff, pose capture, style presets, and local REST control.
metadata:
  version: 0.2.0
  category: workflow-automation
  tags: [chimerax, structure-review, visualization, rendering, pose-capture]
---

# ChimeraX Structure Review

## Purpose

Help a user and agent co-review molecular structures in ChimeraX, capture an approved pose, apply repeatable visual styles, and render still images with provenance.

## Scope

In scope:
- local ChimeraX discovery and preflight
- visible control-session start, status, pause, resume, capture, and stop
- manual orientation handoff
- short-lived localhost REST control through ChimeraX `remotecontrol rest`
- allowlisted commands for view capture, background, labels, surfaces, transparency, colors, chain visibility, and saved renders
- pose manifests that record inputs, commands, output hashes, and tool versions

Out of scope:
- arbitrary free-text ChimeraX command execution
- long-running REST servers
- structure prediction or fold validation
- study-specific biological interpretation
- importing sibling-project code as a runtime dependency

## Required Inputs

- A structure file or an already-open ChimeraX session.
- A review intent: start a session, orient, capture pose, apply style, render still, inspect visibility, or continue a live session.
- Optional: reference model, query model, selection, title, output directory, and render size.

Clarification policy:
- Ask before starting live REST control unless the user explicitly asked to interoperate with an active ChimeraX session.
- Ask before saving outputs outside the current repo or declared workspace.

## Success Criteria

- ChimeraX preflight succeeds or reports a specific missing executable.
- REST control uses `127.0.0.1`, a declared port, and a short-lived session.
- Commands sent over REST are generated from allowlisted templates.
- Pose capture uses a graphical ChimeraX session; headless REST smoke is not evidence that camera/render capture works.
- Live-session dogfood can open the packaged demo structure, change the view, show side-chain atoms, add a surface, and capture from one graphical session.
- Collaboration mode has clear pause points: session-ready, user-steering, agent-action, capture-ready, and stop-or-continue.
- Pose capture writes a session, image, command log, and pose manifest when requested.
- REST control is stopped after capture unless the user explicitly asks to keep the session open.

## Workflow

1. Route the task with `references/workflow-router.md`.
2. For a collaborative session, read `references/collaboration-cadence.md` and start with `scripts/chimerax-session-start.sh`.
3. For live control, read `references/chimerax-rest-contract.md` before sending commands.
4. For natural-language visual edits, read `references/natural-language-control-map.md` and `references/command-allowlist.md`.
5. For pose capture, run the preflight, send allowlisted capture commands, and write the pose manifest described in `references/pose-manifest-contract.md`.
6. For style choices, use `references/style-preset-contract.md`; treat sibling-project examples as patterns only.
7. Run `scripts/audit-chimerax-structure-review-skill.sh` after editing this skill.

## Guardrails

- Never pass raw user prose directly to ChimeraX.
- Do not keep the REST endpoint running after the requested capture.
- When REST is intentionally kept open, report the port, session manifest, and next pause point.
- Do not treat a saved `.cxs` session as the only reproducibility artifact.
- Do not place study-specific biology in this skill's top-level ontology.
- Use installed ChimeraX documentation or official UCSF docs for command claims.

## Required Deliverables

- Command or render action summary.
- Output paths for session, PNG, command log, and pose manifest when capture occurs.
- REST port and stop status for live-control runs.
- Current pause point and the expected next user or agent action.
- Validation evidence or a concrete blocked reason.
- Assumptions and any commands intentionally not sent.

## Output Contract

Return:
- action taken or planned
- artifacts written, with paths
- commands sent, summarized by allowlist key
- validation evidence
- remaining risks, active-session state, and next manual step

## Trigger Tests

Should trigger:
- "Open this structure in ChimeraX and help me orient it."
- "Capture this ChimeraX pose."
- "Make the surface faint and render a white-background PNG."
- "Show only chain A and save the view."

Should not trigger:
- "Predict a protein structure."
- "Run ColabFold."
- "Write a wet-lab protocol."
- "Execute this arbitrary ChimeraX command exactly as typed."

## Progressive Disclosure Resources

- `references/workflow-router.md`
- `references/first-run.md`
- `references/collaboration-cadence.md`
- `references/name-scope-decision.md`
- `references/chimerax-rest-contract.md`
- `references/live-session-contract.md`
- `references/command-allowlist.md`
- `references/natural-language-control-map.md`
- `references/pose-manifest-contract.md`
- `references/style-preset-contract.md`
- `references/sibling-patterns-example.md`
- `references/external-sources.md`
- `references/test-matrix.md`
- `scripts/audit-chimerax-structure-review-skill.sh`
