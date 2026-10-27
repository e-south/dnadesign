# Collaboration Cadence

Use this reference when the user and agent will share one visible ChimeraX session.

## Goal

The user should be able to rotate, pan, zoom, or inspect the same ChimeraX window that the agent can control with allowlisted commands.

The agent should communicate pause points clearly instead of silently running a long command chain.

## Pause Points

| Pause point | Meaning | Agent behavior |
| --- | --- | --- |
| `session-ready` | ChimeraX is open, REST is running, and a session manifest exists. | Report the manifest path, port, and what the user can do next. |
| `user-steering` | The user is manually rotating, zooming, selecting, or inspecting. | Do not send commands unless the user asks or an agreed action is pending. |
| `agent-action` | The user asks the agent to apply a style, view, selection, or capture. | Summarize the intended allowlisted command family before acting when the operation is visual or destructive to the scene. |
| `capture-ready` | The visible pose should be persisted. | Capture PNG, `.cxs`, command log, and pose manifest. |
| `stop-or-continue` | Capture is done. | Ask whether to keep the GUI open, keep REST open, or stop control. Default to stop REST. |

## Communication Rules

- Say when REST is open and report the port.
- Say when REST has been stopped.
- If the user is actively steering the GUI, do not issue commands into that session.
- Before a multi-command visual change, state the intended action in one sentence.
- After a command group, report the result and the next pause point.

## Standard Flow

1. Start: `scripts/chimerax-session-start.sh --structure <path>`.
2. Pause at `session-ready`.
3. User steers manually or asks the agent for a visual change.
4. Agent sends allowlisted commands with `chimerax-send-command.py --session-manifest <manifest>`.
5. Pause at `capture-ready`.
6. Capture with `chimerax-capture-pose.py --session-manifest <manifest>`.
7. Stop control with `scripts/chimerax-session-stop.sh --session-manifest <manifest>`.

## Default Safety

REST should not stay open by accident. Keep it open only when the user is still actively collaborating in the session.

The control-session manifest is a start-time handoff record. Use `chimerax-session-status.sh` for current liveness and `chimerax-session-stop.sh` output for stop evidence; do not treat the start-time `rest_stopped` field as mutable current state.

The command log path in the control-session manifest is append-only session evidence. Commands sent with `chimerax-send-command.py --session-manifest <manifest>` append accepted or failed REST calls to that log.
