# Workflow Router

Pick one mode before acting.

## `preflight`

Use when:
- the user asks whether ChimeraX can be controlled
- a render/capture fails at startup
- the ChimeraX executable path is unknown

Load:
- `chimerax-rest-contract.md`

Run:
- `scripts/chimerax-preflight.sh`

## `manual-pose-handoff`

Use when:
- the user wants to rotate, pan, or zoom manually
- the current view should become the canonical pose

Load:
- `collaboration-cadence.md`
- `pose-manifest-contract.md`
- `style-preset-contract.md`

Behavior:
- Tell the user what session/script to open.
- Wait for "capture now" or equivalent before sending capture commands.

Preferred script:
- `scripts/chimerax-session-start.sh` opens a visible session and writes a control-session manifest.

## `live-control`

Use when:
- the user explicitly asks the agent to interoperate with an active ChimeraX session
- the REST endpoint is already started or should be started now

Load:
- `collaboration-cadence.md`
- `chimerax-rest-contract.md`
- `command-allowlist.md`
- `natural-language-control-map.md`

Behavior:
- Use `127.0.0.1`.
- Read the session manifest when available instead of asking the user to repeat a port.
- Send only allowlisted commands.
- Stop REST after capture unless the user asks to keep it open.

First-run path:
- Use `scripts/chimerax-live-demo.sh` to open the packaged demo structure and prove same-session interop before using study structures.

## `style-apply`

Use when:
- the user asks for a visual change such as surface transparency, background, chain color, title, or visibility

Load:
- `natural-language-control-map.md`
- `style-preset-contract.md`

Behavior:
- Translate natural language to a named style preset or allowlisted command template.
- Do not infer biological meaning from a style.

## `render-capture`

Use when:
- the user asks to save a PNG, session, or pose manifest

Load:
- `pose-manifest-contract.md`
- `chimerax-rest-contract.md`

Behavior:
- Write a command log and pose manifest.
- Hash generated files.
- Report whether REST was stopped.
