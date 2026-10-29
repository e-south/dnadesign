# ChimeraX REST Contract

## Contract

ChimeraX live control is allowed only through a short-lived local REST endpoint.

Start inside ChimeraX:

```text
remotecontrol rest start port <port> json true log true
```

Send commands to:

```text
http://127.0.0.1:<port>/run
```

Stop after use:

```text
remotecontrol rest stop
```

## Required Invariants

- Host must be `127.0.0.1`.
- Port must be explicit or recorded.
- REST must not be left running after a capture unless the user asks for continued control.
- For continued control, a control-session manifest must record the port, PID, input structure, and current pause point.
- Every command sent by an agent must be generated from `command-allowlist.md`.
- The response must be checked for a JSON `error` field when `json true` is enabled.

## Graphical Session Requirement

Use a normal graphical ChimeraX session for pose capture and rendered PNG export.

Do not treat `--nogui` as a supported capture mode. REST commands can start in headless contexts, but camera and image-render commands may fail because no OpenGL view exists.

## Script Suffix Footgun

ChimeraX rejects scripts whose final path suffix is not `.py` or `.cxc`. When creating temporary command files, create a temporary directory and place a fixed filename inside it, such as:

```text
<tmpdir>/remote_smoke.cxc
```

Do not use a temporary filename where random characters are appended after `.cxc`.

## Stop Conditions

Stop and report instead of retrying when:
- the endpoint is not reachable after bounded polling
- the JSON response contains `error`
- a command is not represented in the allowlist
- the user is actively manipulating the same session and has not asked for a capture

## Evidence

Record:
- ChimeraX executable path
- port
- command log
- JSON response status
- REST stop command status
- output file paths and hashes, if any
