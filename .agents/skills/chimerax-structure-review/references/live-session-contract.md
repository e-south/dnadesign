# Live Session Contract

Use this contract when the agent should visibly control one ChimeraX session.

## Purpose

The live-session path proves that an agent can interoperate with the same graphical ChimeraX window that the user can see.

It is not a substitute for a final publication render. It is an operator harness for checking that open, view, side-chain, surface, style, and capture commands work together.

## Required Evidence

Record:
- ChimeraX executable path
- source structure path and hash
- REST host and port
- control-session manifest path
- current pause point
- command log path and hash
- capture pose manifest path
- whether REST was stopped
- whether the GUI process was left open for user inspection

## Same-Session Checks

The dogfood path must use one ChimeraX process and one REST port for:
1. opening the structure;
2. setting a white background and cartoon representation;
3. rotating or otherwise changing the view;
4. showing side-chain atoms for an explicit residue selection;
5. adding a molecular surface;
6. saving a session, PNG, command log, and pose manifest.

## Safety

Default behavior should stop REST after capture while leaving the graphical window visible.

Use an explicit `--keep-rest` option only when the user wants continued live control. Use an explicit `--close-after` option for automated checks where leaving a GUI window open would be noisy.

Before `--close-gui` sends a signal, the stop helper verifies that the manifest PID is positive and live, names the declared ChimeraX executable, and owns the recorded REST listener. This verification requires `lsof` and happens before REST is stopped. Executable wrappers with a basename other than `ChimeraX` or `chimerax` can start a session but cannot use automatic GUI termination. If REST was already stopped, close the GUI manually rather than signaling from a stale manifest. A stale or hand-written manifest must fail without signaling another process.

For multi-turn collaboration, use `chimerax-session-start.sh`, `chimerax-session-status.sh`, and `chimerax-session-stop.sh`. These scripts make the pause points explicit and avoid hiding a long-running REST endpoint.

The control-session manifest records the session at start and follows `assets/control_session_manifest.schema.yaml`. Current state is established by the status or stop scripts.
