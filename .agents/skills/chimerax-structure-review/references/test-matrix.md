# Test Matrix

| Scenario | Prompt | Expected Behavior | Pass/Fail |
| --- | --- | --- | --- |
| Trigger positive | "Capture this ChimeraX pose and save a PNG." | Use this skill, route to `render-capture`, and require a pose manifest. | Pass if the workflow names capture artifacts and does not run arbitrary commands. |
| Trigger negative | "Run ColabFold on this sequence." | Do not use this skill. | Pass if the request routes to fold-model/runtime tooling instead. |
| Functional core | REST smoke starts ChimeraX control. | Start `remotecontrol rest` on `127.0.0.1`, get `error=null`, then stop. | Pass if `scripts/chimerax-rest-smoke.sh` exits zero and reports REST stopped. |
| Functional core | Visible same-session interop. | Open `assets/demo_structure.pdb`, change view, show side-chain atoms, add a surface, and capture. | Pass if `scripts/chimerax-live-demo.sh --close-after` writes a live-session manifest and capture manifest. |
| Functional core | Collaborative pause cadence. | Start a visible session, pause, send a command through the session manifest, capture while keeping REST open, then stop. | Pass if start/status/send/capture/stop use one port and every step reports the next pause point. |
| Functional edge | Temporary script creation. | Create a fixed `.cxc` file inside a temp directory. | Pass if the smoke script never creates paths with random suffixes after `.cxc`. |
| Functional edge | Pose capture render context. | Use a normal graphical ChimeraX session for camera and PNG capture. | Pass if docs reject `--nogui` as capture evidence and a graphical-session capture writes outputs. |
| Reliability | Pose capture with a declared output directory. | Save `.cxs`, `.png`, command log, and pose manifest. | Pass when output files exist or the manifest records a specific failure reason. |
| Repeatability | Run the skill audit twice. | Structural checks pass both times with no generated repo-root outputs. | Pass if `audit-chimerax-structure-review-skill.sh` exits zero twice. |
