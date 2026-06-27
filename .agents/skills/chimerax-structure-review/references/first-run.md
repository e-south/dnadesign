# First Run

Use this path when a user or agent wants to prove the skill works before using a study structure.

## Visible Demo

Run:

```bash
.agents/skills/chimerax-structure-review/scripts/chimerax-live-demo.sh
```

Expected behavior:
- opens `assets/demo_structure.pdb` in a graphical ChimeraX window;
- starts REST on `127.0.0.1`;
- changes the view in the same session;
- shows side-chain atoms for a declared residue range;
- adds a faint molecular surface;
- writes a PNG, `.cxs`, command log, pose manifest, and live-session manifest;
- stops REST by default while leaving the window visible for inspection.

Use automated cleanup when running in tests:

```bash
.agents/skills/chimerax-structure-review/scripts/chimerax-live-demo.sh --close-after
```

Use continued control only when needed:

```bash
.agents/skills/chimerax-structure-review/scripts/chimerax-live-demo.sh --keep-rest
```

## Interpretation

Passing this demo proves the local interop path works. It does not prove that a study-specific pose, color map, or publication figure is correct.

## Collaborative Session

Start a session and pause:

```bash
.agents/skills/chimerax-structure-review/scripts/chimerax-session-start.sh \
  --structure .agents/skills/chimerax-structure-review/assets/demo_structure.pdb
```

Check that it is still live:

```bash
.agents/skills/chimerax-structure-review/scripts/chimerax-session-status.sh \
  --session-manifest <control_session.yaml>
```

Send one visible action:

```bash
.agents/skills/chimerax-structure-review/scripts/chimerax-send-command.py \
  --session-manifest <control_session.yaml> \
  --command 'turn y 20 12'
```

Capture while continuing to collaborate:

```bash
.agents/skills/chimerax-structure-review/scripts/chimerax-capture-pose.py \
  --session-manifest <control_session.yaml> \
  --output-dir <capture-dir> \
  --pose-id manual_pose_v1 \
  --keep-rest-open
```

Stop control:

```bash
.agents/skills/chimerax-structure-review/scripts/chimerax-session-stop.sh \
  --session-manifest <control_session.yaml>
```
