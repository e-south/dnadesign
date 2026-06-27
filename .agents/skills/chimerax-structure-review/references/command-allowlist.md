# Command Allowlist

Commands sent over REST must come from this table or a narrower study-specific command file.

| Key | Template | Purpose | Notes |
| --- | --- | --- | --- |
| `rest_port` | `remotecontrol rest port` | Confirm endpoint. | Read-only. |
| `rest_stop` | `remotecontrol rest stop` | Stop local control. | Use after capture. |
| `set_background` | `set bgColor <color>` | Set background color. | Prefer white for publication stills. |
| `camera_mode` | `camera ortho` or `camera mono` | Set camera mode. | Named views do not store camera mode. |
| `name_view` | `view name <pose_id>` | Save current view name in session. | Use stable IDs. |
| `restore_view` | `view <pose_id>` | Restore a named view. | Pair with explicit camera mode. |
| `fit_view` | `view <selection> pad <fraction>` | Fit the visible view around a declared selection. | Use before capture when a scene is cropped too tightly. |
| `save_session` | `save "<path>.cxs"` | Save review session. | Session is provenance, not the sole contract. |
| `save_image` | `save "<path>.png" width <int> height <int> supersample <int>` | Save still image. | Use declared dimensions. |
| `open_model` | `open "<path>"` | Open a local structure file. | Use only for declared local paths. |
| `close_session` | `close session` | Clear session. | Use only in setup scripts. |
| `show_selection` | `show <selection> <level>` | Show a declared selection. | Level must be explicit, such as `atoms`, `cartoons`, or `surfaces`. |
| `hide_selection` | `hide <selection>` | Hide a declared selection. | Selection must be explicit. |
| `show_only_chain` | `hide <model> target acs` then `show <model>/<chain> cartoons` | Focus one declared chain. | This is two explicit commands, not an inferred inverse selection. |
| `atom_style` | `style <selection> stick` or `style <selection> ball` or `style <selection> sphere` | Change atom/bond display style for an explicit selection. | Use `sidechain` or `sideonly` atomspec classes for side-chain inspection. |
| `surface_selection` | `surface <selection>` | Draw molecular surface. | Use with transparency if needed. |
| `surface_transparency` | `transparency <selection> <percent> target s` | Make a surface faint or opaque. | Percent is transparency, not opacity. |
| `cartoon_transparency` | `transparency <selection> <percent> target r` | Make cartoons faint or opaque. | Use sparingly for overlays. |
| `color_selection` | `color <selection> <color> target <targets>` | Color a selection. | Targets commonly `r`, `s`, or `rs`. |
| `cartoon_style` | `cartoon style width <float> thick <float>` | Tune ribbon visual weight. | Scene-level visual polish. |
| `turn_view` | `turn x <degrees> <frames>` or `turn y <degrees> <frames>` or `turn z <degrees> <frames>` | Rotate the current scene for a visible same-session view change. | Use `wait` after animated turns. |
| `wait` | `wait <frames>` | Wait for an animated command to finish. | Use after `turn_view`. |
| `lighting` | `lighting soft` or `lighting full` | Set lighting mode. | Use declared style preset. |
| `silhouettes` | `graphics silhouettes true` | Add silhouettes. | Useful for white backgrounds. |
| `title_label` | `2dlabels text "<title>" xpos <float> ypos <float> size <int> color <color> bgColor none` | Add a title. | Use escaped title text. |
| `title_label_cleanup` | `2dlabels delete all` | Remove existing 2D labels before adding a title. | Use only when replacing the skill-owned title layer. |
| `align_model` | `matchmaker <query_model> to <reference_model>` | Align query model to reference. | Verify model IDs first. |

Forbidden patterns:
- semicolon-separated arbitrary command strings
- Python evaluation commands
- shell commands
- `runscript` over REST unless a task-specific contract allows it
- remote URLs unless the user explicitly authorizes network fetches

Session-manifest use:
- Prefer `--session-manifest <control_session.yaml>` over repeating ports in multi-turn collaboration.
- The manifest resolves the REST port only; command validation remains unchanged.
