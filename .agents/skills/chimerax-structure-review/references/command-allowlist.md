# Command Allowlist

Commands sent over REST must come from this table or a narrower study-specific command file.

| Key | Template | Purpose | Notes |
| --- | --- | --- | --- |
| `rest_port` | `remotecontrol rest port` | Confirm endpoint. | Read-only. |
| `rest_stop` | `remotecontrol rest stop` | Stop local control. | Use after capture. |
| `set_background` | `set bgColor <color>` | Set background color. | Prefer white for publication stills. |
| `camera_mode` | `camera ortho` or `camera mono` | Set camera mode. | Named views do not store camera mode. |
| `window_size` | `windowsize <width> <height>` | Set deterministic render dimensions. | Use positive declared integer dimensions. |
| `name_view` | `view name <pose_id>` | Save current view name in session. | Use stable IDs. |
| `restore_view` | `view <pose_id>` | Restore a named view. | Pair with explicit camera mode. |
| `fit_view` | `view <selection> pad <fraction>` | Fit the visible view around a declared selection. | Use before capture when a scene is cropped too tightly. |
| `save_session` | `save "<path>.cxs"` | Save review session. | Session is provenance, not the sole contract. |
| `save_image` | `save "<path>.png" width <int> height <int> supersample <int>` | Save still image. | Use declared dimensions. |
| `save_movie_frame` | `save "<frame-path>.png" width <int> height <int> supersample <int> transparentBackground false` | Capture one fixed-size frame in a checked scientific frame series. | Use numbered local paths and validate all four background corners before encoding. |
| `open_model` | `open "<path>"` | Open a local structure file. | Use only for declared local paths. |
| `rename_model` | `rename <model> <stable_name>` | Give an opened or generated model a semantic name. | Do not change IDs unless a separate task contract requires it. |
| `name_role` | `name <role_name> <selection>` | Create a reusable molecule-role selector. | Prefer `protein_role`, `dna_role`, and `rna_role`. |
| `close_session` | `close session` | Clear session. | Use only in setup scripts. |
| `show_selection` | `show <selection> <level>` | Show a declared selection. | Level must be explicit, such as `atoms`, `cartoons`, or `surfaces`. |
| `hide_selection` | `hide <selection>` | Hide a declared selection. | Selection must be explicit. |
| `nucleotide_ladder` | `nucleotides <selection> ladder` | Show nucleotide pair rungs and stubs attached to the native cartoon. | Default DNA/RNA representation for review scenes. |
| `nucleotide_atoms` | `nucleotides <selection> atoms` | Remove nucleotide slab, ladder, and filled-ring depictions. | Pair with the native nucleic cartoon and explicit atom sticks. |
| `show_only_chain` | `hide <model> target acs` then `show <model>/<chain> cartoons` | Focus one declared chain. | This is two explicit commands, not an inferred inverse selection. |
| `atom_style` | `style <selection> stick` or `style <selection> ball` or `style <selection> sphere` | Change atom/bond display style for an explicit selection. | Show complete nucleotide atoms when connected sugar/base geometry matters. |
| `stick_radius` | `size <selection> stickRadius <float>` | Set atom-stick weight. | Keep nucleotide sticks thin enough that the cartoon remains legible. |
| `surface_selection` | `surface <selection>` | Draw molecular surface. | Use with transparency if needed. |
| `coulombic_surface` | `coulombic <selection> palette red-white-blue range <min>,<max> key true` | Calculate and display qualitative Coulombic surface potential. | Record the selection, range, charge method, dielectric defaults or overrides, and tool version. Do not describe the result as binding energy. |
| `color_key_cleanup` | `key delete` | Remove the skill-owned electrostatic color key before another scene. | Use only after a skill-owned key was created. |
| `surface_transparency` | `transparency <selection> <percent> target s` | Make a surface faint or opaque. | Percent is transparency, not opacity; the shared review value is `35`, equivalent to 0.65 alpha. |
| `cartoon_transparency` | `transparency <selection> <percent> target r` | Make cartoons faint or opaque. | Use sparingly for overlays. |
| `color_selection` | `color <selection> <color> target <targets>` | Color a selection. | Include `f` for ring fill and nucleotide representations; use `acf` when a chain's atoms, cartoon, and ladder must match. |
| `cartoon_style` | `cartoon style width <float> thick <float>` or `cartoon style nucleic xsect <oval|rectangle|barbell> width <float> thick <float>` | Tune ribbon visual weight. | Use the nucleic-specific form for a visible backbone ribbon without changing the protein cartoon style. |
| `cartoon_backbone_suppression` | `cartoon <selection> suppressBackboneDisplay true` | Keep the native nucleic cartoon while suppressing overlapping phosphate-backbone atoms. | Displayed C3'/C4' atoms remain eligible for cartoon tethers. |
| `cartoon_tether` | `cartoon tether nucleic shape cylinder sides <int> scale <float> opacity <float>` | Connect displayed nucleotide atoms to the native cartoon. | This prevents detached sugar/base sticks without custom shape models. |
| `phosphate_ribbon_fallback` | `shape ribbon <chain>@P width <float> height <float> followBonds false color <color> modelId <#N>` | Supply a continuous chain path when a deposited complex has no native nucleic cartoon. | Use only after visual confirmation; pair with complete nucleotide atoms and rename each generated model. |
| `turn_view` | `turn x <degrees> <frames>` or `turn y <degrees> <frames>` or `turn z <degrees> <frames>` | Rotate the current scene for a visible same-session view change. | Use `wait` after animated turns. |
| `wait` | `wait <frames>` | Wait for an animated command to finish. | Use after `turn_view`. |
| `movie_record` | `movie record [size <width>,<height>]` | Start ordinary frame capture for a declared structure story. | Use the documented command path when its buffer passes the final-aspect background check; otherwise use checked `save_movie_frame` output. |
| `movie_encode` | `movie encode "<path>.mp4" framerate <int>` | Encode captured frames as MP4. | Output must be a declared local path; record the final hash in the pose or render manifest. |
| `lighting` | `lighting soft` or `lighting full` | Set lighting mode. | Use declared style preset. |
| `silhouettes` | `graphics silhouettes true` | Add silhouettes. | Useful for white backgrounds. |
| `title_label` | `2dlabels text "<title>" xpos <float> ypos <float> size <int> color <color> bgColor none` | Add a title. | Use escaped title text. |
| `title_label_cleanup` | `2dlabels delete all` | Remove existing 2D labels before adding a title. | Use only when replacing the skill-owned title layer. |
| `source_label_cleanup` | `label delete` | Remove automatic 3D labels before capture. | Use after opening structures that carry missing-segment labels. |
| `pseudobond_cleanup` | `hide <selection> pseudobonds` | Hide missing-segment and other source pseudobonds. | Use when pseudobonds are not evidence in the intended view. |
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
