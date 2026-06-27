# Natural-Language Control Map

Translate user phrases to allowlisted command keys. Do not send user prose as a command.

| User intent | Command keys | Required parameters |
| --- | --- | --- |
| Capture this pose. | `name_view`, `save_session`, `save_image` | `pose_id`, output directory, dimensions. |
| Stop control. | `rest_stop` | port. |
| Use a white background. | `set_background` | `color=white`. |
| Make the surface faint. | `surface_selection`, `surface_transparency` | selection, transparency percent. |
| Make the surface shield visible but subtle. | `surface_selection`, `surface_transparency`, `color_selection` | selection, color, transparency percent. |
| Show only one chain. | `show_only_chain` | model ID, chain ID, display levels. |
| Show side chains. | `show_selection`, `atom_style`, `color_selection` | explicit residue or chain selection using the `sidechain` or `sideonly` atomspec class. |
| Color a chain or region. | `color_selection` | selection, color, target. |
| Rotate or change the view. | `turn_view`, `wait`, `fit_view` | axis, degrees, frames, optional fit selection. |
| Add a title. | `title_label` | title, position, size, color. |
| Align this model to the reference. | `align_model` | query model ID, reference model ID. |
| Render a still image. | `save_image` | path, width, height, supersample. |

Parameter defaults:
- background: `white`
- render size: `1800 x 1200`
- supersample: `2`
- surface transparency for a faint shield: `70`
- title position: `xpos 0.035`, `ypos 0.89`
- title size: `30`

Escalate instead of acting when:
- the target selection is ambiguous
- the user asks for an unsupported command
- the same live session may be under manual manipulation
- output paths are not declared
