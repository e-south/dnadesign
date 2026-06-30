# External Sources

Retrieved: 2026-06-27

| Source | Evidence posture | Used for |
| --- | --- | --- |
| UCSF ChimeraX `remotecontrol` command, https://www.cgl.ucsf.edu/chimerax/docs/user/commands/remotecontrol.html | Official web documentation; installed help may be used only as a local mirror. | REST start/stop, localhost endpoint, JSON response behavior. |
| UCSF ChimeraX `view` command, https://www.cgl.ucsf.edu/chimerax/docs/user/commands/view.html | Official web documentation; installed help may be used only as a local mirror. | Named views and the limitation that camera mode is not part of a named view. |
| UCSF ChimeraX `save` command, https://www.cgl.ucsf.edu/chimerax/docs/user/commands/save.html | Official web documentation; installed help may be used only as a local mirror. | Session-file and image-save behavior. |
| UCSF ChimeraX `transparency` command, https://www.cgl.ucsf.edu/chimerax/docs/user/commands/transparency.html | Official web documentation; installed help may be used only as a local mirror. | Surface/cartoon/atom transparency command templates. |
| UCSF ChimeraX `color` command, https://www.cgl.ucsf.edu/chimerax/docs/user/commands/color.html | Official web documentation; installed help may be used only as a local mirror. | Coloring targets, surface color behavior, and color names. |
| UCSF ChimeraX `show` command, https://www.cgl.ucsf.edu/chimerax/docs/user/commands/show.html | Official web documentation; installed help may be used only as a local mirror. | Showing atoms, cartoons, and surfaces by target. |
| UCSF ChimeraX `style` command, https://www.cgl.ucsf.edu/chimerax/docs/user/commands/style.html | Official web documentation; installed help may be used only as a local mirror. | Stick, ball, and sphere atom display styles for side-chain inspection. |
| UCSF ChimeraX `surface` command, https://www.cgl.ucsf.edu/chimerax/docs/user/commands/surface.html | Official web documentation; installed help may be used only as a local mirror. | Molecular surface creation and display behavior. |
| UCSF ChimeraX `turn` command, https://www.cgl.ucsf.edu/chimerax/docs/user/commands/turn.html | Official web documentation; installed help may be used only as a local mirror. | Same-session view rotation. |
| UCSF ChimeraX `wait` command, https://www.cgl.ucsf.edu/chimerax/docs/user/commands/wait.html | Official web documentation; installed help may be used only as a local mirror. | Waiting for animated view changes before later commands. |
| UCSF ChimeraX `2dlabels` command, https://www.cgl.ucsf.edu/chimerax/docs/user/commands/2dlabels.html | Official web documentation; installed help may be used only as a local mirror. | Title and figure-label placement. |
| Sibling `ec86kit` apply-map script | Optional sibling-repo pattern, not an import dependency or required runtime path. | Prior pattern for white background, silhouettes, surface coloring, and saved sessions. |
| Sibling `ec86kit` command utility | Optional sibling-repo pattern, not an import dependency or required runtime path. | Prior pattern for logging ChimeraX command execution. |
| Sibling `ec86kit` pairing script | Optional sibling-repo pattern, not an import dependency or required runtime path. | Prior pattern for chain-role aliases and role-specific coloring. |

Notes:
- Public UCSF documentation is the command authority.
- Installed local documentation can be consulted as a mirror, but checked-in
  skill docs must not record machine-local absolute paths.
- Sibling-project code is example material only; this skill must not depend on
  importing it or on a workstation-specific sibling checkout path.
