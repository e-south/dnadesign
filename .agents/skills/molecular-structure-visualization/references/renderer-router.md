# Renderer Router

Choose one primary rendering surface before styling.

| Need | Primary surface | Route |
| --- | --- | --- |
| Interactive notebook or browser inspection | py3Dmol through `dnadesign.thread.structure_views` | Use `py3dmol-rendering-contract.md`. |
| Manual camera orientation and pose approval | ChimeraX | Use `chimerax-structure-review`. |
| Coulombic surface calculation | ChimeraX | Use `chimerax-structure-review`; record charge method and range. |
| Publication still or scripted rotation movie | ChimeraX | Use `chimerax-structure-review` capture and verification contracts. |
| Cross-renderer review | Both, with one declared primary artifact | Use `cross-renderer-contract.md`; compare semantics, not low-level primitives. |

Do not make py3Dmol launch ChimeraX, and do not embed ChimeraX GUI lifecycle in
a notebook materializer. Optional desktop renders must remain explicit.
