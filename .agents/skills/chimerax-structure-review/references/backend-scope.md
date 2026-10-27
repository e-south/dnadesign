# Backend Scope

`chimerax-structure-review` owns ChimeraX-specific execution:

- graphical session lifecycle;
- localhost REST control;
- ChimeraX molecule styles and named selections;
- manual camera and pose handoff;
- Coulombic surfaces;
- still, session, and movie capture.

`molecular-structure-visualization` owns renderer-neutral molecule roles,
coordinate integrity, py3Dmol behavior, and cross-renderer mapping.

Do not place py3Dmol API details in this skill. Do not place ChimeraX process or
REST lifecycle in the renderer-neutral skill.
