# Name And Scope Decision

Current name: `chimerax-structure-review`.

Decision: keep the name for now.

Rationale:
- The durable user goal is reviewing molecular structures.
- Live session collaboration is the operating mode, not the only work product.
- The name remains generic and discoverable for prompts about opening, orienting, styling, inspecting, and rendering structures.

Rejected rename for now:
- `chimerax-session-collaboration`: accurate for the control mechanism, but too broad and less clear for structure-review prompts.
- `chimerax-visual-review`: close, but underplays inspection and command interop.

Revisit if:
- the skill begins to handle non-structure ChimeraX scenes as a first-class use case;
- another skill owns static structure review and this one becomes purely a live-control bridge.
