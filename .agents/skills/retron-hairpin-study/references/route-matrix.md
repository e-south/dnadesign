# Route Matrix

Use this matrix to route Retron MSD compiler tasks without defaulting to
study-phase reporting.

| User question | Primary surface | Why |
| --- | --- | --- |
| "Here is an MSD ID." | `msd-design-references.md` | Complete labels should lint/compile directly. |
| "I have payload, cap, left base, right base, and repeat count." | `msd-design-references.md`, then compiler CLI | User supplied parts; do not run solvers unless validation fails or metadata is missing. |
| "I have left/right bases but no profile." | compiler parser | The profile is derived from bases and fails fast if `S0` is not `M`. |
| "I have a profile but not bases." | scar-nick route in `routes.md` | Base feasibility is a primitive search problem. |
| "I need a cap or shortening geometry." | released-product Snapback route in `routes.md` | Cap/shortening is solved by Snapback, not the compiler. |
| "I need mismatch/boundary illustration." | YIU route in `routes.md` | YIU is contrast-only. |
| "I need sequence, SVG, or GenBank for selected parts." | compile reference, then Construct/Folding/BaseRender service handoff | Service tools act after part selection and should write to explicit output dirs. |
| "Where is the old study status?" | `cruncher-study-status --study-dir docs/studies/retron_hairpin_design --json` | Use only for progress/history questions. |
| "What blocks a legacy study run?" | `cruncher-study-preflight --study-dir docs/studies/retron_hairpin_design --scope next --json` | Use only for blocker/readiness questions. |

## Compiler-First Boundary

- Start with input completeness, not study phase.
- If all required parts are present, validate and compile.
- If a part is missing, route to the smallest primitive solver.
- Use `pipeline.yaml` only when a machine-readable command group is needed.
- Do not reconstruct compiler behavior from generic Cruncher docs.
