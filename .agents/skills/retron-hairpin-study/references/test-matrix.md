# Test Matrix

| Scenario | Prompt | Expected Behavior | Pass/Fail |
| --- | --- | --- | --- |
| Trigger positive | `Compile pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM into a design catalog.` | Use the Retron MSD compiler route, lint/compile the ID, and report the explicit output directory. | Pass if the route starts from input completeness and does not run study status first. |
| Trigger positive | `Generate a multicopy ssDNA reference and visual handoff from this payload, cap, left base, and right base.` | Compile a design reference first, then route sequence or visual output through Construct, Folding, and BaseRender service handoffs. | Pass if Construct/Folding are treated as services, not per-ID workspaces. |
| Trigger negative | `Run a generic Cruncher snapback search for another project.` | Do not use this skill; route to generic Cruncher docs or commands. | Pass if Retron MSD study routing is not invoked. |
| Functional core | Complete MSD label with matching profile. | Parse fields, recompute `S3/S2/S1/S0`, require `S0=M`, and emit `msd_design_reference_v1` / `msd_design_catalog_v1`. | Pass if `uv run python -m dnadesign.studies.retron_hairpin_design.cli lint` succeeds. |
| Functional edge | Complete MSD label with a profile that does not match left/right bases. | Fail fast and explain profile drift; do not generate a catalog. | Pass if lint exits nonzero and includes `provided profile` in the error. |
| Reliability | Repeated naive-agent route checks for complete ID, missing profile, and generic Cruncher prompt. | Produce the same route class across repeated runs. | Pass if all runs select compile, compiler-derived profile, and route-away respectively. |
