# Test Matrix

| Scenario | Prompt | Expected Behavior | Pass/Fail |
| --- | --- | --- | --- |
| Trigger positive | `Make the SPOP heatmap with structure thumbnails.` | Open the RT-lnRNA condition-structure route, then run or report the composite materializer. | Pass if the route does not start in LatentDNA or retron-hairpin design generation. |
| Trigger positive | `Build the condition structure matrix for Reader SPOP.` | Use `reader_spop_composite.materialize` and report matrix rows, conditions, variants, and missing cells. | Pass if missing condition cells remain missing. |
| Trigger positive | `Plot variant rows with ViennaRNA structure next to SPOP conditions.` | Resolve thumbnails from retron-hairpin materialized outputs and render the RT-lnRNA plot. | Pass if missing thumbnails route to retron-hairpin materialization only. |
| Trigger negative | `Run a generic LatentDNA UMAP.` | Do not use this skill. | Pass if generic LatentDNA routing is selected. |
| Trigger negative | `Design new retron MSD sequences.` | Do not use this skill; route to `retron-hairpin-study`. | Pass if no Reader SPOP materializer is run. |
| Contract | Rebuild the plot. | Run `uv run python -m dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_spop_composite.materialize --reader-root ../reader --json`. | Pass if parquet, PNG, SVG, and manifest outputs exist under the RT-lnRNA workbench output root. |
| Contract | Inspect plot style. | Read `plots/manifest.json` after rebuild. | Pass if tiles are square, missing cells are white, low measured cells are not white, the row category band is left of the heatmaps, the x-axis title is blank, the y-axis label names lnRNA variants in the retron Eco1 system, primitive columns are present, and thumbnails are cap-right vector primitives with upright nucleotide labels and no structure-column border. |
| Contract | Explain cross-experiment rigor. | Read `normalization_basis` and `normalization_scope` in `plots/manifest.json`. | Pass if the response states the plot uses within-observation Reader SPOP normalized derepression, not raw absolute fluorescence across experiments. |
| Contract | Plot with a stale available-thumbnail path. | Render through `reader_spop_composite.render`. | Pass if the renderer raises an explicit contract error instead of silently drawing `na`. |
| Architecture | Review imports. | Keep SPOP composite semantics inside the RT-lnRNA study unit. | Pass if generic LatentDNA modules do not import `reader_spop_composite`. |
| Skill audit | Run the skill audit script. | Validate frontmatter, route references, trigger language, and line budget. | Pass if the audit exits zero. |
