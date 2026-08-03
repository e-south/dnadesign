## RT-lnRNA Trait-Axis Projection

Fits separate Crawford and Khan abundance axes in the four intermediate
candidate-X LatentDNA views, then projects labeled, reference, compiler, and
held-out DMS populations onto those axes.

### rt_lnrna_trait_axis_projection_rows | Trait-axis projection rows

Row-level scalar with one projection record per scored
candidate/view/trait/endpoint/population role. Crawford and Khan are configured
as separate source-scoped traits.

### rt_lnrna_trait_axis_projection_summary | Trait-axis projection summary

Compact summary scalar for view-level trait-axis evidence: source-value
correlations, endpoint separation, endpoint stability, and fitted-axis
concordance where available.

### rt_lnrna_trait_axis_projection_concordance | Crawford-Khan axis concordance

Generic sidecar-table scalar that publishes `axis_concordance.parquet` as a
plot-ready table. It compares fitted Crawford and Khan axis vectors for matched
view, endpoint definition, and normalization policy without pooling their source
measurements.

### rt_lnrna_trait_axis_existence | Trait-axis existence

Source assay value versus trait-axis projection for one selected source-scoped
axis. The default rendered filter is Crawford; the notebook control can switch
to Khan without pooling their values.

### rt_lnrna_crawford_khan_axis_agreement | Crawford-Khan axis agreement

Direct fitted-vector concordance between Crawford and Khan axes by endpoint
definition and candidate-X view. This is a geometry agreement check, not a
merged abundance target.

### rt_lnrna_trait_axis_endpoint_sensitivity | Endpoint sensitivity

Endpoint-definition stability relative to each configured primary endpoint.
Unstable endpoint definitions should be treated as review risks before a view is
promoted as abundance geometry.

### rt_lnrna_trait_axis_view_scorecard | View scorecard

Source-value Spearman scorecard for candidate-X view selection across Crawford
and Khan trait axes. This replaces older proxy-only ordering for the trait-axis
decision funnel.

### rt_lnrna_reference_compiler_axis_projection | Reference and compiler projection

Fit, eval, reference, compiler, and DMS population placement on the selected
axis. Reference and DMS rows remain projection evidence, not fit labels.

Current claim boundary: this deliverable supports abundance-geometry audit and
candidate triage. It does not invent reporter-response labels, pool Crawford
and Khan abundance values, or treat DMS rows as fit evidence.
