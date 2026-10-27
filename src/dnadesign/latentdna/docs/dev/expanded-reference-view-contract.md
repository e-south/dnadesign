# Reference and view coverage

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-26

A workspace declares reference membership, representation views, and the plots
that compare them. Tool code resolves those declarations without choosing
biological reference groups or interpreting their names.

## Reference membership

Declare named `reference_sets` with explicit IDs or metadata selectors. Use
`display_labels` for readable names and `label_mode` to choose callouts or
highlight-only markers. Each rendered panel records expected and matched IDs.
Sampling must preserve configured reference rows without expanding an entire
metadata ledger into Python objects.

## Views and plots

View tags describe the model, representation family, and sequence context.
Notebook controls enumerate configured, available views. Missing artifacts are
reported explicitly; their absence does not count as a successful comparison.

| Surface | Contract |
| --- | --- |
| Projection scatters | Resolve reference membership independently in each panel and record completeness. |
| Scree curves | Read each configured reducer and report missing artifacts. |
| Reference summaries | Use explicit contrasts, reference sets, or metadata grouping columns. |
| Strength comparisons | Keep collections separate when their measurements have different scales. |
| Distance distributions and heatmaps | Compare the configured populations and declared subsets. |
| Notebook controls | Expose available views and configured reference modes. |

Labels, compact labels, category order, and colors belong in `metadata.axes`.
Promoter-specific extraction, family assignment, and activity interpretation
belong in the calling study.

## Verification

Run the sampling, plot-rendering, scalar-builder, and notebook-control tests
for the changed surface. A deep workspace validation checks declared inputs;
regenerating a large projection is a separate computation.

```sh
uv run latentdna validate workspace --workspace /path/to/workspace --deep
uv run latentdna workspace snapshot --workspace /path/to/workspace --json
```
