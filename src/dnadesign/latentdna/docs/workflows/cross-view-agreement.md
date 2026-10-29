<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA guide](../README.md) / [Workflows](README.md)

# Compare structure across views

Use cross-view agreement when you want to know whether different representations place the same sequences near similar neighbors or in similar groups. This compares relationships between rows, so it does not require subtracting coordinates from different model spaces.

## Prepare and compare

1. Materialize the views and choose an explicit shared row population, using an alignment or sample where needed.
2. Fit a `neighbor_set` for each view with a declared metric and neighborhood size. Fit cluster assignments if cluster agreement is also needed.
3. Run `agreement compare` with the saved neighbor or cluster artifacts.
4. Render an `agreement_summary` plot from the resulting `agreement_set`.

For a workspace with two existing neighbor artifacts named `left_knn` and `right_knn`:

```bash
# Compare the saved neighborhoods for the same row population.
latentdna agreement compare neighborhood_agreement \
  --workspace /path/to/workspace \
  --left-neighbors left_knn --right-neighbors right_knn
```

Higher neighborhood overlap means more of the same nearby rows are recovered under the chosen settings. It does not establish equivalent biological information or predictive performance. Keep row coverage, metric and neighborhood size explicit when interpreting differences.

See [alignment](../reference/alignment-contract.md) for matching rows, [CLI reference](../reference/cli-contracts.md) for cluster options, and [context comparisons](context-shift.md) for direct paired geometry within compatible coordinates.
