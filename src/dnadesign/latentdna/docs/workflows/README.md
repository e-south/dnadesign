<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA guide](../README.md)

# Choose an analysis

Start with the question you want the representation to answer. Each workflow uses declared inputs and persists the comparison so plots and notebooks can review the same result.

| Goal | Required relationship | Workflow |
| --- | --- | --- |
| Quantify a change in representation | Paired rows in compatible coordinates | [Context comparisons](context-shift.md) |
| Compare structure across models or transformations | Shared row identity, even if coordinates differ | [Cross-view agreement](cross-view-agreement.md) |
| Score resemblance to references | A view and declared landmark membership | [Reference distances](control-distances.md) |
| Summarize labels near references | Neighbor sets, landmarks and cohort metadata | [Landmark neighborhoods](landmark-neighborhoods.md) |
| Use features in another tool | An explicit row basis and export blocks | [Export matrices and tables](export-opal-x.md) |

If you have not yet materialized a view, run the [local walkthrough](../getting-started/README.md) first. Use [coordinate-space concepts](../concepts/coordinate-spaces.md) to decide whether direct arithmetic or structural comparison is appropriate.
