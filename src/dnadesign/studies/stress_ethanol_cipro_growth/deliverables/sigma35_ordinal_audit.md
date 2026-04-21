# Sigma-35 Ordinal Audit

The audit measures whether each candidate space preserves the declared order across the five Sigma-35 variants.

### sigma35_ordinal_audit | Sigma-35 ordinal audit

#### Plot details

**Data.** Each candidate is evaluated on a stratified sample over the synthetic promoter population using the workspace Sigma-35 order mapping.

**Preprocessing.** All centroid distances use view-level standardization followed by row L2 normalization. Study builders use the collapse-tolerant path: zero-variance columns are set to `0.0` after scaling, and zero-norm rows remain `0.0`.

**Definition.** For the declared Sigma-35 order, the audit compares expected
rank gaps

$$
\Delta_{\mathrm{rank}}(g,h) = |r_g - r_h|
$$

to observed centroid distances

$$
d_{\mathrm{emb}}(g,h) = 1 - \cos(c_g, c_h).
$$

It reports Spearman and Kendall correlations between those two vectors, plus a
balanced Sigma-35 Spearman, a within-family mean Spearman, a within-regulator
mean Spearman, and a shuffled-label permutation p-value. Confidence intervals
are reported for the Spearman-based summary rows.

**Decision use.** Ordered Sigma-35 structure adds a within-design signal beyond coarse cohort separation.

**Limits.** The current Sigma-35 order file stays exploratory until the exact literature note is checked into the repo. The output measures ordered design structure in pooled embeddings. It does not estimate promoter activity.
