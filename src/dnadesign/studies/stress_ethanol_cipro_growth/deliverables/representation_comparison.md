# Representation comparison

This deliverable compares the eight realized candidate spaces without collapsing them into a hidden winner score. The plotted summaries stay tied to named evidence metrics and named context-stability metrics.

### representation_tradeoff_scatter | Candidate evidence vs context stability

#### Plot details

**Data.** Each point is one of the eight realized candidate representation spaces. The panels compare a realized evidence summary with the same context-stability summary.

**Definition.** The ethanol panel plots `wildtype_margin_ethanol_auprc` against `context_self_cosine_median`. The ciprofloxacin panel plots `wildtype_margin_cipro_auprc` against `context_self_cosine_median`. The dual panel plots `wildtype_margin_dual_joint_auprc` against `context_self_cosine_median`. The reference-neighbor panel plots `reference_in_knn_rate` against `context_self_cosine_median`.

**Interpretation.** Read the x-axis as the named evidence metric in that panel, not as a generic biology score. The y-axis is median context self-cosine, so higher values indicate more stable anchor-to-context embeddings. The plot is a tradeoff view across candidate spaces.
