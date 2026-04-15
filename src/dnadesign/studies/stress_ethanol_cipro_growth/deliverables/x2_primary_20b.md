# X2 Primary 20B

`x2_primary_20b` is the aligned anchor-plus-expanded-context export bundle for
downstream supervised benchmarking.

The bundle should carry the anchor-only pooled space, the 1 kb
expanded-context pooled space, the centered reference-margin sidecars with
explicit `anchor_ref_*` and `seq_ref_*` feature names, the core
context-audit scalars, and the 20B anchor/context mean-per-token
log-likelihood side channels on one explicit row basis.
Use it when the next question is model selection or grouped benchmark
performance, not when the task is exploratory atlas browsing.
