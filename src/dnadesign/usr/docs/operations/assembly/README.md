## USR Assembly Runbooks

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-14

Use this folder when multiple source datasets or Construct handoffs need one
durable USR dataset boundary before Infer, Notify, Cluster, or OPAL continues.

- [Multi-source shared dataset](multi-source-shared-dataset.md): merge multiple USR-backed sources before Construct and Infer share one downstream dataset.
- [Construct -> USR -> Infer shared dataset](construct-infer-shared-dataset-runbook.md): use one Construct-backed dataset as the durable Infer handoff.
- [Permuter -> Construct -> Infer shared dataset](permuter-construct-infer-shared-dataset.md): route Permuter-originated RT-lnRNA (`rt_lnrna`) variants through study-owned construct-subject promotion, Construct context realization, and Infer-owned sidecars.
