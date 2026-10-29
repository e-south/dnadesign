<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA guide](../README.md) / [Concepts](README.md)

# Tool ownership

LatentDNA reads precomputed representations and produces comparison artifacts. [Infer](../../../infer/) owns model execution, [SeqLedger](../../../seqledger/) owns stored sequence records and annotations, and downstream tools own their prediction or selection workflows.

The calling workspace defines biological groups, reference membership, thresholds and the interpretation of scores. A geometric association is an analysis result; deciding whether it supports a biological claim requires the relevant experimental evidence.

Keep study-specific configuration and data in an [external workspace](../../../../../docs/integrations/study-workspaces.md). Integrations should use documented CLI commands, snapshots and exported formats rather than importing internal analysis modules.
