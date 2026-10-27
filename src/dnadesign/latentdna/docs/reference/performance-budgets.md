# Performance budgets

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-27

LatentDNA keeps interactive control paths separate from array-heavy analysis.
Notebook controls and workspace summaries read manifest-backed shapes. Freshness
checks stream payload bytes to verify digests, sharing cached hashes across
dependencies without loading embedding arrays into memory. CLI help loads command
definitions without importing numerical analysis services.

## Public regression checks

Checked-in tests use small fixtures to verify:

- schema and artifact integrity;
- manifest-backed row and dimension reporting;
- bounded workspace and notebook control-plane work;
- explicit failure when required metadata is absent.

Fixture tests prove behavior, not production-scale throughput. Calling
workspaces should record representative timing and peak-memory evidence beside
their own input manifests.

## Runtime posture

- Publish row counts and dimensions in manifests so browsers do not open large
  array files.
- Use digest ledgers when rescanning large external source trees becomes the
  dominant freshness cost.
- Keep BLAS and OpenMP concurrency bounded. Two threads is the conservative
  default for mixed pipelines on 16 GB machines; raise it only after measuring
  the complete workload.
- Treat seeded UMAP as single-worker even when a wider process thread cap is
  configured.
- Measure cold start, warmed control-plane latency, analysis runtime, and peak
  resident memory separately. They answer different operational questions.

Study-scale measurements and hardware-specific thresholds belong with the
external workspace that produced them. Promote a benchmark here only when it
describes a stable public contract or a repository-wide default.
