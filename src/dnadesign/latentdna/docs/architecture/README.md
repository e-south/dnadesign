<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA guide](../README.md)

# Architecture and contribution routes

The CLI resolves workspace configuration and delegates to services. Services read source adapters, run analysis primitives and persist artifacts with manifests. Plotters and notebooks consume those saved results.

| Change | Owning code |
| --- | --- |
| Command arguments and output formatting | [CLI](../../src/cli/) |
| Input formats and reads | [Source adapters](../../src/sources/) |
| Configuration and validation | [Contracts](../../src/contracts/) and [workspaces](../../src/workspaces/) |
| Analysis orchestration | [Services](../../src/services/) |
| Plot rendering | [Plots](../../src/plots/) |
| Generated notebook review | [Notebooks](../../src/notebooks/) |

Keep source identity, row alignment and coordinate-space checks explicit. Scientific cohort meanings, thresholds and acceptance decisions belong in the calling workspace. The [ownership guide](../concepts/ownership-boundary.md) explains these boundaries.

Use [analysis contracts](../dev/README.md) for reference handling and trait-axis projection. Tests live under [tests](../../tests/); start with the relevant contract or CLI module, then follow the [repository change gates](../../../../../docs/repo-change-gate.md).
