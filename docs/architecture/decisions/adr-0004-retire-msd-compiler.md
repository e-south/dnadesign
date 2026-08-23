## ADR 0004: Retire the Retron MSD compiler

**Status:** accepted
**Date:** 2026-08-22
**Owner:** dnadesign-maintainers

### Context

`dnadesign.msd` combined generic hairpin composition with Retron labels,
registries, and study-specific compilation policy. HOP now owns the generic
hairpin design and processing product as a standalone, versioned package.
Research Studies owns Retron identity, lineage, context, interpretation, and
product publication. Construct and Folding retain their generic placement and
assessment roles.

The Research Studies migration branch moves its default physical-record
publication path and its independent RT-lnRNA consumer to pinned HOP products.
The final supported predecessor comparison found no unexplained molecular
difference and was persisted before its executable adapter was removed. That
branch contains no executable `dnadesign.msd` import. The current Research
Studies main branch remains reproducible because it pins an earlier immutable
DNA Design revision while its protected migration PR is pending.

### Decision

Remove `dnadesign.msd`, its documentation, tests, banner, architecture edge,
and coverage registration. Do not retain an import alias, HOP forwarding
facade, schema reader, or fallback compiler.

Keep `MsdDesignCatalogV1` and `MsdDesignReferenceV1` in the neutral sequence
contracts package for now. Research Studies still consumes those contracts in
review and artifact-validation code. Their later retirement requires a
separate consumer audit and does not block removal of the compiler package.

This decision does not remove Cruncher hairpin mechanics. Those capabilities
require their own function-level consumer and parity audit.

### Alternatives considered

- Retain `dnadesign.msd` as a facade over HOP. Rejected because it would
  preserve an unnecessary dependency direction and two public names for one
  authority.
- Freeze the package as a historical compiler. Rejected because executable
  historical code can still be imported and mistaken for the supported path;
  the durable parity and lineage records preserve the necessary evidence.
- Move Retron registries and study policy into HOP. Rejected because it would
  couple a public hairpin utility to one biological application.

### Consequences

- HOP is the only generic hairpin compiler authority.
- Research Studies fails explicitly when a requested production method is not
  supported; DNA Design cannot provide a silent substitute.
- Construct, Folding, BaseRender, and the neutral sequence contracts remain
  independent reusable services.
- Historical compiler specifications may remain when immutable study lineage
  cites them, but DNA Design no longer executes them.
- Reintroducing a compiler or compatibility surface requires a new decision
  with a demonstrated consumer not served by HOP and study-owned records.

### Evidence

- HOP release: [v0.1.0a6](https://github.com/e-south/hop-design/releases/tag/v0.1.0a6)
- Research Studies migration: [draft PR #13](https://github.com/e-south/research-studies/pull/13)
- Construct handoff boundary: [ADR 0002](adr-0002-generic-linear-ssdna-composition.md)
