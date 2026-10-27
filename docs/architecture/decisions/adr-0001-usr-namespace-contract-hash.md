## ADR 0001: Namespace-scoped compatibility hashes for USR overlays

**Status:** proposed
**Date:** 2026-03-22
**Owner:** dnadesign-maintainers

### Context
USR is the durable data-plane contract for sibling tools such as `construct`, `densegen`, `infer`, and `cluster`.
Today, overlay validation keys compatibility to `usr:registry_hash`, which is derived from the full serialized root `registry.yaml`.

That full-root hash is too broad for cross-tool compatibility:
- unrelated namespace additions can invalidate otherwise compatible overlays
- catalog-only edits such as owner and description changes participate in the hash
- tools become coupled through the entire shared registry instead of only the namespaces they emit

The current stabilization fix keeps whole-root hashing intact but removes accidental construct column-order drift.
This ADR defines the next, narrower compatibility boundary.

### Decision
USR will add a namespace-scoped compatibility hash for overlays:

- overlay writers will stamp both `usr:registry_hash` and `usr:namespace_contract_hash`
- `usr:namespace_contract_hash` will be computed from the namespace id plus ordered column name/type pairs only
- registry `owner` and `description` remain catalog metadata and are excluded from namespace compatibility hashing
- namespace-scoped validation will be opt-in through explicit registry modes:
  - `namespace-current`
  - `namespace-frozen`
  - `namespace-either`
- existing `current`, `frozen`, and `either` modes will keep their whole-root behavior during migration

### Rationale
- narrows coupling to the overlay namespace that a tool actually owns
- preserves explicit contracts and fail-fast behavior
- keeps migration reversible by dual-writing both hashes before any default-mode change
- avoids silent fallback by requiring operators to select namespace-scoped validation explicitly

### Consequences
- What becomes easier
  - cross-tool sharing when unrelated namespaces evolve independently
  - doc-only registry metadata edits without invalidating overlay compatibility
  - future namespace-level compatibility rules that do not depend on whole-root byte identity

- What becomes harder
  - operators must understand two validation scopes during migration
  - overlay metadata and registry-mode docs become slightly more complex
  - eventual default-mode migration still requires an explicit rollout decision

### Links
- Proposal: current branch tracer-bullet implementation
- PR(s): [#39](https://github.com/e-south/dnadesign/pull/39)
- Follow-ups: switch strict shared-root validation defaults after ecosystem migration
