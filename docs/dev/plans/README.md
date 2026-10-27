## Design Proposals

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-08

Design proposals capture intent and tradeoffs before or during implementation.
Use semantic lanes instead of a flat chronology.

### Lanes

- `tools/<tool>/`: proposals owned mostly by one tool.
- `cross-tool/<topic>/`: proposals that define cross-package contracts or
  handoffs.
- `archive/`: inactive historical proposals retained for traceability.

### Current Proposals

- [Cruncher proposals](tools/cruncher/)
- [DenseGen proposals](tools/densegen/)
- [USR proposals](tools/usr/)

Study-owned proposals live in their private study workspaces. Promote a
reusable boundary into this index only when it changes a public dnadesign
contract.
