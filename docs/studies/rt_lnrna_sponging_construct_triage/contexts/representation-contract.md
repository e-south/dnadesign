---
doc_id: study-rt-lnrna-sponging-construct-triage-representation-contract
surface: study-context
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-05-22
---

## Representation Contract

The representation contract is
`dual_cassette_construct_context_embedding_v1`.

Start with exactly three representation views:

| View | Required source |
| --- | --- |
| `dual_cassette_1600bp_seq_mean` | One forward `realized_context` sequence view. |
| `dual_cassette_1600bp_fwd_rc_concat` | Forward plus reverse-complement `realized_context` sequence views, then an explicit downstream concatenation view. |
| `lnrna_span_in_construct_anchor_mean` | Full construct context with lnRNA span bounds declared for `anchor_mean` pooling. |

The views are study representation names, not new USR `product_kind` values.
Persist them through sequence-view names, aliases, and view semantics. Infer
must consume declared sequence views and must not reverse-complement, pad,
window, or infer missing spans implicitly.

Construct provides the coordinate authority. The focal lnRNA span remains
available through `construct__anchor_start` / `construct__anchor_end` for
existing `anchor_mean` consumers, and every named slot is auditable through
`construct__slots`. The RT CDS slot is required in the emitted context even when
the pooling operation targets the lnRNA slot.

### Failure Rules

- Missing constants fail before Infer.
- Invalid DNA alphabet fails before Infer.
- Any anchor, lnRNA slot, or RT CDS slot clipping or truncation fails before
  Infer.
- Missing reverse-complement view fails the concat alias.
- Missing lnRNA span fails the diagnostic anchor-mean view.

The diagnostic lnRNA view runs Evo2 over the full construct context and pools
the lnRNA span. It is not a naked lnRNA-only embedding.
