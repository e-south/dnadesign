# Native/Core60 Shift Summary

## Purpose

Compare each native RegulonDB promoter source record to its derived TSS-upstream core60 view before treating the two row bases as interchangeable. This answers the length/pooling question directly instead of relying on UMAP layout.

## Inputs

- Native source-record Evo 2 7B sequence-mean views, pooled over token
  positions in the stored source-record orientation.
- TSS-upstream core60 Evo 2 7B core60-mean views, pooled over token positions
  in the configured forward core60 window.
- Parent promoter identity carried through materialized view rows.

## Outputs

- `native_core60_shift_summary`

### native_core60_shift_summary | Native/Core60 Shift Summary

#### Plot details

Metric panels compare paired native versus core60 row shifts for both 7B
intermediate embeddings and 7B output-layer means. The plot is
representation-family agnostic: native/core60 alignment metrics come from
configured paired views, not promoter-box centering assumptions. The comparison
is between stored row vectors after pooling, not between raw sequences.

## Interpretation

Higher paired self-cosine and lower L2 shift mean the native and core60 products preserve similar rowwise geometry in that representation family. A large shift means downstream comparisons should keep native and core60 views separate.
