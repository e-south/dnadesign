# Native/Core60 Shift Summary

## Purpose

Compare each native RegulonDB promoter source record to its derived TSS-upstream core60 view before treating the two row bases as interchangeable. This answers the length/pooling question directly instead of relying on UMAP layout.

## Inputs

- Native source-record Evo 2 7B sequence-mean views.
- TSS-upstream core60 Evo 2 7B core60-mean views.
- Parent promoter identity carried through materialized view rows.

## Outputs

- `native_core60_shift_summary`

## Interpretation

Higher paired self-cosine and lower L2 shift mean the native and core60 products preserve similar rowwise geometry in that representation family. A large shift means downstream comparisons should keep native and core60 views separate.
