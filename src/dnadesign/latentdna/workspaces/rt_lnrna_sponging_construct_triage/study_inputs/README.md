# RT-lnRNA study inputs

The Khan and Crawford YAML files are the only active source-order inputs for
this LatentDNA workspace.

Retired `reader_spop_*.parquet` snapshots are not retained here. They predated
the current Reader provenance contract, had no active producer, and were not
valid label or objective inputs.

New assay evidence uses the descriptive
`rt_lnrna_reporter_response_profile.v2` contract. It is not a LatentDNA label
source or an OPAL objective. No compatibility loader or placeholder remains
for the retired snapshots.
