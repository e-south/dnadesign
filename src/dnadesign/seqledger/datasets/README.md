# SeqLedger data roots

Keep sequence datasets outside the installed package. The CLI defaults to
`./seqledger_datasets`; select another location with `seqledger --root <path>`.
A data root contains a namespace registry and one directory per dataset.

The checked-in `registry.yaml` declares shared overlay namespaces. Copy or extend
it when preparing a data root for tools that write those namespaces.

See the [dataset layout](../docs/reference/dataset-layout-and-code-map.md) for
record, overlay and sequence-view storage, and the
[quickstart](../docs/getting-started/cli-quickstart.md) to create a dataset.
