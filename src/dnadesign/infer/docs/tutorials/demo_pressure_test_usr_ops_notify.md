<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[Infer guide](../README.md)

# How a SeqLedger extraction job works

A stored inference result connects a sequence record to a model, job and requested output. This lets you inspect predictions alongside the input and resume processing without repeating completed work.

The [SeqLedger extraction procedure](../operations/pressure-test-agnostic-models.md) contains the runnable commands. This walkthrough explains what each stage establishes.

## Prepare records and output types

A workspace holds `config.yaml`; a SeqLedger dataset holds the sequence records. Creating a workspace does not populate the dataset. Import your sequences, then register the output names and types required by the job.

For example, the supplied extraction job requests a pooled logits vector and a mean log-likelihood. SeqLedger stores them as `list<float64>` and `float64`, respectively, under names of the form `infer__<model_id>__<job_id>__<out_id>`. The job's output id is a label; its function and parameters determine the calculation.

## Validate, then execute

Configuration validation checks the request. A dry run also resolves the input dataset and required field. Neither step loads model weights. Actual extraction requires the model runtime and hardware described in the [Evo2 reference](../reference/evo2-provider.md).

When extraction succeeds with write-back enabled, the values are attached to record IDs. Inspect both the resulting columns and dataset events before using them downstream.

## Resume deliberately

Running the same job again can skip completed outputs. Feature bundles also compare request metadata before reusing values. Changing a model, selector or pooling request can require recomputation; an existing column alone does not establish compatibility.

The procedure includes an optional `infer prune` step for archiving the Infer overlay before a fresh extraction. It is a reset operation, not part of a routine resume, and it does not remove sequence-view sidecars.

## Schedule the same job when needed

OPS can plan a scheduled run from the workspace configuration. Its `--no-submit` path checks the planned commands before `--submit` sends work to the scheduler. Notifications are optional. Local extraction does not require either OPS or Notify.

After extracting features, choose the output needed for [clustering](../../../cluster/docs/workflows/exploratory-clustering.md) or an [OPAL campaign](../../../opal/docs/workflows/seqledger-infer-x-active-learning.md). For sequence-view features, use the [documented reader or matrix export](../reference/feature-schema.md) instead of reading storage fragments directly.
