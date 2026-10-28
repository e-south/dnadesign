## Observability and events

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
This document defines the event and diagnostics boundaries across DenseGen, SeqLedger, and Notify. Read it when you need to debug watcher wiring, understand where runtime failures are reported, or avoid mixing telemetry with mutation events.

### Core streams

- **DenseGen diagnostics** live at `outputs/meta/events.jsonl` and describe DenseGen runtime behavior such as stage transitions, failure reasons, and solver outcomes.
- **SeqLedger mutation events** live at `<dataset>/.events.log` and describe dataset write operations such as materialization and flush outcomes.
- **Notify input contract** is SeqLedger mutation events only; Notify should never consume DenseGen diagnostics directly.

### Source of truth by use case

- Use `outputs/meta/events.jsonl` when debugging DenseGen execution logic.
- Use `<dataset>/.events.log` when validating delivery to downstream watchers.
- Use both only when correlating runtime behavior with dataset mutations.

### Common mistakes

- Pointing Notify at `outputs/meta/events.jsonl` instead of SeqLedger `.events.log`.
- Assuming a successful DenseGen run implies a watcher is reading the right dataset.
- Running in dual-sink mode and forgetting which sink notebook/plot commands read.
- Reusing a stale Notify profile after changing workspace config paths.

### Where to go next

- Read the **[DenseGen to SeqLedger to Notify tutorial](../../tutorials/demo_usr_notify.md)** for an end-to-end validation flow.
- Read the **[Notify SeqLedger events operators guide](../../../../../../docs/notify/seqledger-events.md)** for watcher setup, spool, and drain operations.
- Read the **[SeqLedger package guide](../../../../seqledger/README.md)** for dataset/event schema and mutation semantics.
