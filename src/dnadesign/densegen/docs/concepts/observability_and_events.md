## Observability and events

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


This document defines the event and diagnostics boundaries across DenseGen, USR, and Notify. Read it when you need to debug watcher wiring, understand where runtime failures are reported, or avoid mixing telemetry with mutation events.

### Core streams
This section defines each stream and what system is allowed to consume it.

- **DenseGen diagnostics** live at `outputs/meta/events.jsonl` and describe DenseGen runtime behavior such as stage transitions, failure reasons, and solver outcomes.
- **USR mutation events** live at `<dataset>/.events.log` and describe dataset write operations such as materialization and flush outcomes.
- **Notify input contract** is USR mutation events only; Notify should never consume DenseGen diagnostics directly.

### Source of truth by use case
This section maps common operator questions to the correct event source.

- Use `outputs/meta/events.jsonl` when debugging DenseGen execution logic.
- Use `<dataset>/.events.log` when validating delivery to downstream watchers.
- Use both only when correlating runtime behavior with dataset mutations.

### Common mistakes
This section lists the recurring footguns that cause silent confusion in event-driven workflows.

- Pointing Notify at `outputs/meta/events.jsonl` instead of USR `.events.log`.
- Assuming a successful DenseGen run implies a watcher is reading the right dataset.
- Running in dual-sink mode and forgetting which sink notebook/plot commands read.
- Reusing a stale Notify profile after changing workspace config paths.

### Where to go next
This section points to the task guides that operationalize these boundaries.

- Read the **[DenseGen to USR to Notify tutorial](../tutorials/demo_usr_notify.md)** for an end-to-end validation flow.
- Read the **[Notify USR events operators guide](../../../../../docs/notify/usr-events.md)** for watcher setup, spool, and drain operations.
- Read the **[USR package guide](../../../usr/README.md)** for dataset/event schema and mutation semantics.
