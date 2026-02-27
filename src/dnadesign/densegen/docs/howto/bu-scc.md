## DenseGen on BU SCC

This how-to guide captures BU SCC specifics that differ from the base DenseGen HPC runbook. Read it when running DenseGen on BU SCC and you need the correct scheduler flags, certificate setup, and BU docs.

For scheduler-agnostic DenseGen run patterns, use the **[DenseGen HPC runbook](hpc.md)**.

### BU SCC-specific differences
This section lists the operational details that are specific to BU SCC.

- Scheduler is SGE (`qsub`, `qrsh`, `qstat`).
- Project/account flags are required (typically `-P <project>`).
- Webhook delivery may require explicit certificate bundle export.

### Minimal environment setup
This section gives a short environment preflight for watcher-enabled runs.

```bash
# Export BU SCC certificate bundle for TLS webhook delivery when required.
export SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem

# Validate the workspace config before scheduling.
uv run dense validate-config --probe-solver -c /abs/path/to/workspace/config.yaml
```

### BU SCC reference docs
This section points to BU SCC repo-level docs so this page stays concise and non-duplicative.

- Use the **[BU SCC quickstart](../../../../../docs/bu-scc/quickstart.md)** for login-to-first-job flow.
- Use the **[BU SCC install guide](../../../../../docs/bu-scc/install.md)** for environment bootstrap.
- Use the **[BU SCC batch plus Notify runbook](../../../../../docs/bu-scc/batch-notify.md)** for watcher deployment patterns.
- Use the **[BU SCC job templates](../../../../../docs/bu-scc/jobs/README.md)** for submit-ready scripts.

### Event boundary reminder
This section links to the event-boundary doc to avoid semantic drift.

For DenseGen diagnostics versus USR mutation event boundaries, read **[observability and events](../concepts/observability_and_events.md)**.
