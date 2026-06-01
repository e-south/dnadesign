## Orchestration runbooks

**Type:** runbook
**Plane:** control-plane
**Owner-boundary:** ops
**Entry artifact:** workspace-scoped ops runbook YAML plus scheduler and optional notify intent
**Exit artifact:** audit JSON, rendered command plan, and optional submit or watcher contract
**Registry-id:** ops.control-plane.orchestration
**Summary:** Deterministic control-plane runbook contract for DenseGen or Infer batch submit flows with optional Notify chaining.
**Execution-kind:** executable
**Status-kind:** ops-audit-json

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-27

This contract defines machine-readable runbooks for cross-tool BU SCC control-plane orchestration.
It does not own durable USR-backed data-plane workflows; return to the root docs router or USR operations docs when the next procedure is about shared datasets rather than scheduler sequencing.

### Why this exists

1. Keep orchestration policy outside package internals (`densegen`, `infer`, `notify`).
2. Force deterministic sequencing (`preflight -> notify_smoke -> submit`) with fail-fast behavior.
3. Keep notify as a decoupled contract so batch-only routes do not require watcher/smoke wiring.
4. Keep notify smoke safe by default (`dry`) unless a user explicitly sets `live`.
5. Emit audit JSON for reproducible review and traceability.

### Runbook bootstrap path

Generate a valid runbook scaffold before planning or executing:

```bash
uv run ops runbook init \
  --workflow densegen \
  --runbook <path-to-runbook.yaml> \
  --workspace-root <workspace-root> \
  --repo-root <repo-root> \
  --project <project> \
  --id <runbook-id>
```

The generic `init` path does not infer site identity. Provide exactly one of:

1. `--project <project>` for an explicit scheduler account or project id.
2. `--preset <preset>` for an explicit site-owned shortcut such as `bu-scc-dunlop`.

Explicit preset example:

```bash
uv run ops runbook init \
  --workflow densegen \
  --runbook <path-to-runbook.yaml> \
  --workspace-root <workspace-root> \
  --repo-root <repo-root> \
  --preset bu-scc-dunlop \
  --id <runbook-id>
```

DenseGen scaffolds include notify by default; add `--no-notify` only when you explicitly want batch-only submit.
Use a workspace-scoped runbook path to avoid root-level clutter and repeated ad-hoc YAML fan-out; recommended pattern: `<workspace-root>/outputs/logs/ops/runbooks/<runbook-id>.yaml`.
`<project>` must be the scheduler account or project id used by that workspace or study.

Optional resource overrides for prompt-driven runs:

```bash
uv run ops runbook init \
  --workflow densegen \
  --runbook <path-to-runbook.yaml> \
  --workspace-root <workspace-root> \
  --repo-root <repo-root> \
  --h-rt 02:00:00 \
  --pe-omp 12 \
  --mem-per-core 8G
```

Infer scaffold:

```bash
uv run ops runbook init \
  --workflow infer \
  --runbook <path-to-runbook.yaml> \
  --workspace-root <workspace-root> \
  --repo-root <repo-root> \
  --project <project> \
  --id <runbook-id>
```

Infer scaffolds also include notify by default; add `--no-notify` only when you explicitly want batch-only submit.

Use `--repo-root` when the runbook file is outside the repository so template paths resolve to runnable qsub templates.
Relative `--workspace-root` values are resolved against `--repo-root` (or current working directory when `--repo-root` is omitted) before the runbook is written.

DenseGen batch-only scaffold:

```bash
uv run ops runbook init \
  --workflow densegen \
  --runbook <path-to-runbook.yaml> \
  --workspace-root <workspace-root> \
  --repo-root <repo-root> \
  --project <project> \
  --id <runbook-id> \
  --no-notify
```

### Packaged presets

List packaged starter runbooks:

```bash
uv run ops runbook presets
```

`ops runbook presets` now lists two explicit preset families:

1. `init_presets`: named site/account shortcuts used by `ops runbook init --preset ...`
2. `presets`: packaged starter runbooks that you copy or materialize into a workspace before planning or execution

Current init presets include:

1. `bu-scc-dunlop`

Current presets include:

1. `src/dnadesign/ops/runbooks/presets/densegen_stress_ethanol_cipro_batch.yaml`
2. `src/dnadesign/ops/runbooks/presets/densegen_stress_ethanol_cipro_batch_with_notify.yaml`
3. `src/dnadesign/ops/runbooks/presets/infer_regulondb_native_promoter_core60_tss_upstream_7b_batch_with_notify.yaml`
4. `src/dnadesign/ops/runbooks/presets/infer_regulondb_native_promoter_native_full_7b_batch_with_notify.yaml`
5. `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_only_20b_batch_with_notify.yaml`
6. `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_plus_template_20b_batch_with_notify.yaml`
7. `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_sequence_views_anchor_construct_insert_7b_batch_with_notify.yaml`
8. `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_sequence_views_context_forward_seq_and_anchor_mean_7b_batch_with_notify.yaml`
9. `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_sequence_views_context_reverse_complement_seq_and_anchor_mean_7b_batch_with_notify.yaml`
10. `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_sequence_views_reference_analysis_window_core60_7b_batch_with_notify.yaml`
11. `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_sequence_views_reference_context_forward_seq_and_anchor_mean_7b_batch_with_notify.yaml`
12. `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_sequence_views_reference_context_reverse_complement_seq_and_anchor_mean_7b_batch_with_notify.yaml`

Infer auto/resume resolves exactly one USR write-back destination or one sequence-view input dataset per
runbook. If a study has separate anchor, forward-context, and reverse-complement context datasets or
view selectors, ship one batch preset per operational lane instead of one multi-destination Infer
preset.

Infer runbooks whose config uses `feature_bundle.sequence_view_inputs` render an
extra preflight command before `infer run --dry-run`:
`uv run infer validate sequence-view-completion --format json
--max-missing-products 0 --max-stale-vectors 0`. This gate is intentionally not
`--max-missing-vectors 0`: missing feature vectors are the batch workload, while
missing sequence products and stale vectors are submit blockers.

Keep `presets/` for reusable starters only. Do not point `ops runbook plan` or `ops runbook execute` at an installed-package preset in place. Materialize a workspace-scoped runbook with `uv run ops runbook init`, or copy the preset content into `<workspace-root>/outputs/logs/ops/runbooks/<runbook-id>.yaml` and rewrite the workspace-relative paths before use. Store run-specific variants under `<workspace-root>/outputs/logs/ops/runbooks/` with a stable runbook id.

### 2-minute dry-run path

Run only gate checks, no qsub submit phase:

```bash
uv run ops runbook execute \
  --runbook <path-to-runbook.yaml> \
  --repo-root <repo-root> \
  --audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json
```

Only workspace-scoped audit paths are accepted; `--audit-json` must stay under `<workspace-root>/outputs/logs/ops/audit/`.
On workstations without `qstat`, add `--allow-missing-qstat` to keep the queue probe explicit but non-fatal for dry-run/demo use. The resulting audit remains readable, and `ops progress show ops.control-plane.orchestration` will report attention when queue readiness is degraded.

Result expectations:

1. `execution.ok=true` means all preflight and notify smoke commands passed.
2. `execution.failed_phase` identifies the first failed phase when non-zero.
3. `submit` commands are not executed unless `--submit` is provided.
4. Command timeout defaults to 300 seconds per command; override with `--command-timeout-seconds`.
5. Preflight includes template QA, submit-shape advisor output, and operator brief output before notify smoke.
6. Supported diagnostics live under `ops runbook diagnostics ...`; `session-counts`, `submit-shape-advisor`, and `operator-brief` accept `--qstat-file <fixture>` for deterministic pressure tests and use the same machine-readable `advisor`, `submit_gate`, and `queue_policy` fields as the repo-local `sge-hpc-ops` overlay.
7. DenseGen preflight probes GUROBI with explicit runtime env injection (`GUROBI_HOME`, `GRB_LICENSE_FILE`, `TOKENSERVER`, `LD_LIBRARY_PATH`).
8. DenseGen preflight runs an overlay-sprawl guard that projects overlay part growth from config + mode/run-args and can auto-compact existing overlay parts before submit.
9. DenseGen preflight runs a records-part guard that projects `records__part-*.parquet` growth and applies age/count maintenance before submit.
10. DenseGen preflight runs an archived-overlay retention guard that enforces `_derived/_archived` count/size thresholds before submit.

### Fill remaining Infer lanes

Use the fill command when a checked-in study has multiple Infer runbooks and the
operator question is "run whatever sequence-view inference is still missing":

```bash
uv run ops runbook fill-infer --study-dir docs/studies/<study-id>
uv run ops runbook fill-infer --study-dir docs/studies/<study-id> --execute
uv run ops runbook fill-infer --study-dir docs/studies/<study-id> --submit
```

The command is a control-plane wrapper around ordinary Infer runbooks. It reads
study `execution_surfaces` or repeated `--runbook <path>` inputs, runs the
Infer sequence-view completion inventory for each lane, skips complete lanes,
blocks lanes with missing sequence products or missing durability metadata, and
plans or executes only lanes with missing vectors/scalars or stale sidecars that
must be repaired under the current runtime fingerprint. It does not merge
runbooks or event streams: Notify remains one watcher per lane, and each
executed lane writes its audit JSON under
`<workspace-root>/outputs/logs/ops/audit/<runbook-id>.fill-infer.json`.

For sequence-view feature backfills, the completion inventory also emits a
`shard_plan` with shard count, pending vector/scalar key counts, runtime
fingerprint key, checkpoint ledger path, and the
`temp_validate_promote`/`skip_committed_retry_failed` durability policy.
Broad multi-shard lanes require that durability policy before SGE plan
rendering. The Infer runner commits sidecar payloads at shard boundaries,
validates persisted vector/scalar keys, updates the checkpoint ledger atomically,
and emits `infer_feature_bundle_shard_commit` events before final bundle
completion. A rerun treats current-fingerprint payloads as reusable and
recomputes missing or stale raw-case/fingerprintless sidecars.

For workstation planning without a scheduler, add
`--no-discover-active-jobs --allow-unknown-active-jobs --allow-missing-qstat`.
Do not combine `--allow-missing-qstat` with `--submit`; real submission still
requires a submit-capable HPC shell, scheduler visibility, and Notify webhook
materialization.

### Orchestration workflow ids

Workflow ids are transport-neutral. Notify transport and delivery wiring live in `runbook.notify`, not in `workflow_id`.

1. `densegen_batch_submit`: CPU-focused DenseGen submit without notify smoke/watcher phases.
2. `densegen_batch_with_notify`: CPU-focused DenseGen submit with notify watcher chain.
3. `infer_batch_submit`: GPU-focused Infer submit without notify smoke/watcher phases.
4. `infer_batch_with_notify`: GPU-focused Infer submit with notify watcher chain.
5. DenseGen routes reject GPU resource keys to prevent mismatched scheduler requests.
6. Infer routes require `resources.gpus` and `resources.gpu_capability` to keep GPU requests explicit.
7. Infer routes accept optional `resources.gpu_memory_gib` for explicit per-GPU capacity planning.

### Runbook schema (v1)

```yaml
runbook:
  schema_version: 1
  id: study_stress_ethanol_cipro
  workflow_id: densegen_batch_submit
  project: <project>
  workspace_root: src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro
  logging:
    stdout_dir: src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/outputs/logs/ops/sge/study_stress_ethanol_cipro
    retention:
      keep_last: 20
      max_age_days: 14
  densegen:
    config: src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml
    qsub_template: docs/bu-scc/jobs/densegen-cpu.qsub
    post_run:
      qsub_template: docs/bu-scc/jobs/densegen-analysis.qsub
      resources:
        pe_omp: 1
        h_rt: 00:20:00
        mem_per_core: 2G
    run_args:
      fresh: --fresh --no-plot
      resume: --resume --no-plot
    overlay_guard:
      max_projected_overlay_parts: 3000
      max_existing_overlay_parts: 1000
      auto_compact_existing_overlay_parts: true
      overlay_namespace: densegen
    records_part_guard:
      max_projected_records_parts: 3000
      max_existing_records_parts: 1000
      max_existing_records_part_age_days: 14
      auto_compact_existing_records_parts: true
    archived_overlay_guard:
      max_archived_entries: 1000
      max_archived_bytes: 2147483648
  resources:
    pe_omp: 12
    h_rt: 08:00:00
    mem_per_core: 8G
  mode_policy:
    default: auto
    on_active_job: hold_jid
```

Path behavior:

1. Relative paths are resolved from the runbook file parent directory.
2. Absolute paths remain unchanged.
3. `runbook.notify` is required only for `*_with_notify` workflows.
4. `workflow_id` must be one of the transport-neutral ids listed above; deprecated transport-specific ids are rejected.
5. `runbook.logging.stdout_dir` is required and is used for all scheduler stdout (`qsub -o`) in verify and submit phases.
6. Notify-enabled runbooks accept `notify.orchestration_events` (default `true`) to control direct orchestration lifecycle notifications.
7. DenseGen runbooks accept `densegen.post_run.qsub_template` for the dependent analysis submit that runs plots after CPU generation completes.
8. DenseGen runbooks can set `densegen.post_run.resources` to size analysis submits (`pe_omp`, `h_rt`, `mem_per_core`); when omitted, the default post-run sizing is `pe_omp=4`, `h_rt=01:00:00`, `mem_per_core=4G`.

### Single-study accumulation contract

For repeated attempts on one study workspace, keep paths stable to avoid fan-out:

1. Keep one stable `workspace_root` and `densegen.config` (`<workspace-root>/config.yaml`).
2. Keep one stable runbook id and one stable runbook file path per study at `<workspace-root>/outputs/logs/ops/runbooks/<runbook-id>.yaml`.
3. Keep one stable `audit-json` path and overwrite it each run (for example: `<workspace-root>/outputs/logs/ops/audit/latest.json`).
4. Use `--mode auto` for normal loops so ops resolves `fresh` vs `resume` deterministically.
5. Keep `logging.retention.keep_last` and `logging.retention.max_age_days` explicit in the runbook so stale logs are pruned before each submit attempt.
6. Use `uv run dense campaign-reset -c <workspace-root>/config.yaml --yes` before an intentional full fresh campaign reset; this clears run artifacts while preserving notify/log scaffolding and preserves USR registry unless `--purge-usr-registry` is set.
7. After reset, run `ops ... --mode fresh --allow-fresh-reset` once, then return to `--mode auto`.

### Planner and executor commands

Generate deterministic command specs:

```bash
uv run ops runbook plan --runbook <path-to-runbook.yaml> --repo-root <repo-root>
```

Use `--repo-root` on `plan`, `active-jobs`, and `execute` when invoking from outside the repository so runbook path contracts are checked against the canonical repository root.

Optional plan overrides:

```bash
uv run ops runbook plan \
  --runbook <path-to-runbook.yaml> \
  --repo-root <repo-root> \
  --mode auto \
  --notify-smoke live \
  --active-job-id 81234 \
  --active-job-id 81235

# Equivalent manual input form:
uv run ops runbook plan \
  --runbook <path-to-runbook.yaml> \
  --repo-root <repo-root> \
  --mode auto \
  --active-job-id 81234,81235

# Destructive fresh reset when artifacts already exist requires explicit acknowledgement.
uv run ops runbook plan \
  --runbook <path-to-runbook.yaml> \
  --repo-root <repo-root> \
  --mode fresh \
  --allow-fresh-reset
```

Discover matching active jobs before planning or execution:

```bash
uv run ops runbook active-jobs --runbook <path-to-runbook.yaml> --repo-root <repo-root>
```

`active-jobs` emits `active_job_ids` plus ready-to-paste chaining hints: `active_job_ids_csv`, `active_job_id_args`, and `plan_command_hint`.
OPS-submitted jobs also carry explicit scheduler identity tags so automatic matching does not rely on workspace-path token scraping alone. The visible job name remains operator-friendly, while the machine-readable identity carries the stable run-group and workspace ids used by active-job discovery and audit correlation.
The payload also includes `runtime_visibility` so `no_match`, `multiple_matches`, and degraded scheduler visibility stay explicit in both JSON and CLI-adjacent tooling.

Supported scheduler diagnostics:

```bash
uv run ops runbook diagnostics session-counts --qstat-file <fixture>
uv run ops runbook diagnostics submit-shape-advisor --qstat-file <fixture> --planned-submits <N> --warn-over-running 3
uv run ops runbook diagnostics operator-brief --qstat-file <fixture> --planned-submits <N> --warn-over-running 3
```

Run gate checks and optionally submit:

Use a workspace-scoped audit output path for every execute invocation:
`<workspace-root>/outputs/logs/ops/audit/<file>.json`

```bash
uv run ops runbook execute \
  --runbook <path-to-runbook.yaml> \
  --repo-root <repo-root> \
  --audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json \
  --no-submit
```

```bash
uv run ops runbook execute \
  --runbook <path-to-runbook.yaml> \
  --repo-root <repo-root> \
  --audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json \
  --command-timeout-seconds 600 \
  --submit

# Recommended for deterministic HTTPS notify delivery across hosts.
export SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem

# Destructive fresh reset when artifacts already exist requires explicit acknowledgement.
uv run ops runbook execute \
  --runbook <path-to-runbook.yaml> \
  --repo-root <repo-root> \
  --audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json \
  --mode fresh \
  --allow-fresh-reset \
  --submit
```

### Contract rules

1. `notify.smoke` defaults to `dry` when `runbook.notify` is present; `live` is an explicit override.
2. DenseGen `mode=auto` uses explicit workspace-state classification: `none -> fresh`, `resume_ready -> resume`, `partial -> contract error`; `resume_ready` includes run manifest, non-empty DenseGen attempts artifacts (`attempts.parquet` or `attempts_part-*.parquet`), or non-empty DenseGen-shaped records (`records.parquet` or `records__part-*.parquet`) validated against Arrow logical field names.
3. DenseGen `mode=auto` treats registry-only reset state (`outputs/usr_datasets/registry.yaml` with no run manifest and no non-empty records) as `none`, so dry-run scaffolds can proceed without an explicit mode override.
4. Active-job behavior is explicit from `mode_policy.on_active_job` (`hold_jid` or `stop`), never implicit.
5. `ops runbook plan` and `ops runbook execute` auto-discover matching active jobs by default from explicit OPS scheduler identity tags carried in job metadata; disable with `--no-discover-active-jobs`.
6. The scheduler-visible job name is for operators; automatic matching uses the machine-readable run-group and workspace identity tags rather than name or path heuristics alone.
7. `ops runbook plan` and `ops runbook active-jobs` expose `runtime_visibility` with explicit `scheduler_probe_state` and `active_job_resolution_state` values.
8. `ops runbook plan` may still return a useful dry-run plan when runtime visibility is degraded, but it records that degraded posture explicitly, emits an operator-visible warning, and does not silently collapse `unknown` into `no_match`.
9. `ops runbook execute --submit` fails closed by default when `runtime_visibility.active_job_resolution_state=unknown`; use `--allow-unknown-active-jobs` only when degraded submit is intentionally accepted and audit JSON should record that override.
10. Manual active-job input (`--active-job-id`) accepts repeat flags or comma-delimited values and normalizes to a deduplicated job-id set before `-hold_jid` chaining.
11. Execution is fail-fast by phase: `preflight`, optional `notify_smoke`, then optional `submit`.
12. `--command-timeout-seconds` applies per command and fails phase on timeout; default is `300`.
13. DenseGen `mode=fresh` is blocked when resume artifacts already exist unless `--allow-fresh-reset` is explicitly provided.
14. DenseGen routes reject `runbook.infer` and reject GPU resource fields.
15. Infer routes reject `runbook.densegen` and require GPU resource fields.
16. Infer planning validates runbook GPU resources against infer `model.parallelism` and capacity contracts before preflight command rendering.
17. Infer capacity preflight uses `resources.gpu_memory_gib` when provided; otherwise it uses capability hints for known classes (`gpu_capability=8.9 -> 45.0 GiB`, `gpu_capability=9.0 -> 80.0 GiB`).
18. Infer `validate config` on non-GPU hosts validates schema/contracts and reports capacity-check skip; runbook planning remains the deterministic place for declared scheduler-resource capacity checks.
19. Notify workflows require `runbook.notify`, with policy matching workflow family (`densegen|generic` for DenseGen, `infer|generic` for Infer).
20. Notify setup is non-interactive and file-only by contract: planner resolves webhook file with explicit precedence (`<notify.webhook_env>_FILE` first, otherwise `notify.profile` webhook `secret_ref` when it is a `file://` path) and uses `--secret-source file --secret-ref file://<resolved-path> --no-store-webhook`; missing/unreadable webhook file fails fast before submit.
21. Notify setup writes an explicit TLS CA bundle in profile wiring (`--tls-ca-bundle`, defaulting to `SSL_CERT_FILE` or `/etc/pki/tls/certs/ca-bundle.crt`).
22. Notify smoke bootstraps the resolved USR events file path before dry-watch so fresh workspaces do not fail on missing `.events.log`.
23. Batch-only workflows reject `runbook.notify` to keep notify decoupled.
24. Preflight creates and verifies `runbook.logging.stdout_dir` (`mkdir -p` + writable check) before any verify or submit command.
25. Verify and submit commands always pass explicit `-o <runbook.logging.stdout_dir>/$JOB_NAME.$JOB_ID.out` to avoid cwd-dependent stdout placement.
26. Preflight runs `prune-ops-logs`-backed log-retention guards before scheduler verify/submit for both `runbook.logging.stdout_dir` and `<workspace-root>/outputs/logs/ops/runtime`, with `--stdout-dir`, `--runbook-id`, `--keep-last`, and `--max-age-days` from runbook logging retention settings.
27. `runbook.logging.stdout_dir` must be exactly `<workspace-root>/outputs/logs/ops/sge/<runbook.id>`; mismatched id variants are rejected at runbook load time to prevent path fan-out.
28. DenseGen notify workflows run `dense inspect run --usr-events-path` in preflight so local-only output configs fail before submit.
29. Notify watcher submit injects `NOTIFY_PROFILE`, `WEBHOOK_ENV`, and `WEBHOOK_FILE` in `qsub -v`; webhook values are not embedded in scheduler metadata.
30. When `notify.orchestration_events=true`, execute emits direct notify lifecycle events for submit orchestration (`started`, then `success` or `failure`) in addition to watcher event notifications, using file-backed `--secret-ref`.
31. Notify DenseGen progress semantics use workspace-session counters (`rows_written_session`, `run_quota`, and `fingerprint.rows`) and default heartbeat cadence is 1800 seconds (`progress_heartbeat_seconds`) unless explicitly overridden.
32. Notify workflows resolve a TLS CA bundle from `SSL_CERT_FILE` or known system CA paths and pass it explicitly to notify setup/orchestration commands; when no readable CA bundle is resolvable, planning fails fast before submit.
33. DenseGen preflight runs `usr-overlay-guard` with explicit thresholds from its overlay-guard block; when projected overlay parts exceed `max_projected_overlay_parts`, planning fails fast with required tuning guidance.
34. Sequence-view Infer preflight does not render `usr-overlay-guard`: those jobs write sidecar features under `_derived/infer/`, not row-overlay parts. It renders `infer validate sequence-view-completion` instead, blocks missing source products, and requires shard-level durability metadata before SGE submit.
35. `densegen.overlay_guard.overlay_namespace` must match `^[a-z][a-z0-9_]*$` so runbook contracts remain compatible with USR overlay namespace rules.
36. Legacy row-writeback Infer runbooks remain explicit direct-runbook surfaces. The study-level `fill-infer` primitive skips non-sequence-view Infer runbooks before SGE plan rendering, so old USR/write-back shapes do not enter the intelligent fill queue.
37. `usr-overlay-guard` and `usr-records-part-guard` are explicit about degraded mode: tools that do not emit overlay parts or records-part files return `guard_status=skipped` with a reason (no silent fallback).
38. DenseGen preflight runs `usr-records-part-guard` with explicit thresholds from `densegen.records_part_guard`; when projected records parts exceed `max_projected_records_parts`, planning fails fast with tuning guidance (`runtime.max_accepted_per_library`, `output.parquet.chunk_size`).
39. When existing records parts exceed `densegen.records_part_guard.max_existing_records_parts` or oldest part age exceeds `max_existing_records_part_age_days`, preflight compacts `records__part-*.parquet` into `records.parquet` when `auto_compact_existing_records_parts=true`; otherwise planning fails fast with explicit maintenance guidance.
40. DenseGen preflight runs `usr-archived-overlay-guard` with explicit thresholds from `densegen.archived_overlay_guard`; when `_derived/_archived` file count or byte total exceeds thresholds, planning fails fast with explicit retention guidance.
41. `ops runbook execute --audit-json` must be exactly `<workspace-root>/outputs/logs/ops/audit/<file>.json`; non-workspace audit paths fail fast.
42. transient operational working directories at repo root (for example `.codex_tmp/`, `.tmp_ops/`, `tmp_ops/`) are not allowed; place disposable operational working state under `/scratch` and keep durable orchestration artifacts under `<workspace-root>/outputs/logs/ops/`.
43. DenseGen run args remain `--no-plot` by default for generation throughput; submit phase adds a dependent `densegen-analysis.qsub` job (`-hold_jid <densegen_cpu_job_name>`) that runs `dense plot --only "$DENSEGEN_ANALYSIS_PLOTS"` with a static default set (`stage_a_summary,placement_map,run_health,tfbs_usage`).
44. DenseGen post-run submit always uses `densegen.post_run.resources`; default post-run resources are `pe_omp=4`, `h_rt=01:00:00`, `mem_per_core=4G`.

Infer resume note:

- Sequence-view Infer jobs resume from sidecar aliases/vectors/scalars under `_derived/infer/`; Ops does not pass `INFER_RUN_ARGS=--overwrite` for these jobs.
- Legacy row-writeback Infer jobs may still use `INFER_RUN_ARGS=--overwrite` when an operator directly plans that runbook and intentionally wants to recompute requested write-back outputs in place without implicitly pruning the namespace.
- If merge carry preserved stale legacy `infer` values that should be removed rather than recomputed in place, prune the overlay first; use `--mode fresh --allow-fresh-reset` only when recomputing the current requested outputs is intentional.

### Related docs

1. [Ops orchestration index](README.md)
2. [Ops package README](../../../src/dnadesign/ops/README.md)
3. [OPS runtime visibility](../status/runtime-visibility.md)
4. [BU SCC jobs docs](../../bu-scc/jobs/README.md)
