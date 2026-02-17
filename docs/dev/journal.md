## Development journal

This journal records cross-package engineering decisions, investigations, and validation notes for maintainers. Read it when you need historical rationale behind implementation changes.

### Contents
This section lists dated entries so you can jump to a specific investigation window.

- [2026-02-04](#2026-02-04)
- [2026-02-05](#2026-02-05)
- [2026-02-11](#2026-02-11)
- [2026-02-12](#2026-02-12)
- [2026-02-13](#2026-02-13)
- [2026-02-15](#2026-02-15)
- [2026-02-18](#2026-02-18)

### 2026-02-04
- Investigated the reported stall in `test_round_robin_chunk_cap.py::test_stall_detected_with_no_solutions`.
- Root cause: the test intentionally sleeps ~1.1s in the `_EmptyAdapter` generator to simulate a no-solution stall. `pytest --durations=10` shows this test as ~1.12s, matching the sleep.
- No infinite hang observed after rerunning `uv run pytest -q src/dnadesign/densegen/tests --durations=10`.

### 2026-02-05
- Added a tool-agnostic notifier CLI (`notify`) with generic/Slack/Discord payloads and explicit URL validation.
- Documented notifier usage in `docs/notify/README.md` and registered the CLI entry in `pyproject.toml`.
- Expanded notifier docs with Slurm and BU SCC examples plus local testing guidance.

### 2026-02-11
- Prepared implementation plan for Notify ergonomics/UX across human operators and agentic automation.
- Scope in: watcher lifecycle controls, multi-workflow policies (DenseGen + infer/Evo2), richer failure reporting, machine-friendly CLI outputs, secret backend robustness, spool/cursor correctness.
- Scope out: scheduler-specific orchestration frameworks beyond current qsub templates.

- Plan slice 1 (reliability + lifecycle, highest risk first):
  - Add terminal DenseGen failure event emission at orchestration boundary so runs always emit a terminal outcome signal.
  - Add watcher lifecycle controls: `--wait-for-events`, `--idle-timeout`, `--stop-on-terminal-status`.
  - Update qsub watcher defaults to avoid idle slot burn after terminal state.
  - Add tests for terminal failure signal and controlled watcher shutdown.

- Plan slice 2 (workflow policy model, decoupled and extensible):
  - Replace DenseGen-only implicit defaults with explicit workflow policies (for example `densegen`, `infer_evo2`, `generic`).
  - Add policy-driven defaults for filters/status mapping/noise budget while keeping CLI overrides explicit.
  - Extend wizard and docs to select policy first, then provider and secret wiring.
  - Add regression tests proving infer events are not silently filtered by DenseGen defaults.

- Plan slice 3 (error expressivity + human/agent output surfaces):
  - Preserve concise Slack text for routine progress, but include compact actionable failure detail.
  - Add structured machine-readable CLI outputs (`--json`) for profile wizard/doctor and watcher setup status.
  - Ensure failure events can surface key error context without requiring full raw event payloads.
  - Add tests for both human-readable and machine-readable output contracts.

- Plan slice 4 (operational robustness):
  - Harden secret backend probing beyond command existence so `auto` does not produce false positives in batch contexts.
  - Fix spool drain to honor per-spool-file provider metadata unless explicitly overridden.
  - Add cursor locking to prevent duplicate concurrent watchers on the same cursor path.
  - Revisit dry-run cursor semantics and make behavior explicit.

- Decision gates before code changes:
  - Dry-run cursor behavior: keep current cursor advancement vs switch default to non-advancing dry-run.
  - Backward compatibility scope for preset names/semantics when introducing policy model.
  - Whether watcher stop-on-terminal should be opt-in or default for batch templates.

- Validation strategy:
  - TDD for each slice with targeted tests first, then implementation.
  - Run `uv run pytest src/dnadesign/notify/tests -q` and relevant DenseGen runtime tests per slice.
  - Keep each slice independently shippable with docs updated in the same PR.

- Slice 2 implemented (policy model):
  - Added explicit profile policy support with defaults for `densegen`, `infer_evo2`, and `generic`.
  - Preserved `--preset densegen` compatibility and mapped it onto policy behavior.
  - Added watcher fallback so profile `policy` applies defaults when explicit `only_actions`/`only_tools` are absent.
  - Updated BU SCC watcher template to use `NOTIFY_POLICY` with optional `NOTIFY_ACTIONS`/`NOTIFY_TOOLS` overrides.
  - Set qsub env-mode default policy to `generic` to avoid silently missing non-DenseGen workflows.
  - Added targeted regression tests for profile policy wiring and qsub infer policy filtering.

- Slice 3 implemented (human + agent output surfaces):
  - Added `--json` output mode to `notify profile wizard` and `notify profile doctor` for automation-friendly parsing.
  - Added structured failure JSON (`ok=false`, `error`) for wizard/doctor error paths.
  - Improved failure message expressivity for `densegen_health` and `densegen_flush_failed` by including event error text when present.
  - Added targeted tests covering JSON output contracts and failure message content.

- Slice 4 implemented (operational robustness):
  - Hardened secret backend availability checks to include runtime probing (not command existence only).
  - For `secretservice`, availability now requires a DBus session plus a successful probe command return contract.
  - Updated spool drain to honor per-file stored provider metadata by default, with `--provider` as an explicit override.
  - Added cursor lock files for watcher runs to prevent concurrent watchers from sharing the same cursor path.
  - Added stale lock recovery when lock holder PID is no longer running.
  - Added explicit dry-run cursor control via `--advance-cursor-on-dry-run/--no-advance-cursor-on-dry-run`.
  - Added targeted tests for secret probing, cursor locking, dry-run cursor semantics, and spool provider selection.

- Cleanup pass:
  - Updated remaining wizard docs to use policy-based defaults (`--policy densegen`) instead of preset examples.
  - Added explicit operator guidance for manual Slack webhook entry via hidden wizard prompt (no `--webhook-url` in command line).
  - Confirmed this node has DBus session and `secret-tool`, so secure prompt-based wizard flow is viable in the current session.

- Observer-only setup slice implemented (tool-agnostic control-plane direction):
  - Added `notify setup slack` command to reduce operator cognitive load and avoid manual events-path wiring.
  - Added resolver mode (`--tool densegen --config <densegen-config.yaml>`) that computes the expected USR `.events.log` path before run start.
  - Added pre-run-safe profile creation for resolver mode (events path can be absent at setup time).
  - Added profile `events_source` metadata (`tool`, `config`) so watcher runs can re-resolve events paths and avoid stale-path drift after config changes.
  - Kept non-interference invariant: setup remains observer-only and does not launch or mutate tool execution behavior.
  - Updated notify onboarding docs to lead with `notify setup slack` and clearly distinguish DenseGen `config.yaml` from USR events files.
  - Added targeted tests for setup resolver behavior, missing/unsupported tool validation, and watch-time re-resolution from profile metadata.

- Pragmatic hardening pass:
  - Removed implicit env-var fallback for env secret mode; `--url-env` is now required when `--secret-source env` is selected.
  - Added watcher poll cadence control (`--poll-interval-seconds`) to reduce idle wake overhead on shared batch nodes while preserving existing semantics.
  - Split tool events-path resolution into `notify/events_source.py` so tool-specific logic is decoupled from CLI orchestration and easier to extend.
  - Added targeted tests for strict env-mode validation and poll-interval validation.

- Decoupling pass (option 1 + option 2):
  - Added first-class `infer_evo2` events-source resolver in `notify/events_source.py` with strict ambiguity checks.
  - Resolver contract now fails fast when infer config has no USR write-back jobs or multiple USR destinations.
  - Added `notify/tests/test_events_source.py` covering infer resolver success, env-root behavior, ambiguity, and missing-source errors.
  - Split notify operational helpers into dedicated modules:
    - `notify/profile_ops.py` (profile naming/state root/next-step guidance)
    - `notify/watch_ops.py` (cursor locking and follow-loop primitives)
    - `notify/spool_ops.py` (secure spool directory and payload persistence)
  - Rewired `notify/cli.py` to consume those modules while preserving command behavior and existing CLI contract.

- Second-pass assertive decoupling and watcher ergonomics:
  - Removed infer resolver dependency on infer-internal default root fallback; resolver now requires explicit `ingest.root` or `DNADESIGN_USR_ROOT` for USR jobs.
  - Added `notify setup resolve-events` for tool-agnostic path/policy resolution without profile writes (`plain`, `--print-policy`, `--json`).
  - Refactored BU SCC watcher template to resolve events from `NOTIFY_TOOL` + `NOTIFY_CONFIG` instead of DenseGen-only config wiring.
  - Added policy auto-resolution in watcher env mode when policy is not explicitly set, reducing manual filter wiring drift.
  - Removed watcher precondition that `.events.log` must already exist; watcher now supports pre-run startup with `--wait-for-events`.
  - Updated `notify profile doctor` to treat resolver-backed missing events files as pending instead of wiring failures.
  - Updated notify/HPC docs to use tool-agnostic watcher env variables and documented `setup resolve-events`.
  - Added targeted regression tests for strict infer root contract, setup resolver outputs, tool-config watcher resolution, and pre-run missing-events behavior.

- Third-pass strictness/ergonomics hardening (no-fallback direction):
  - Removed legacy `preset` behavior from notify profile creation paths (`profile init`, `profile wizard`) and command registration.
  - Added explicit guard in profile validation: legacy `preset` field now fails fast with actionable migration error (`use policy + explicit filters`).
  - Tightened profile contract: non-generic `policy` now requires explicit `only_actions` and `only_tools` in profile data.
  - Removed watch-time policy fallback injection so event filters come from explicit profile fields or explicit CLI flags.
  - Improved env-mode UX without tool coupling: for `--secret-source env`, default webhook env var now resolves to `NOTIFY_WEBHOOK` when `--url-env` is omitted.
  - Updated BU SCC watcher template default env var from `DENSEGEN_WEBHOOK` to `NOTIFY_WEBHOOK` and added `NOTIFY_POLL_INTERVAL_SECONDS` (default `1.0`) to reduce idle polling overhead.
  - Applied docs alignment across notify/HPC runbooks and DenseGen HPC workflow pages for `NOTIFY_WEBHOOK` and poll tuning.
  - Added/updated targeted tests for: env default var behavior, preset rejection, explicit-policy filter requirement, and qsub poll interval propagation.

- No-backcompat profile schema pass (v2-only):
  - Removed legacy profile v1 acceptance from notify profile parsing and webhook-source resolution.
  - Enforced strict profile version gate: `profile_version` must equal 2.
  - Migrated `notify profile init` writer to emit v2 schema with `webhook={source: env, ref: <ENV_VAR>}`.
  - Kept CLI `--url-env` input for init/setup ergonomics while removing v1 storage semantics.
  - Updated tests from v1 fixtures to v2 fixtures and added explicit rejection test for legacy v1 profiles.
  - Updated operator docs to state profile schema is v2-only and removed v1 wording.

- Docs alignment pass (workspace-scoped Notify artifacts):
  - Standardized canonical docs/examples to run-local Notify artifact paths under each tool workspace:
    - `outputs/notify/profile.json`
    - `outputs/notify/cursor`
    - `outputs/notify/spool`
  - Removed `/projectnb/.../notify/...` examples for watcher profile paths in Notify + DenseGen + HPC runbooks.
  - Added explicit “artifact placement strategy” guidance in `docs/notify/usr-events.md`:
    - default run-local co-location with producing run workspace
    - optional centralized Notify workspaces under `/project/...` with tool/run-qualified watch IDs
    - explicit note that Notify stays decoupled from sink choices (parquet/USR), and always watches USR `.events.log`.
  - Updated workflow demos (`demo_usr_notify`, `demo_pwm_artifacts`) to use `notify setup slack` with workspace config resolution and run-local cursor/spool paths.

### 2026-02-12
- Continued pragmatic decoupling/extraction pass across Notify and USR with registry-oriented boundaries:
  - Notify: extracted command/runtime helpers and registry wiring to reduce tool coupling in command-line orchestration.
  - USR: extracted command-line and dataset hotspots into smaller operation modules to reduce monolith pressure and improve change isolation.
- Performed contiguous semantic-history cleanup on the active feature branch (no cross-branch rewriting), consolidating tightly related extraction commits to keep history reviewable.
- Completed documentation alignment pass for Notify + USR after refactors so command examples and architecture narrative reflect current behavior.
- Completed a didactic clarity pass to expand dense acronym-heavy prose in operator guides while preserving executable command syntax and environment variable names.
- Current branch state after the above: `dev/densegen-hpc-patching` remains the active working branch for this stream of changes.

### 2026-02-13
- Investigated rejoin behavior for BU SCC interactive sessions after client-side interruptions from the agent harness.
- Prior state and evidence:
  - `qstat -u esouth` was empty at check time, so there was no active interactive job to rejoin.
  - `qacct -j 3125614` showed:
    - `qsub_time`: `Fri Feb 13 14:07:21 2026`
    - `start_time`: `Fri Feb 13 14:10:42 2026`
    - `end_time`: `Fri Feb 13 14:10:44 2026`
    - `failed=100` and `exit_status=129`
  - Interpretation of this record is consistent with abrupt client/session termination rather than a clean shell exit.
- Reattach mechanism experiment (`qrsh -inherit`) from login context:
  - `qrsh -help` confirms `-inherit` is supported.
  - Attempt with explicit job id argument failed because required environment values were absent from login shell.
  - Follow-up attempts with `JOB_ID` and `SGE_TASK_ID` set still failed to provide a reliable terminal reattach flow.
  - Conclusion: `qrsh -inherit` is useful for job-context command execution but was not a dependable "reattach old interactive terminal" mechanism in this agent/login context.
- Capability snapshot captured from probes on `scc1`:
  - `scheduler_tools`: `qsub=yes`, `qstat=yes`, `qdel=yes`, `qmod=yes`, `qacct=yes`
  - `interactive_cmd`: `qrsh` (and `qlogin` available)
  - `accounting_flag`: both `-P` and `-A` visible; BU guidance remains `-P <project>`
  - `pe_known`: yes (`omp*`, `mpi*`)
  - `walltime_key`: `h_rt`
  - `memory_key`: `mem_per_core` (also `h_vmem`, `mem_free`)
  - `gpu_keys`: includes `gpus` plus capability complexes (`gpu_c`, `gpu_m`, `gpu_t`, `gpu_dp`)
  - `transfer_key`: `download`
  - `lifecycle`: `qdel=yes`, `qmod -cj=yes`, `qacct=yes`
  - `unknowns`: `ssh_to_compute_allowed` (policy not confirmed in this session)
- Safe OnDemand interactive-session experiment (no compute job submissions):
  - Goal: validate whether an OnDemand-backed reconnectable control plane is reachable from this environment and whether it can be started non-interactively.
  - Baseline checks:
    - `qstat -u esouth | grep -E 'ood-|ood_|ondemand|jupyter|rstudio|desktop'` returned no active OnDemand-pattern jobs.
    - No local `ood` or `ondemand` CLI entrypoint was present; no `ondemand` module was available via `module avail`.
  - Web-path probes:
    - `curl -I https://scc-ondemand.bu.edu/` returned `301` to `https://scc-ondemand1.bu.edu:443/`.
    - `curl -I https://scc-ondemand1.bu.edu/pun/sys/dashboard` returned `302` redirect to BU Shibboleth SAML login flow (`https://shib.bu.edu/...`).
    - `curl -L` with cookie jar reached BU login HTML ("Boston University | Login"), confirming portal reachability and web-auth requirement.
  - Post-check:
    - `qstat -u esouth` remained empty; no accidental interactive job/session was launched by probe traffic.
- Consolidated interpretation:
  - In this harness, interrupting the agent can terminate the attached `qrsh` client process.
  - Plain `qrsh` sessions should be treated as non-rejoinable after client loss from this control path.
  - OnDemand is the safer reconnectable interactive UX for humans because session state is mediated by web auth/session management rather than a single CLI client process.
  - Non-interactive probes can verify OnDemand endpoint reachability and auth redirects, but creating/continuing a real OnDemand interactive session requires authenticated browser steps (including Duo).
- Operational guidance recorded for future runs:
  - Use `qsub` batch for long or interruption-sensitive workflows.
  - Use OnDemand for reconnectable human-interactive workflows.
  - If an interactive job is orphaned and unusable, explicitly clean with `qdel <job_id>` before requesting a replacement allocation.

### 2026-02-15
- Dependency contract update:
  - Kept `marimo` as a core project dependency.
  - Added `openai` as a core project dependency so notebook/API experiments do not require a separate dependency group.
  - Regenerated `uv.lock` after dependency updates.
- DenseGen output naming unification:
  - Updated DenseGen local canonical parquet artifact naming from `outputs/tables/dense_arrays.parquet` to `outputs/tables/records.parquet`.
  - Kept USR dataset canonical artifact naming as `.../records.parquet`; both sinks now use the same filename contract.
  - Updated DenseGen code paths, workspace templates, tests, and DenseGen docs to reflect the unified filename.
- Validation run (targeted):
  - `uv run pytest -q src/dnadesign/densegen/tests/runtime/test_run_paths.py src/dnadesign/densegen/tests/runtime/test_run_metrics.py src/dnadesign/densegen/tests/runtime/test_run_diagnostics_plots.py src/dnadesign/densegen/tests/cli/test_cli_run_modes.py src/dnadesign/densegen/tests/cli/test_cli_workspace_init.py src/dnadesign/densegen/tests/reporting/test_reporting_library_summary_outputs.py src/dnadesign/densegen/tests/plotting/test_plot_manifest.py`
- Dense notebook + BaseRender planning notes:
  - Current BaseRender input contract is intentionally minimal (`sequence`, `densegen__used_tfbs_detail`, optional `id`).
  - DenseGen currently does not expose a first-class workspace-scoped notebook command.
  - Planned direction is a workspace-first Dense notebook flow with:
    - run/workspace metadata header,
    - summary stats block from run artifacts,
    - plot gallery navigation,
    - explicit use of `records.parquet` from local or USR sink.

### 2026-02-18
- Documentation system-of-record scaffold introduced at repo root:
  - `ARCHITECTURE.md`, `DESIGN.md`, `SECURITY.md`, `RELIABILITY.md`, `PLANS.md`, `QUALITY_SCORE.md`.
- `AGENTS.md` was reduced to a map-style entrypoint and now links to canonical policy/runbook docs instead of carrying full prose copies.
- Added docs knowledge-base indexes:
  - `docs/architecture/README.md`
  - `docs/architecture/decisions/README.md`
  - `docs/security/README.md`
  - `docs/reliability/README.md`
  - `docs/quality/README.md`
  - `docs/exec-plans/README.md`
  - `docs/templates/README.md`
- Added reusable templates:
  - `docs/templates/system-of-record.md`
  - `docs/templates/runbook.md`
  - `docs/templates/adr.md`
  - `docs/templates/exec-plan.md`
- Added explicit ADR policy: numbered ADRs are required for new decisions going forward; historical backfill is optional.
- Extended docs check contract to validate relative links from root system-of-record docs, not only `docs/` plus root `README.md`.
- SOR hardening pass implemented with CI/devtool enforcement:
  - Root SOR docs now include required metadata (`Type`, `Owner`, `Last verified`).
  - `dnadesign.devtools.docs_checks` now enforces:
    - root SOR metadata presence/date freshness (`--max-sor-age-days`)
    - execution-plan metadata and traceability link requirements for non-README files under `docs/exec-plans/`.
  - Added `dnadesign.devtools.architecture_boundaries` and wired it into the core CI lane to fail on undeclared cross-tool imports.
  - Added `dnadesign.devtools.quality_score` to generate CI-backed quality score inputs from coverage summary, baseline, and lane outcomes.
  - Added `dnadesign.devtools.quality_entropy` and a scheduled/workflow-dispatch CI job that uploads an entropy report artifact and fails on stale SOR metadata or broken quality evidence links.
  - Updated `QUALITY_SCORE.md` to treat CI evidence links and Codecov status signal as canonical numeric source inputs, keeping narrative guidance separate.
