## Infer Development Journal

This journal tracks `dnadesign.infer` refactor work over time: scope, decisions, evidence, tasks, and validation outcomes.

### Working Agreements

- Active branch for this stream: `fix/ops-postrun-resource-sizing`.
- HPC non-interference policy: during documentation/introspection phases, run read-only scheduler checks only; no `qsub`, `qdel`, or queue mutations.
- Refactor execution policy: small slices, explicit preflight/run/verify, and TDD for behavior-changing code.

## 2026-03-06 - Phase 0 Kickoff (Deep Introspection + IA Plan)

### Scheduler Safety Snapshot (Read-Only)

- `qstat -u esouth` at `2026-03-06` showed:
  - running jobs: `2` (`3626145`, `3626146`)
  - queued/hold jobs: `1` (`3626147`, state `hqw`)
- `sge-session-status.sh --warn-over-running 3` reported:
  - execution locus: `scc_login_shell`
  - threshold exceeded: `no`
- Status card recommendation: proceed with non-invasive work and verify gates.

### Deep Introspection Output

#### 1) Decision Summary

- Target scope: `src/dnadesign/infer` package plus immediate cross-tool contracts with `usr`, `notify`, and `ops`.
- Analysis depth: deep (package architecture, lifecycle, config mapping, contracts).
- Assumptions:
  - This pass is architecture/introspection only (no source moves yet).
  - Existing batch jobs remain untouched.

#### 2) Intent and Use-Case Map

- Problem intent:
  - run sequence-model inference from CLI/API;
  - support multiple ingest sources;
  - optionally write outputs back into USR datasets with resumable behavior.
- Primary use cases:
  - YAML-driven batch inference (`infer run`);
  - ad-hoc extraction (`infer extract`);
  - ad-hoc generation (`infer generate`);
  - preset-driven command simplification.
- Secondary use cases:
  - adapter/function introspection (`infer adapters list|fns`);
  - config and USR health validation (`infer validate config|usr`);
  - embedding inference service usage from Python (`run_extract`, `run_generate`, `run_job`).
- Non-goals in current implementation:
  - broad multi-model adapter coverage (only Evo2 is implemented; ESM2 remains stubbed).

#### 3) Core Functionality and Behavior Contract

- Public surfaces:
  - Python API dispatches by operation via `run_job`.
  - CLI commands are organized as `run`, `extract`, `generate`, plus subcommands (`presets`, `adapters`, `validate`).
- Core contracts:
  - config schema validates ingest and operation-specific requirements through Pydantic models.
  - output function names must resolve via registry-backed namespaced identifiers.
  - extract jobs enforce single adapter namespace per job.
- Failure behavior:
  - domain exceptions mapped to CLI exit codes (`ConfigError`, `ValidationError`, `ModelLoadError`, `RuntimeOOMError`, `WriteBackError`, and others).
  - unsafe `.pt` loading requires explicit opt-in gate.

#### 4) Lifecycle Model

- Phase A: command input normalization
  - config discovery or preset load;
  - CLI overrides merged into model/job config.
- Phase B: ingest and validation
  - source-specific ingest (`sequences`, `records`, `pt_file`, `usr`);
  - alphabet validation (`dna` or `protein`).
- Phase C: adapter resolution and execution
  - adapter instance cache keyed by `(model_id, device, precision)`;
  - output function resolution via registry;
  - micro-batching with OOM auto-derating.
- Phase D: write-back and completion
  - `records`/`.pt`/`usr` write-back based on job io config;
  - for USR, chunked attaches happen during execution and resume planning reads existing records/overlay state.

#### 5) Architecture View Stack

- Context view:
  - upstream inputs: CLI args, YAML config, presets, in-memory API calls.
  - downstream effects: output columns, write-back mutations, USR overlay updates, notify/ops contract consumption.
- Module view:
  - `cli.py` + `_console.py`: command and presentation layer.
  - `config.py` + `errors.py`: schema and error contracts.
  - `api.py` + `engine.py`: execution orchestration core.
  - `adapters/*`: model-specific implementation.
  - `ingest/*` + `writers/*`: boundary I/O adapters.
  - `presets/*`: packaged inference templates.
- Runtime interaction scenario (USR extract):
  - CLI -> config/job normalization -> `load_usr_input` -> `_plan_resume_for_usr` -> adapter fn execution -> `write_back_usr` chunk attaches -> result summary.

#### 6) Config-Schema to Behavior and Architecture Mapping

| Field | Validation/Default | Runtime effect | Impacted components |
| --- | --- | --- | --- |
| `model.id` | required | adapter class + namespace resolution | `config.py`, `registry.py`, `engine.py` |
| `model.device` | required | tensor/device placement and adapter construction key | `config.py`, `engine.py`, `adapters/evo2.py` |
| `model.precision` | `fp32|fp16|bf16` | autocast/compute mode | `config.py`, `adapters/evo2.py` |
| `model.batch_size` | optional | micro-batch size override | `engine.py` |
| `ingest.source` | constrained literal | selects ingest loader and write-back eligibility | `config.py`, `engine.py`, `ingest/sources.py` |
| `ingest.dataset/root/field/ids` | source-conditional checks | USR dataset binding and subset behavior | `config.py`, `ingest/sources.py` |
| `outputs[*].fn` | registry-resolved | adapter method dispatch | `config.py`, `registry.py`, `engine.py` |
| `outputs[*].format` | constrained literal | output conversion type | `config.py`, `utils.py`, `adapters/evo2.py` |
| `io.write_back` | default `false` | enables sink writes and USR attach behavior | `config.py`, `engine.py`, `writers/*` |
| `io.overwrite` | default `false` | controls attach/update behavior | `config.py`, `engine.py`, `writers/*` |

Environment precedence notes:

- `DNADESIGN_PROGRESS` toggles progress visibility in CLI/engine paths.
- `INFER_ALLOW_PICKLE` allows `.pt` ingest without command-line consent flag.
- `DNADESIGN_INFER_BATCH` and `DNADESIGN_INFER_DEFAULT_BS` influence micro-batching.
- `INFER_AUTO_DERATE_OOM` controls OOM fallback behavior.
- `DNADESIGN_USR_ROOT` influences default USR root discovery when explicit root is absent.

#### 7) Interaction Map

- `infer` <-> `usr`:
  - ingest loads dataset rows from `records.parquet`;
  - USR write-back attaches infer-prefixed columns to namespace `infer`.
- `infer` <-> `notify`:
  - notify resolves infer events streams through shared infer USR output contract resolver.
  - notify workspace resolver includes `infer_evo2` workspace route under `src/dnadesign/infer/workspaces`.
- `infer` <-> `ops`:
  - ops runbook schema includes infer workflow ids and notify compatibility rules.
  - ops planning pipeline includes infer config validation and infer-specific overlay guard tooling.

#### 8) Math and Operations Notes

- Extract execution complexity is approximately `O(num_outputs * num_sequences)` plus adapter cost.
- Resume planning in USR mode performs parquet scans and id-index joins to compute `todo_idx`.
- OOM handling is a halving strategy (`bs = bs // 2`) until successful or minimum batch size.
- Variable-length sequence handling in Evo2 adapter falls back to per-sequence execution when batching without padding is not valid.

#### 9) Evidence Ledger

| Claim | Evidence |
| --- | --- |
| API exposes run_extract/run_generate/run_job dispatch | `src/dnadesign/infer/api.py:22`, `src/dnadesign/infer/api.py:51`, `src/dnadesign/infer/api.py:80` |
| CLI primary commands are run/extract/generate | `src/dnadesign/infer/cli.py:126`, `src/dnadesign/infer/cli.py:274`, `src/dnadesign/infer/cli.py:419` |
| Schema validates source-specific ingest and job operation requirements | `src/dnadesign/infer/config.py:42`, `src/dnadesign/infer/config.py:91` |
| Engine performs USR resume planning and chunked write-back | `src/dnadesign/infer/engine.py:131`, `src/dnadesign/infer/engine.py:324`, `src/dnadesign/infer/engine.py:326` |
| USR write-back column naming contract is infer-prefixed | `src/dnadesign/infer/writers/usr.py:47`, `src/dnadesign/infer/README.md:13`, `src/dnadesign/infer/tests/test_infer_usr_docs_contract.py:25` |
| Presets are package-scanned and stem fallback exists | `src/dnadesign/infer/presets/registry.py:28`, `src/dnadesign/infer/presets/registry.py:69` |
| Adapter registration is import side effect and Evo2-only today | `src/dnadesign/infer/__init__.py:17`, `src/dnadesign/infer/adapters/__init__.py:19` |
| Notify uses shared infer output-contract resolver | `src/dnadesign/notify/events/source_builtin.py:19`, `src/dnadesign/notify/events/source_builtin.py:37` |
| Ops infer workflows and notify-tool contract enforcement exist | `src/dnadesign/ops/runbooks/schema.py:34`, `src/dnadesign/ops/runbooks/schema.py:35`, `src/dnadesign/ops/runbooks/schema.py:336` |
| Shared infer producer contract parser exists in `_contracts` | `src/dnadesign/_contracts/usr_producer.py:172` |

#### 10) Open Questions and Risk Notes

- `ingest/sources.py` default USR root fallback and `_contracts/usr_producer.py` infer resolver fallback differ in strictness; this should be normalized before deeper tool-chain coupling.
- Preset stem fallback (`load_preset`) may cause ambiguous selection if namespace collisions grow.
- `src/dnadesign/infer/workspaces` is referenced by notify resolvers but not currently present; ownership and lifecycle rules should be documented in infer docs.

### Create-Plan Artifact: Information Architecture Reorganization

#### Plan Intent Summary

Reorganize infer source and documentation surfaces for progressive disclosure while preserving current runtime behavior and external CLI/API contracts.

#### Explicit Scope

- In scope:
  - establish infer-local docs hierarchy (`src/dnadesign/infer/docs/*`);
  - split infer source layout into clearer layers over phased, behavior-preserving moves;
  - codify cross-tool contracts (usr/notify/ops) in infer docs.
- Out of scope:
  - changing infer external command semantics;
  - modifying running HPC jobs;
  - introducing backward-compatibility shims without explicit approval.

#### Ordered Action Checklist

1. Create infer docs scaffold and index pages:
   - `docs/README.md`, `docs/architecture/`, `docs/reference/`, `docs/operations/`, `docs/dev/`.
2. Capture architecture baseline in docs:
   - module map, lifecycle map, config-field map, cross-tool contracts.
3. Define target source layout for progressive disclosure:
   - separate interface (`cli/api`), core execution, boundaries (`ingest`, `writers`), and adapters.
4. Execute first refactor slice under TDD:
   - add failing tests for a single move/extraction;
   - implement minimal code movement;
   - keep behavior and public imports stable.
5. Update docs and contract tests in same slice:
   - enforce docs-contract tests for moved paths/links where applicable.
6. Integrate harness checks for each slice:
   - preflight (`git status`, targeted tests, docs link checks), run (single-slice change), verify (test + contract outputs).
7. Repeat slices until target IA is reached.

#### Validation and Risk Handling

- Required checks per slice:
  - `uv run pytest -q src/dnadesign/infer/tests`
  - targeted cross-tool contract tests for notify/ops when infer contract parsing is touched.
- Risk controls:
  - only one behavior-affecting change per slice;
  - keep source moves small and reviewable;
  - stop on first contract drift and re-baseline before continuing.

#### Blockers (Current)

- No hard blockers for phase-1 docs and planning.
- Design decision needed before code moves: strictness alignment for infer USR root resolution across infer runtime and shared contracts.

### File-Organizer Style Inventory (Preview Only, No Moves Executed)

Current infer surface (`src/dnadesign/infer`):

- interface: `cli.py`, `api.py`, `__main__.py`, `_console.py`, `_logging.py`
- runtime core: `engine.py`, `config.py`, `registry.py`, `errors.py`, `utils.py`
- boundaries: `ingest/`, `writers/`
- extension: `adapters/`, `presets/`
- tests/docs/assets: `tests/`, `README.md`, `assets/`

Proposed progressive-disclosure target (phase design, not yet applied):

- `src/dnadesign/infer/`
  - `interface/` (`cli.py`, `_console.py`)
  - `application/` (`api.py`)
  - `core/` (`engine.py`, `config.py`, `errors.py`, `registry.py`)
  - `boundaries/ingest/`, `boundaries/writers/`
  - `adapters/`
  - `presets/`
  - `docs/` (local package docs, contracts, dev notes)
  - `tests/`

### Harness Engineering Contract (Refactor Cycle 1)

- Setting: codebase + documentation.
- Selected endpoints:
  - `knowledge-integrity`
  - `review-merge`
  - `architecture-invariants`
- Bottlenecks -> interventions -> metrics:
  - scattered infer knowledge -> local infer docs map -> fewer cross-file hops to answer architecture questions.
  - risky refactor slices -> explicit preflight/run/verify checklist -> lower retry count.
  - contract drift risk across usr/notify/ops -> shared contract evidence checks -> stable cross-tool tests.

Deterministic path for each slice:

1. Preflight: verify branch/worktree state and collect impacted contract files.
2. Run: one bounded change set.
3. Verify: infer tests + relevant cross-tool contract tests.
4. Record: add journal entry with evidence and next slice.

### Task Board

- [x] Capture infer package introspection baseline.
- [x] Capture HPC read-only safety snapshot.
- [x] Create infer-local development journal.
- [x] Create infer docs index scaffold for progressive disclosure.
- [ ] Decide strictness contract for infer USR root resolution (`infer` runtime vs shared `_contracts` parser).
- [x] Execute first behavior-preserving source-layout slice under TDD.

## 2026-03-06 - Phase 1 Slice A (Docs Scaffold + Input Parsing Extraction)

### Implemented

- Added infer-local docs scaffold:
  - `src/dnadesign/infer/docs/README.md`
  - `src/dnadesign/infer/docs/architecture/README.md`
  - `src/dnadesign/infer/docs/reference/README.md`
  - `src/dnadesign/infer/docs/operations/README.md`
  - `src/dnadesign/infer/docs/dev/README.md`
- Added infer package README link to infer-local docs index.
- Added shared input parsing helper module:
  - `src/dnadesign/infer/input_parsing.py`
- Refactored `src/dnadesign/infer/cli.py` to use shared helpers:
  - removed duplicated `_read_ids_arg` and per-command `_load_lines` closures,
  - rewired to `read_ids_arg` and `load_nonempty_lines`.

### TDD Evidence

- Red:
  - added `src/dnadesign/infer/tests/test_input_parsing.py`
  - initial run failed with `ModuleNotFoundError: No module named 'dnadesign.infer.input_parsing'`
- Green:
  - implemented `input_parsing.py`
  - updated `cli.py` call sites
  - test run passed:
    - `uv run pytest -q src/dnadesign/infer/tests/test_input_parsing.py`
    - `uv run pytest -q src/dnadesign/infer/tests/test_input_parsing.py src/dnadesign/infer/tests/test_presets.py src/dnadesign/infer/tests/test_infer_usr_docs_contract.py`
    - `uv run pytest -q src/dnadesign/infer/tests`

### Notes

- Refactor remains behavior-preserving for existing CLI contracts.
- No scheduler mutation commands were run during this slice.

## 2026-03-06 - Phase 1 Slice B (Maintainer Audit + Contract Hardening)

### Audit Scope

- Package: `src/dnadesign/infer`
- Primary user surfaces:
  - CLI: `infer run`, `infer extract`, `infer presets`, `infer validate`
  - API/runtime: `run_extract_job`, `run_generate_job`
- Pressure objective:
  - keep infer model-agnostic by explicit namespace contract,
  - verify USR namespaced write-back columns remain deterministic,
  - provide standalone and ops-runbook pressure-test commands.

### Baseline Evidence

- Scheduler (read-only):
  - `qstat -u esouth` showed 2 running + 1 held job (`hqw`), no Eqw.
  - `sge-session-status.sh --warn-over-running 3` reported threshold not exceeded.
- Tests:
  - baseline infer suite green before hardening pass.
- CLI:
  - `uv run infer --help`
  - `uv run infer presets list`
  - `uv run infer extract --preset evo2/extract_logits_ll --seq ACGT --dry-run`

### Prioritized Findings

1. High: missing fail-fast namespace contract in runtime dispatch
   - extract path accepted output namespaces that did not match model namespace.
   - generate path accepted explicit `job.fn` namespaces that did not match model namespace.
   - risk: accidental cross-namespace dispatch and ambiguous adapter behavior under extension.
2. Medium: infer USR output naming logic duplicated in multiple modules
   - writer and resume planner constructed infer column names independently.
   - risk: drift in namespaced contract over time.
3. Medium: ops pressure-test docs path had contract gaps
   - notify-enabled scaffold requires webhook secret wiring.
   - audit json path must be under `outputs/logs/ops/audit/`.
   - risk: operator friction and false-start runbook execution.

### Changes Applied

- Added explicit infer contracts module:
  - `src/dnadesign/infer/contracts.py`
  - fail-fast namespace validation for extract/generate dispatch.
  - centralized infer USR column-name builder.
- Hardened runtime:
  - `src/dnadesign/infer/engine.py`
    - extract now validates output namespace against model namespace before adapter load.
    - generate now validates explicit fn namespace against model namespace before adapter load.
    - resume planner now uses centralized infer USR column-name builder.
  - `src/dnadesign/infer/writers/usr.py`
    - uses shared infer USR column-name builder.
- Added adversarial and pressure tests:
  - `src/dnadesign/infer/tests/test_namespace_contracts.py`
    - mismatch extract namespace fails fast.
    - mismatch generate namespace fails fast.
    - agnostic model + USR pressure path verifies infer-prefixed namespaced column attach.
- Added pressure-test operations docs:
  - `src/dnadesign/infer/docs/operations/pressure-test-agnostic-models.md`
  - `src/dnadesign/infer/docs/operations/examples/pressure_test_infer_config.yaml`
  - updated `src/dnadesign/infer/docs/operations/README.md`
- Added docs contract test:
  - `src/dnadesign/infer/tests/test_pressure_runbook_docs_contract.py`

### Adversarial Hardening Evidence

- Red-to-green TDD evidence for namespace mismatch:
  - initial failures: mismatch namespace tests did not raise `ConfigError`.
  - after hardening: mismatch tests pass with fail-fast errors.
- Agnostic-model pressure path:
  - test adapter namespace path writes expected infer-prefixed USR columns:
    - `infer__<model_id>__<job_id>__logits`
    - `infer__<model_id>__<job_id>__llr`

### Runbook Path Verification Notes

- `ops runbook` no-submit path was executed successfully with:
  - workspace config at `<workspace-root>/config.yaml`
  - audit json path under `<workspace-root>/outputs/logs/ops/audit/*.json`
  - `--no-notify` baseline
- notify-enabled path remains optional and requires explicit webhook secret wiring.

### Verification Commands (Executed)

- `uv run pytest -q src/dnadesign/infer/tests/test_namespace_contracts.py`
- `uv run pytest -q src/dnadesign/infer/tests/test_pressure_runbook_docs_contract.py`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run ops runbook init --runbook <tmp>/infer-pressure.runbook.yaml --workflow infer --workspace-root <tmp>/workspace --id infer_pressure_test --no-notify`
- `uv run ops runbook plan --runbook <tmp>/infer-pressure.runbook.yaml`
- `uv run ops runbook execute --runbook <tmp>/infer-pressure.runbook.yaml --audit-json <tmp>/workspace/outputs/logs/ops/audit/infer-pressure.audit.json --no-submit`

### Task Board

- [x] Add fail-fast namespace contract hardening for extract/generate runtime dispatch.
- [x] Add adversarial namespace pressure tests.
- [x] Add infer pressure-test operations runbook docs and config example.
- [ ] Next slice candidate: extract adapter-dispatch block from `engine.py` into a dedicated module with invariant tests.

## 2026-03-06 - Phase 1 Slice C (Adapter Dispatch Module Extraction)

### Scope

- Objective: extract adapter function dispatch from `engine.py` into a dedicated module with invariant tests.
- Constraint: behavior must remain unchanged for existing extract/generate flows.

### Changes Applied

- Added dedicated adapter dispatch module:
  - `src/dnadesign/infer/adapter_dispatch.py`
  - responsibilities:
    - resolve extract callable from namespaced function,
    - invoke extract callable with v1 method contract (`log_likelihood`, `logits`, `embedding`),
    - resolve generate callable from namespaced function.
- Refactored runtime to use dispatch module:
  - `src/dnadesign/infer/engine.py`
  - removed inline adapter capability/dispatch logic from job loops and delegated to module helpers.
- Added invariant tests:
  - `src/dnadesign/infer/tests/test_adapter_dispatch.py`
  - covers callable resolution, missing method failures, format forwarding for logits, and unsupported method rejection.

### TDD Evidence

- Red:
  - added `src/dnadesign/infer/tests/test_adapter_dispatch.py` importing new module.
  - initial run failed with:
    - `ModuleNotFoundError: No module named 'dnadesign.infer.adapter_dispatch'`
- Green:
  - implemented `adapter_dispatch.py`
  - rewired `engine.py` call sites
  - verification passed:
    - `uv run pytest -q src/dnadesign/infer/tests/test_adapter_dispatch.py`
    - `uv run pytest -q src/dnadesign/infer/tests/test_namespace_contracts.py`
    - `uv run pytest -q src/dnadesign/infer/tests`

### Notes

- This slice narrows `engine.py` responsibilities without changing public CLI/API behavior.
- Dispatch invariants are now unit-testable independently of ingest/write-back paths.

### Task Board

- [x] Add fail-fast namespace contract hardening for extract/generate runtime dispatch.
- [x] Add adversarial namespace pressure tests.
- [x] Add infer pressure-test operations runbook docs and config example.
- [x] Extract adapter-dispatch block from `engine.py` into dedicated module with invariant tests.
- [ ] Next slice candidate: split ingest loading branch from `run_extract_job` into a focused helper with parity tests.

## 2026-03-06 - Phase 1 Slice D (Extract Ingest Helper Split)

### Scope

- Objective: split ingest-source branch loading from `run_extract_job` into a focused helper with parity tests.
- Constraint: preserve existing ingest behavior and error messages.

### Changes Applied

- Added ingest helper in runtime module:
  - `src/dnadesign/infer/engine.py`
  - new helper: `_load_extract_ingest(inputs, ingest=...)`
  - handles `sequences`, `records`, `pt_file`, and `usr` source contracts and returns normalized ingest state tuple.
- Refactored `run_extract_job` to delegate ingest state construction to helper.
- Added parity tests:
  - `src/dnadesign/infer/tests/test_extract_ingest_helper.py`
  - coverage:
    - sequences path
    - records path (field fallback)
    - pt-file path validation error
    - usr argument forwarding
    - unknown source fail-fast

### TDD Evidence

- Red:
  - added `src/dnadesign/infer/tests/test_extract_ingest_helper.py`
  - initial run failed with:
    - `ImportError: cannot import name '_load_extract_ingest' from 'dnadesign.infer.engine'`
- Green:
  - implemented `_load_extract_ingest` and rewired `run_extract_job`
  - resolved one test fixture issue by using a lightweight ingest stub for records field-fallback parity
  - verification passed:
    - `uv run pytest -q src/dnadesign/infer/tests/test_extract_ingest_helper.py`
    - `uv run pytest -q src/dnadesign/infer/tests`

### Notes

- Ingest source branching is now isolated to one helper, reducing `run_extract_job` surface area.
- Existing engine monkeypatch seams used by other tests remained stable.

### Task Board

- [x] Add fail-fast namespace contract hardening for extract/generate runtime dispatch.
- [x] Add adversarial namespace pressure tests.
- [x] Add infer pressure-test operations runbook docs and config example.
- [x] Extract adapter-dispatch block from `engine.py` into dedicated module with invariant tests.
- [x] Split ingest loading branch from `run_extract_job` into a focused helper with parity tests.
- [ ] Next slice candidate: split generate ingest loading branch from `run_generate_job` into a focused helper with parity tests.

## 2026-03-06 - Phase 1 Slice E (Generate Ingest Helper Split, DRY Contract)

### Scope

- Objective: split generate ingest loading from `run_generate_job` into a focused helper.
- Pragmatic constraint: keep ingest-source knowledge DRY by using one fail-fast ingest contract path.

### Boundary and Contract Decision

- Added `_load_generate_ingest(inputs, ingest=...)` in `engine.py`.
- `_load_generate_ingest` delegates to `_load_extract_ingest` and returns only prompts.
- Result: one source of truth for ingest source handling (`sequences`, `records`, `pt_file`, `usr`) and one fail-fast error surface for unknown sources/invalid pt-file input types.

### Changes Applied

- Runtime:
  - `src/dnadesign/infer/engine.py`
  - `run_generate_job` now calls `_load_generate_ingest` instead of maintaining an inline ingest branch.
- Tests:
  - added `src/dnadesign/infer/tests/test_generate_ingest_helper.py`
  - parity coverage for all source paths and fail-fast behavior.

### TDD Evidence

- Red:
  - added `src/dnadesign/infer/tests/test_generate_ingest_helper.py`
  - initial run failed with:
    - `ImportError: cannot import name '_load_generate_ingest' from 'dnadesign.infer.engine'`
- Green:
  - implemented `_load_generate_ingest` as delegation to `_load_extract_ingest`
  - rewired `run_generate_job`
  - verification passed:
    - `uv run pytest -q src/dnadesign/infer/tests/test_generate_ingest_helper.py`
    - `uv run pytest -q src/dnadesign/infer/tests/test_extract_ingest_helper.py`
    - `uv run pytest -q src/dnadesign/infer/tests`

### Notes

- This is a reversible refactor: if needed, helper delegation can be changed without public API changes.
- No silent fallback paths were introduced.

### Task Board

- [x] Add fail-fast namespace contract hardening for extract/generate runtime dispatch.
- [x] Add adversarial namespace pressure tests.
- [x] Add infer pressure-test operations runbook docs and config example.
- [x] Extract adapter-dispatch block from `engine.py` into dedicated module with invariant tests.
- [x] Split ingest loading branch from `run_extract_job` into a focused helper with parity tests.
- [x] Split generate ingest loading branch from `run_generate_job` into a focused helper with parity tests.
- [ ] Next slice candidate: isolate micro-batch/derating execution loop in extract path behind a helper with chunk-contract tests.
