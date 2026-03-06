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

## 2026-03-06 - Phase 1 Slice F (Extract Execution Module, Maintainer Hardening)

### Skill Composition Decision

- Coordinator: `maintainer-audit-hardening`.
- Applied principles: `pragmatic-programming-principles` (DRY knowledge, orthogonal boundaries, explicit contracts, no silent fallback).
- `file-organizer` note: source-repo refactoring is out-of-scope for that skill; only inventory/IA guidance pattern was applied (no destructive file moves, no broad directory reshuffle).

### Baseline Audit Evidence (Pre-change)

- Branch state clean and ahead:
  - `git status --short --branch`
- Tests baseline:
  - `uv run pytest -q src/dnadesign/infer/tests`
- Primary CLI user paths:
  - `uv run infer --help`
  - `uv run infer extract --preset evo2/extract_logits_ll --seq ACGT --dry-run`

### Prioritized Finding

1. Medium: extract runtime execution loop remained monolithic in `engine.py`.
   - the chunk loop mixed batching policy, OOM derating, adapter invocation, per-chunk persistence, and progress updates in one block.
   - risk: change surface too wide for future edits and pressure-hardening; higher regression likelihood.

### Boundary and Contract Decisions

- Added `src/dnadesign/infer/extract_execution.py` with `execute_extract_output(...)` as a dedicated chunk execution boundary.
- Contract preserved and made explicit in one place:
  - fail-fast `RuntimeOOMError` on non-derated or exhausted OOM cases,
  - fail-fast chunk output cardinality check,
  - explicit callback contracts for progress and per-chunk write-back.
- `engine.py` now orchestrates and delegates execution details, reducing monolithic sprawl.

### TDD and Adversarial Evidence

- Red:
  - added `src/dnadesign/infer/tests/test_extract_execution.py`
  - initial run failed with:
    - `ModuleNotFoundError: No module named 'dnadesign.infer.extract_execution'`
- Green:
  - implemented `extract_execution.py`
  - rewired `run_extract_job` to call `execute_extract_output`
- Adversarial tests now cover:
  - OOM derating retry path,
  - non-derated OOM fail-fast path,
  - wrong chunk output length fail-fast path,
  - hook and value propagation contracts.

### Verification Commands (Executed)

- `uv run pytest -q src/dnadesign/infer/tests/test_extract_execution.py`
- `uv run pytest -q src/dnadesign/infer/tests/test_namespace_contracts.py src/dnadesign/infer/tests/test_usr_writeback_contract.py`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run infer extract --preset evo2/extract_logits_ll --seq ACGT --dry-run`

### Information Architecture Update

- Updated `src/dnadesign/infer/docs/architecture/README.md` runtime map to include:
  - `adapter_dispatch.py` (dispatch contracts)
  - `extract_execution.py` (chunk execution loop)

### Task Board

- [x] Add fail-fast namespace contract hardening for extract/generate runtime dispatch.
- [x] Add adversarial namespace pressure tests.
- [x] Add infer pressure-test operations runbook docs and config example.
- [x] Extract adapter-dispatch block from `engine.py` into dedicated module with invariant tests.
- [x] Split ingest loading branch from `run_extract_job` into a focused helper with parity tests.
- [x] Split generate ingest loading branch from `run_generate_job` into a focused helper with parity tests.
- [x] Isolate extract micro-batch/derating execution loop behind a helper with chunk-contract tests.
- [ ] Next slice candidate: isolate generate micro-batch execution loop behind a helper with generation-output contract tests.

## 2026-03-06 - Phase 1 Slice G (Generate Execution Module, No-Silent-Fallback Contract)

### Scope

- Objective: isolate generate micro-batch execution from `engine.py` into a focused helper module.
- Hardening objective: remove silent fallback behavior in chunked generation output handling.

### Prioritized Finding

1. Medium: generate chunk path used permissive output extraction (`out.get('gen_seqs', [])`).
   - risk: malformed adapter output could silently produce partial/empty results and hide defects.

### Boundary and Contract Decisions

- Added `src/dnadesign/infer/generate_execution.py` with:
  - `validate_generate_payload(payload)`
  - `execute_generate_batches(...)`
- `run_generate_job` now delegates chunked execution to `execute_generate_batches`.
- Explicit contract enforced in both chunked and non-chunked paths:
  - payload must be a mapping,
  - payload must include `gen_seqs`,
  - `gen_seqs` must be a list,
  - per-chunk generated sequence count must equal prompt chunk size.

### TDD and Adversarial Evidence

- Red:
  - added `src/dnadesign/infer/tests/test_generate_execution.py`
  - initial run failed with:
    - `ModuleNotFoundError: No module named 'dnadesign.infer.generate_execution'`
- Green:
  - implemented `generate_execution.py`
  - rewired `engine.py` generate path
- Adversarial tests cover:
  - OOM derating retry,
  - non-derated OOM fail-fast,
  - missing `gen_seqs` fail-fast,
  - prompt/output cardinality mismatch fail-fast.

### Verification Commands (Executed)

- `uv run pytest -q src/dnadesign/infer/tests/test_generate_execution.py`
- `uv run pytest -q src/dnadesign/infer/tests/test_namespace_contracts.py`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run infer generate --model-id evo2_7b --device cpu --precision bf16 --alphabet dna --prompt ACGT --max-new-tokens 4 --dry-run`

### Information Architecture Update

- Runtime map updated in `src/dnadesign/infer/docs/architecture/README.md`:
  - added `generate_execution.py` as dedicated generation execution boundary.

### Task Board

- [x] Add fail-fast namespace contract hardening for extract/generate runtime dispatch.
- [x] Add adversarial namespace pressure tests.
- [x] Add infer pressure-test operations runbook docs and config example.
- [x] Extract adapter-dispatch block from `engine.py` into dedicated module with invariant tests.
- [x] Split ingest loading branch from `run_extract_job` into a focused helper with parity tests.
- [x] Split generate ingest loading branch from `run_generate_job` into a focused helper with parity tests.
- [x] Isolate extract micro-batch/derating execution loop behind a helper with chunk-contract tests.
- [x] Isolate generate micro-batch execution loop behind a helper with generation-output contract tests.
- [ ] Next slice candidate: isolate progress-handle creation and lifecycle into a shared helper to remove repeated pbar setup/close logic.

## 2026-03-06 - Phase 1 Slice H (Progress Lifecycle Module Extraction)

### Scope

- Objective: remove repeated progress-handle setup logic from `engine.py` by introducing a dedicated module.
- Constraint: preserve existing CLI/runtime behavior for progress enabled/disabled modes.

### Prioritized Finding

1. Low-Medium: duplicated progress setup and tqdm fallback logic in extract and generate paths.
   - risk: divergence between code paths and larger `engine.py` maintenance surface.

### Boundary and Contract Decisions

- Added `src/dnadesign/infer/progress.py` with explicit progress contracts:
  - `resolve_tqdm_factory()`
  - `create_progress_handle(...)`
  - `_NoTQDM` no-op handle.
- `engine.py` now imports `ProgressFactory` and `create_progress_handle` and no longer owns progress fallback implementation.
- No silent fallback behavior added; missing tqdm remains operator-visible via existing logger info.

### TDD Evidence

- Red:
  - added `src/dnadesign/infer/tests/test_progress_handles.py`
  - initial run failed with:
    - `ModuleNotFoundError: No module named 'dnadesign.infer.progress'`
- Green:
  - implemented `progress.py`
  - rewired engine progress-handle call sites
  - verification passed:
    - `uv run pytest -q src/dnadesign/infer/tests/test_progress_handles.py`
    - `uv run pytest -q src/dnadesign/infer/tests`

### Verification Commands (Executed)

- `uv run pytest -q src/dnadesign/infer/tests/test_progress_handles.py`
- `uv run pytest -q src/dnadesign/infer/tests/test_extract_execution.py src/dnadesign/infer/tests/test_generate_execution.py`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run infer extract --preset evo2/extract_logits_ll --seq ACGT --dry-run`
- `uv run infer generate --model-id evo2_7b --device cpu --precision bf16 --alphabet dna --prompt ACGT --max-new-tokens 4 --dry-run`

### Information Architecture Update

- Runtime map updated in `src/dnadesign/infer/docs/architecture/README.md` with `progress.py`.

### Task Board

- [x] Add fail-fast namespace contract hardening for extract/generate runtime dispatch.
- [x] Add adversarial namespace pressure tests.
- [x] Add infer pressure-test operations runbook docs and config example.
- [x] Extract adapter-dispatch block from `engine.py` into dedicated module with invariant tests.
- [x] Split ingest loading branch from `run_extract_job` into a focused helper with parity tests.
- [x] Split generate ingest loading branch from `run_generate_job` into a focused helper with parity tests.
- [x] Isolate extract micro-batch/derating execution loop behind a helper with chunk-contract tests.
- [x] Isolate generate micro-batch execution loop behind a helper with generation-output contract tests.
- [x] Isolate progress-handle creation and lifecycle into a shared helper.
- [ ] Next slice candidate: split final write-back dispatch in extract path into a dedicated helper with contract tests.

## 2026-03-06 - Phase 1 Slice I (Extract Final Write-Back Dispatch Module)

### Scope

- Objective: isolate extract final write-back dispatch from `engine.py` into a dedicated boundary module.
- Constraint: preserve chunk-level USR write-back behavior and keep existing source contracts unchanged.

### Prioritized Finding

1. Low-Medium: final write-back source dispatch remained embedded in `run_extract_job`.
   - risk: unnecessary orchestration coupling and slower iteration for source-specific write-back hardening.

### Boundary and Contract Decisions

- Added `src/dnadesign/infer/writeback_dispatch.py` with:
  - `run_extract_write_back(...)`
- Explicit source-contract behavior:
  - `records` -> `write_back_records(...)`
  - `pt_file` -> `write_back_pt_file(...)`
  - `usr` -> fail-fast if `ids` or dataset handle missing; otherwise no final-op (chunk writes remain authoritative)
  - any other source -> `WriteBackError` fail-fast.
- `engine.py` now delegates final write-back dispatch to `run_extract_write_back(...)`.

### TDD Evidence

- Red:
  - added `src/dnadesign/infer/tests/test_writeback_dispatch.py`
  - initial run failed with:
    - `ModuleNotFoundError: No module named 'dnadesign.infer.writeback_dispatch'`
- Green:
  - implemented `writeback_dispatch.py`
  - rewired `run_extract_job` to use `run_extract_write_back(...)`
  - verification passed for targeted contracts and existing USR/namespace regressions.

### Verification Commands (Executed)

- `uv run pytest -q src/dnadesign/infer/tests/test_writeback_dispatch.py src/dnadesign/infer/tests/test_usr_writeback_contract.py src/dnadesign/infer/tests/test_namespace_contracts.py`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run infer extract --preset evo2/extract_logits_ll --seq ACGT --dry-run`
- `uv run infer generate --model-id evo2_7b --device cpu --precision bf16 --alphabet dna --prompt ACGT --max-new-tokens 4 --dry-run`

### Information Architecture Update

- Runtime map updated in `src/dnadesign/infer/docs/architecture/README.md` with:
  - `writeback_dispatch.py` as final extract write-back dispatch boundary.

### Task Board

- [x] Add fail-fast namespace contract hardening for extract/generate runtime dispatch.
- [x] Add adversarial namespace pressure tests.
- [x] Add infer pressure-test operations runbook docs and config example.
- [x] Extract adapter-dispatch block from `engine.py` into dedicated module with invariant tests.
- [x] Split ingest loading branch from `run_extract_job` into a focused helper with parity tests.
- [x] Split generate ingest loading branch from `run_generate_job` into a focused helper with parity tests.
- [x] Isolate extract micro-batch/derating execution loop behind a helper with chunk-contract tests.
- [x] Isolate generate micro-batch execution loop behind a helper with generation-output contract tests.
- [x] Isolate progress-handle creation and lifecycle into a shared helper.
- [x] Split final write-back dispatch in extract path into a dedicated helper with contract tests.
- [ ] Next slice candidate: isolate `_plan_resume_for_usr(...)` from `engine.py` behind a dedicated resume planner module with overlay/records parity tests.

## 2026-03-06 - Phase 1 Slice J (USR Resume Planner Module Extraction)

### Scope

- Objective: extract `_plan_resume_for_usr(...)` from `engine.py` to an explicit module boundary.
- Constraint: preserve existing overlay-aware resume behavior and engine compatibility hooks used by tests.

### Prioritized Finding

1. Medium: USR resume planning remained a large in-engine block mixing parquet scans, overlay merge, and todo index derivation.
   - risk: orchestration drift in `engine.py` and broader blast radius for future resume hardening.

### Boundary and Contract Decisions

- Added `src/dnadesign/infer/resume_planner.py` with:
  - `plan_resume_for_usr(...)`
- Kept engine compatibility contract by importing alias:
  - `from .resume_planner import plan_resume_for_usr as _plan_resume_for_usr`
- Explicit fail-fast contract preserved:
  - unreadable records table -> `WriteBackError`,
  - overlay values override records values only when non-null,
  - overwrite/empty dataset bypasses scans and returns full todo index.

### TDD Evidence

- Red:
  - added `src/dnadesign/infer/tests/test_resume_planner.py`
  - initial run failed with:
    - `ModuleNotFoundError: No module named 'dnadesign.infer.resume_planner'`
- Green:
  - implemented `resume_planner.py`
  - removed in-engine implementation and wired alias import
  - verification passed for new module tests and existing USR/namespace regressions.

### Verification Commands (Executed)

- `uv run pytest -q src/dnadesign/infer/tests/test_resume_planner.py src/dnadesign/infer/tests/test_usr_writeback_contract.py src/dnadesign/infer/tests/test_namespace_contracts.py`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run infer extract --preset evo2/extract_logits_ll --seq ACGT --dry-run`
- `uv run infer generate --model-id evo2_7b --device cpu --precision bf16 --alphabet dna --prompt ACGT --max-new-tokens 4 --dry-run`

### Information Architecture Update

- Runtime map updated in `src/dnadesign/infer/docs/architecture/README.md` with:
  - `resume_planner.py` as USR resume planning boundary.

### Task Board

- [x] Add fail-fast namespace contract hardening for extract/generate runtime dispatch.
- [x] Add adversarial namespace pressure tests.
- [x] Add infer pressure-test operations runbook docs and config example.
- [x] Extract adapter-dispatch block from `engine.py` into dedicated module with invariant tests.
- [x] Split ingest loading branch from `run_extract_job` into a focused helper with parity tests.
- [x] Split generate ingest loading branch from `run_generate_job` into a focused helper with parity tests.
- [x] Isolate extract micro-batch/derating execution loop behind a helper with chunk-contract tests.
- [x] Isolate generate micro-batch execution loop behind a helper with generation-output contract tests.
- [x] Isolate progress-handle creation and lifecycle into a shared helper.
- [x] Split final write-back dispatch in extract path into a dedicated helper with contract tests.
- [x] Isolate `_plan_resume_for_usr(...)` behind `resume_planner.py` with parity coverage.
- [ ] Next slice candidate: separate extract chunk write-back callback construction from `run_extract_job` to reduce closure coupling and allow direct callback-contract tests.

## 2026-03-06 - Phase 1 Slice K (Extract Chunk Write-Back Callback Module)

### Scope

- Objective: remove inline USR chunk write-back closure from `run_extract_job` into a dedicated callback-construction module.
- Constraint: preserve chunk write-back chunking semantics and existing monkeypatch-based tests.

### Prioritized Finding

1. Medium: extract chunk write-back callback was an inline closure in orchestration code.
   - risk: tighter coupling between runtime execution loop and sink-specific write-back behavior, with weaker unit-level contract visibility.

### Boundary and Contract Decisions

- Added `src/dnadesign/infer/extract_chunk_writeback.py` with:
  - `build_extract_chunk_write_back(...)`
- Explicit callback contract:
  - non-`usr` source or disabled write-back -> return `None` callback,
  - `usr` + write-back requires both `ids` and dataset handle, else fail-fast `WriteBackError`,
  - callback writes only chunk ids and chunk values for the active output id.
- Preserved testability boundary by injecting `writer` callable; engine passes `write_back_usr`.

### TDD Evidence

- Red:
  - added `src/dnadesign/infer/tests/test_extract_chunk_writeback.py`
  - initial run failed with:
    - `ModuleNotFoundError: No module named 'dnadesign.infer.extract_chunk_writeback'`
- Green:
  - implemented `extract_chunk_writeback.py`
  - rewired `run_extract_job` to call `build_extract_chunk_write_back(...)`
  - verification passed for new callback tests and existing USR/namespace regressions.

### Verification Commands (Executed)

- `uv run pytest -q src/dnadesign/infer/tests/test_extract_chunk_writeback.py src/dnadesign/infer/tests/test_usr_writeback_contract.py src/dnadesign/infer/tests/test_namespace_contracts.py`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run infer extract --preset evo2/extract_logits_ll --seq ACGT --dry-run`
- `uv run infer generate --model-id evo2_7b --device cpu --precision bf16 --alphabet dna --prompt ACGT --max-new-tokens 4 --dry-run`

### Information Architecture Update

- Runtime map updated in `src/dnadesign/infer/docs/architecture/README.md` with:
  - `extract_chunk_writeback.py` as extract chunk write-back callback boundary.

### Task Board

- [x] Add fail-fast namespace contract hardening for extract/generate runtime dispatch.
- [x] Add adversarial namespace pressure tests.
- [x] Add infer pressure-test operations runbook docs and config example.
- [x] Extract adapter-dispatch block from `engine.py` into dedicated module with invariant tests.
- [x] Split ingest loading branch from `run_extract_job` into a focused helper with parity tests.
- [x] Split generate ingest loading branch from `run_generate_job` into a focused helper with parity tests.
- [x] Isolate extract micro-batch/derating execution loop behind a helper with chunk-contract tests.
- [x] Isolate generate micro-batch execution loop behind a helper with generation-output contract tests.
- [x] Isolate progress-handle creation and lifecycle into a shared helper.
- [x] Split final write-back dispatch in extract path into a dedicated helper with contract tests.
- [x] Isolate `_plan_resume_for_usr(...)` behind `resume_planner.py` with parity coverage.
- [x] Separate extract chunk write-back callback construction into `extract_chunk_writeback.py`.
- [ ] Next slice candidate: isolate adapter cache and loader concerns from `engine.py` into a dedicated adapter runtime module with cache-contract tests.

## 2026-03-06 - Phase 1 Slice L (Adapter Runtime/Cache Module Extraction)

### Scope

- Objective: extract adapter cache/loading and runtime policy helpers from `engine.py` into a dedicated runtime module.
- Constraint: preserve existing `engine` monkeypatch seam names used by tests (`_get_adapter`, `_is_oom`, `_auto_derate_enabled`).

### Prioritized Finding

1. Medium: adapter cache and load policy were embedded in orchestration module.
   - risk: broader regression surface for adapter policy changes and reduced test isolation for load/cache contracts.

### Boundary and Contract Decisions

- Added `src/dnadesign/infer/adapter_runtime.py` with explicit helper contracts:
  - `clear_adapter_cache()`
  - `get_adapter(...)`
  - `is_oom(...)`
  - `auto_derate_enabled()`
- `engine.py` now imports alias bindings:
  - `get_adapter as _get_adapter`
  - `is_oom as _is_oom`
  - `auto_derate_enabled as _auto_derate_enabled`
- Error contract preserved:
  - `InferError` subclasses are re-raised,
  - non-infer exceptions are wrapped as `ModelLoadError`.

### TDD Evidence

- Red:
  - added `src/dnadesign/infer/tests/test_adapter_runtime.py`
  - initial run failed with:
    - `ModuleNotFoundError: No module named 'dnadesign.infer.adapter_runtime'`
- Green:
  - implemented `adapter_runtime.py`
  - rewired `engine.py` to import alias helpers
  - verification passed for new runtime tests and existing regressions.

### Verification Commands (Executed)

- `uv run pytest -q src/dnadesign/infer/tests/test_adapter_runtime.py src/dnadesign/infer/tests/test_namespace_contracts.py src/dnadesign/infer/tests/test_usr_writeback_contract.py`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run infer extract --preset evo2/extract_logits_ll --seq ACGT --dry-run`
- `uv run infer generate --model-id evo2_7b --device cpu --precision bf16 --alphabet dna --prompt ACGT --max-new-tokens 4 --dry-run`

### Information Architecture Update

- Runtime map updated in `src/dnadesign/infer/docs/architecture/README.md` with:
  - `adapter_runtime.py` as adapter cache/runtime policy boundary.

### Task Board

- [x] Add fail-fast namespace contract hardening for extract/generate runtime dispatch.
- [x] Add adversarial namespace pressure tests.
- [x] Add infer pressure-test operations runbook docs and config example.
- [x] Extract adapter-dispatch block from `engine.py` into dedicated module with invariant tests.
- [x] Split ingest loading branch from `run_extract_job` into a focused helper with parity tests.
- [x] Split generate ingest loading branch from `run_generate_job` into a focused helper with parity tests.
- [x] Isolate extract micro-batch/derating execution loop behind a helper with chunk-contract tests.
- [x] Isolate generate micro-batch execution loop behind a helper with generation-output contract tests.
- [x] Isolate progress-handle creation and lifecycle into a shared helper.
- [x] Split final write-back dispatch in extract path into a dedicated helper with contract tests.
- [x] Isolate `_plan_resume_for_usr(...)` behind `resume_planner.py` with parity coverage.
- [x] Separate extract chunk write-back callback construction into `extract_chunk_writeback.py`.
- [x] Isolate adapter runtime/cache policy into `adapter_runtime.py`.
- [ ] Next slice candidate: align infer docs information architecture with sibling package pattern (light top README + workflow/docs index + explicit pressure-test demo route) and add docs/wrapper contract tests.

## 2026-03-06 - Phase 1 Slice M (Docs IA Parity + Wrapper Contract Hardening)

### Scope

- Objective: align infer docs layout with sibling package pattern:
  - lightweight top README,
  - workflow-first docs entry,
  - by-type docs index,
  - explicit end-to-end pressure-test demo route.
- Objective: add wrapper correctness checks for module entrypoint and public API exports.

### Prioritized Findings

1. Medium: infer top README remained monolithic compared with sibling package router pattern.
2. Medium: infer docs lacked an explicit `docs/index.md` by-type index and dedicated tutorial route for end-to-end pressure testing.
3. Low-Medium: wrapper correctness (`python -m dnadesign.infer`, public API exports) lacked explicit contract tests.

### Boundary and Contract Decisions

- Replaced `src/dnadesign/infer/README.md` with a lightweight router containing:
  - documentation map,
  - entrypoint contract,
  - boundary reminder with USR namespaced write-back contract.
- Reworked docs IA to progressive disclosure:
  - `src/dnadesign/infer/docs/README.md` as workflow-first map,
  - new `src/dnadesign/infer/docs/index.md` as by-type index,
  - new getting-started section (`docs/getting-started/README.md`, `cli-quickstart.md`),
  - new tutorial route (`docs/tutorials/demo_pressure_test_usr_ops_notify.md`),
  - updated operations index to link both runbook and tutorial.
- Added wrapper correctness tests:
  - module entrypoint existence,
  - public API callable export contracts,
  - `python -m dnadesign.infer --help` execution contract.

### TDD Evidence

- Red:
  - added `src/dnadesign/infer/tests/test_docs_information_architecture_contracts.py`
  - added `src/dnadesign/infer/tests/test_wrapper_contracts.py`
  - initial run failed on missing docs IA files/sections and missing tutorial routes.
- Green:
  - implemented docs IA files/links and README reduction
  - added tutorial and getting-started surfaces
  - verification passed for new docs/wrapper contracts and existing infer suites.

### Verification Commands (Executed)

- `uv run pytest -q src/dnadesign/infer/tests/test_docs_information_architecture_contracts.py src/dnadesign/infer/tests/test_wrapper_contracts.py`
- `uv run pytest -q src/dnadesign/infer/tests/test_infer_usr_docs_contract.py src/dnadesign/infer/tests/test_pressure_runbook_docs_contract.py src/dnadesign/infer/tests/test_adapter_runtime.py`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run infer extract --preset evo2/extract_logits_ll --seq ACGT --dry-run`
- `uv run infer generate --model-id evo2_7b --device cpu --precision bf16 --alphabet dna --prompt ACGT --max-new-tokens 4 --dry-run`
- `uv run python -m dnadesign.infer --help`

### Information Architecture Update

- Added/updated docs surfaces:
  - `src/dnadesign/infer/README.md`
  - `src/dnadesign/infer/docs/README.md`
  - `src/dnadesign/infer/docs/index.md`
  - `src/dnadesign/infer/docs/getting-started/README.md`
  - `src/dnadesign/infer/docs/getting-started/cli-quickstart.md`
  - `src/dnadesign/infer/docs/tutorials/README.md`
  - `src/dnadesign/infer/docs/tutorials/demo_pressure_test_usr_ops_notify.md`
  - `src/dnadesign/infer/docs/operations/README.md`
  - `src/dnadesign/infer/docs/reference/README.md`
  - `src/dnadesign/infer/docs/reference/command-contracts.md`

### Task Board

- [x] Add fail-fast namespace contract hardening for extract/generate runtime dispatch.
- [x] Add adversarial namespace pressure tests.
- [x] Add infer pressure-test operations runbook docs and config example.
- [x] Extract adapter-dispatch block from `engine.py` into dedicated module with invariant tests.
- [x] Split ingest loading branch from `run_extract_job` into a focused helper with parity tests.
- [x] Split generate ingest loading branch from `run_generate_job` into a focused helper with parity tests.
- [x] Isolate extract micro-batch/derating execution loop behind a helper with chunk-contract tests.
- [x] Isolate generate micro-batch execution loop behind a helper with generation-output contract tests.
- [x] Isolate progress-handle creation and lifecycle into a shared helper.
- [x] Split final write-back dispatch in extract path into a dedicated helper with contract tests.
- [x] Isolate `_plan_resume_for_usr(...)` behind `resume_planner.py` with parity coverage.
- [x] Separate extract chunk write-back callback construction into `extract_chunk_writeback.py`.
- [x] Isolate adapter runtime/cache policy into `adapter_runtime.py`.
- [x] Align infer docs IA to sibling pattern with lightweight README + workflow/type indexes + explicit pressure-test demo route.
- [x] Add infer wrapper correctness contracts for module entrypoint and public API exports.
- [ ] Next slice candidate: split extract/generate batch-policy parsing (`DNADESIGN_INFER_BATCH`, `DNADESIGN_INFER_DEFAULT_BS`) into a dedicated runtime policy module with invariant tests.

## 2026-03-06 - Phase 1 Slice N (Batch Policy Runtime Module + Profiling Cycle)

### Scope

- Objective: extract batch policy parsing from `engine.py` into a dedicated runtime-policy module.
- Objective: run a measurement-first optimization cycle on extract execution loop and keep only measured wins.

### Prioritized Findings

1. Low-Medium: extract/generate duplicated environment parsing for batch policy in `engine.py`.
2. Medium (performance): extract execution loop dominated cumulative runtime in synthetic profiling baseline.

### Boundary and Contract Decisions

- Added `src/dnadesign/infer/batch_policy.py` with:
  - `resolve_micro_batch_size(...)`
  - `resolve_default_extract_batch_size()`
  - `resolve_extract_batch_policy(...)`
- Rewired `engine.py` extract/generate paths to consume batch policy helpers.
- Preserved fail-fast behavior on invalid integer environment values (ValueError propagation).

### TDD Evidence

- Red:
  - added `src/dnadesign/infer/tests/test_batch_policy.py`
  - initial run failed with:
    - `ModuleNotFoundError: No module named 'dnadesign.infer.batch_policy'`
- Green:
  - implemented `batch_policy.py`
  - rewired `engine.py` batch policy resolution
  - all targeted and full infer tests passed.

### Performance Optimization Cycle (Measurement-first)

- Baseline workload:
  - `execute_extract_output` synthetic run (`N=120000`, `micro_bs=64`, five runs).
  - baseline mean: `0.008402s` (stdev `0.001635s`).
  - cProfile hotspot: `extract_execution.execute_extract_output` dominated cumulative time.
- Hypothesis tested:
  - replace per-batch indexed sequence gather with contiguous fast path / precompute variants.
- Outcome:
  - post-change means worsened (`0.009236s` and `0.011089s` in successive attempts).
  - decision: **revert optimization changes** and keep original execution logic.
- Recommendation:
  - continue investigation with larger realistic adapter-bound workloads; current micro-optimization did not produce validated gain.

### Verification Commands (Executed)

- `uv run pytest -q src/dnadesign/infer/tests/test_batch_policy.py`
- `uv run pytest -q src/dnadesign/infer/tests/test_batch_policy.py src/dnadesign/infer/tests/test_extract_execution.py src/dnadesign/infer/tests/test_namespace_contracts.py src/dnadesign/infer/tests/test_usr_writeback_contract.py`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run infer extract --preset evo2/extract_logits_ll --seq ACGT --dry-run`
- `uv run infer generate --model-id evo2_7b --device cpu --precision bf16 --alphabet dna --prompt ACGT --max-new-tokens 4 --dry-run`
- profiling harness (baseline and after-change):
  - `uv run python - <<'PY' ... execute_extract_output benchmark + cProfile ... PY`

### Information Architecture Update

- Runtime map updated in `src/dnadesign/infer/docs/architecture/README.md` with:
  - `batch_policy.py` as runtime policy boundary.

### Task Board

- [x] Add fail-fast namespace contract hardening for extract/generate runtime dispatch.
- [x] Add adversarial namespace pressure tests.
- [x] Add infer pressure-test operations runbook docs and config example.
- [x] Extract adapter-dispatch block from `engine.py` into dedicated module with invariant tests.
- [x] Split ingest loading branch from `run_extract_job` into a focused helper with parity tests.
- [x] Split generate ingest loading branch from `run_generate_job` into a focused helper with parity tests.
- [x] Isolate extract micro-batch/derating execution loop behind a helper with chunk-contract tests.
- [x] Isolate generate micro-batch execution loop behind a helper with generation-output contract tests.
- [x] Isolate progress-handle creation and lifecycle into a shared helper.
- [x] Split final write-back dispatch in extract path into a dedicated helper with contract tests.
- [x] Isolate `_plan_resume_for_usr(...)` behind `resume_planner.py` with parity coverage.
- [x] Separate extract chunk write-back callback construction into `extract_chunk_writeback.py`.
- [x] Isolate adapter runtime/cache policy into `adapter_runtime.py`.
- [x] Align infer docs IA to sibling pattern with lightweight README + workflow/type indexes + explicit pressure-test demo route.
- [x] Add infer wrapper correctness contracts for module entrypoint and public API exports.
- [x] Split extract/generate batch-policy parsing into `batch_policy.py`.
- [x] Run measurement-first extract-loop optimization cycle and revert non-improving variants.
- [ ] Next slice candidate: isolate progress-manager construction and per-job execution envelope from CLI command handlers into a dedicated runtime helper to reduce CLI monolith size.

## 2026-03-06 - Phase 1 Slice O (Typed Ingest Payload + CLI Helper Boundary)

### Scope

- Objective: reduce change fragility in `engine.py` by replacing the untyped ingest 5-tuple with a typed payload contract.
- Objective: reduce CLI monolith duplication by extracting shared model/progress assembly into a dedicated helper module.

### Boundary and Contract Decisions

- Added `ExtractIngestPayload` dataclass in `src/dnadesign/infer/engine.py` and updated engine call sites to consume named fields (`seqs`, `ids`, `records`, `pt_path`, `dataset`).
- Preserved ingest source behavior and fail-fast semantics for unknown sources and invalid pt-file inputs.
- Added `src/dnadesign/infer/cli_builders.py` with:
  - `build_model_config(...)`
  - `run_with_progress(...)`
- Rewired `run`, `extract`, and `generate` command handlers in `src/dnadesign/infer/cli.py` to use the shared helper boundary while preserving command surface and output semantics.

### TDD Evidence

- Red:
  - Added `test_load_extract_ingest_returns_payload_object` and confirmed failure against tuple contract:
    - `AttributeError: 'tuple' object has no attribute 'seqs'`
  - Added `src/dnadesign/infer/tests/test_cli_builders.py` and confirmed initial failure:
    - `ModuleNotFoundError: No module named 'dnadesign.infer.cli_builders'`
- Green:
  - Implemented payload dataclass + CLI helper module and rewired call sites.
  - All targeted and full infer tests passed.

### Verification Commands (Executed)

- `uv run pytest -q src/dnadesign/infer/tests/test_extract_ingest_helper.py -k returns_payload_object`
- `uv run pytest -q src/dnadesign/infer/tests/test_extract_ingest_helper.py src/dnadesign/infer/tests/test_generate_ingest_helper.py src/dnadesign/infer/tests/test_namespace_contracts.py src/dnadesign/infer/tests/test_usr_writeback_contract.py`
- `uv run pytest -q src/dnadesign/infer/tests/test_cli_builders.py`
- `uv run pytest -q src/dnadesign/infer/tests/test_cli_builders.py src/dnadesign/infer/tests/test_presets.py src/dnadesign/infer/tests/test_extract_ingest_helper.py src/dnadesign/infer/tests/test_generate_ingest_helper.py src/dnadesign/infer/tests/test_namespace_contracts.py src/dnadesign/infer/tests/test_wrapper_contracts.py`
- `uv run pytest -q src/dnadesign/notify/tests/test_events_source.py src/dnadesign/notify/tests/test_workspace_source.py src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py -k "infer"`
- `uv run pytest -q src/dnadesign/infer/tests`

### Task Board

- [x] Replace untyped ingest tuple in `engine.py` with typed payload contract.
- [x] Extract shared CLI model/progress assembly into dedicated helper module.
- [x] Rewire `run`/`extract`/`generate` handlers to use shared helper boundary.
- [x] Verify no regressions in infer tests and infer-linked notify/usr contracts.
- [ ] Next slice candidate: split command-specific request assembly from Typer entrypoints into `cli_commands/` modules to further reduce `cli.py` size without changing CLI behavior.

## 2026-03-06 - Phase 1 Slice P (CLI Ingest Builder Boundary)

### Scope

- Objective: reduce `cli.py` command-body branching by extracting ingest source selection and input materialization into a focused module.

### Boundary and Contract Decisions

- Added `src/dnadesign/infer/cli_ingest.py` with explicit request builders:
  - `build_extract_ingest(...)`
  - `build_generate_ingest(...)`
  - `CliIngestRequest` dataclass (`ingest`, `inputs`)
- Rewired `extract` and `generate` command handlers in `src/dnadesign/infer/cli.py` to delegate ingest-source assembly to `cli_ingest.py`.
- Preserved source precedence and fail-fast contracts:
  - extract precedence: `usr` > `pt` > `records_jsonl` > `seq_file` > `seq`
  - generate precedence: `usr` > `prompt_file` > `prompt`
  - unchanged explicit errors when no valid source is provided.

### TDD Evidence

- Red:
  - Added `src/dnadesign/infer/tests/test_cli_ingest.py` and confirmed initial failure:
    - `ModuleNotFoundError: No module named 'dnadesign.infer.cli_ingest'`
- Green:
  - Implemented `cli_ingest.py` and rewired `cli.py` call sites.
  - New ingest tests and full infer suite passed.

### Verification Commands (Executed)

- `uv run pytest -q src/dnadesign/infer/tests/test_cli_ingest.py`
- `uv run pytest -q src/dnadesign/infer/tests/test_cli_ingest.py src/dnadesign/infer/tests/test_cli_builders.py src/dnadesign/infer/tests/test_presets.py src/dnadesign/infer/tests/test_wrapper_contracts.py`
- `uv run pytest -q src/dnadesign/infer/tests src/dnadesign/notify/tests/test_events_source.py src/dnadesign/notify/tests/test_workspace_source.py src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py -k "infer"`

### Information Architecture Update

- Updated `src/dnadesign/infer/docs/architecture/README.md` interface-layer map to include `cli_ingest.py`.

### Task Board

- [x] Extract CLI ingest source branching into dedicated module.
- [x] Keep command behavior stable with characterization tests.
- [x] Verify infer + infer-linked notify/usr tests remain green.
- [ ] Next slice candidate: move command-specific request assembly (`run` preset job, extract output spec params, generate params) into `cli_commands/` modules and keep Typer entrypoints declarative.

## 2026-03-06 - Phase 1 Slice Q (CLI Request Assembly Boundary)

### Scope

- Objective: reduce extract/generate command complexity in `cli.py` by moving request assembly logic (model + job + params/outputs) into a dedicated module.

### Boundary and Contract Decisions

- Added `src/dnadesign/infer/cli_requests.py` with:
  - `build_extract_request(...)`
  - `build_generate_request(...)`
  - dataclasses `ExtractRequest` and `GenerateRequest`
- Rewired `extract` and `generate` command handlers in `src/dnadesign/infer/cli.py` to consume the request builders.
- Preserved existing fail-fast validation contracts:
  - extract still requires `--fn` + `--format` when `--preset` absent
  - generate preset-kind mismatch still fails with `ConfigError`
  - default generate params remain `max_new_tokens=64`, `temperature=1.0`.

### TDD Evidence

- Red:
  - Added `src/dnadesign/infer/tests/test_cli_requests.py` and confirmed initial failure:
    - `ModuleNotFoundError: No module named 'dnadesign.infer.cli_requests'`
- Green:
  - Implemented `cli_requests.py` and rewired `cli.py` call sites.
  - New request tests and full infer suite passed.

### Verification Commands (Executed)

- `uv run pytest -q src/dnadesign/infer/tests/test_cli_requests.py`
- `uv run pytest -q src/dnadesign/infer/tests/test_cli_requests.py src/dnadesign/infer/tests/test_cli_ingest.py src/dnadesign/infer/tests/test_cli_builders.py src/dnadesign/infer/tests/test_presets.py src/dnadesign/infer/tests/test_wrapper_contracts.py`
- `uv run pytest -q src/dnadesign/infer/tests src/dnadesign/notify/tests/test_events_source.py src/dnadesign/notify/tests/test_workspace_source.py src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py -k "infer"`

### Information Architecture Update

- Updated `src/dnadesign/infer/docs/architecture/README.md` interface-layer map to include `cli_requests.py`.

### Task Board

- [x] Extract request assembly for extract/generate into dedicated module.
- [x] Keep dry-run output rendering contracts stable through request return shape.
- [x] Verify infer + infer-linked notify/usr tests remain green.
- [ ] Next slice candidate: split Typer command entrypoints into `cli_commands/` package (`run.py`, `extract.py`, `generate.py`) with one function per command and shared exception envelope.

## 2026-03-06 - Phase 2 Slice B (Top-Level Sprawl Consolidation)

### Goal

Reduce `infer` top-level module sprawl and align package shape with sibling progressive-disclosure layout: `README.md`, `docs/`, `src/`, `tests/`.

### TDD record

1. Added failing IA contract test: `src/dnadesign/infer/tests/test_source_tree_contracts.py`.
2. Confirmed red state:
   - `uv run pytest -q src/dnadesign/infer/tests/test_source_tree_contracts.py`
3. Implemented structural consolidation.
4. Re-ran tests to green.

### Implemented changes

- Moved internal implementation modules under `src/dnadesign/infer/src/`:
  - interface/runtime modules (`cli.py`, `engine.py`, `config.py`, etc.)
  - boundaries/extensions subpackages (`adapters/`, `ingest/`, `presets/`, `writers/`)
- Kept only thin package entrypoints at top level:
  - `__init__.py`, `__main__.py`, `cli.py`
- Updated package exports and side-effect registration wiring:
  - `dnadesign.infer.__init__` now imports from `dnadesign.infer.src.*`
- Updated tests and monkeypatch targets to `dnadesign.infer.src.*` for internal modules.
- Updated preset package resource path to `dnadesign.infer.src.presets`.
- Added internal source-tree guide:
  - `src/dnadesign/infer/src/README.md`
- Updated docs architecture and index routes to reflect `infer/src/` layout.

### Verification evidence

- `uv run pytest -q src/dnadesign/infer/tests/test_source_tree_contracts.py src/dnadesign/infer/tests/test_wrapper_contracts.py src/dnadesign/infer/tests/test_docs_information_architecture_contracts.py`
- `uv run pytest -q src/dnadesign/infer/tests src/dnadesign/notify/tests/test_events_source.py src/dnadesign/notify/tests/test_workspace_source.py src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py -k "infer"`
- `uv run infer --help`

### Notes / remaining opportunities

- `activations_debug.log` still exists at infer root and should be moved to workspace-local logs or removed by explicit hygiene policy.
- Next decoupling slice should split `src/cli.py` command handlers into `src/cli_commands/` to reduce command-path coupling while preserving CLI contracts.
- Additional verification rerun (unfiltered infer suite):
  - `uv run pytest -q src/dnadesign/infer/tests`
  - `uv run pytest -q src/dnadesign/notify/tests/test_events_source.py src/dnadesign/notify/tests/test_workspace_source.py src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py`

## 2026-03-06 - Phase 2 Slice C (CLI Package + Test Area Modularization)

### Goal

Reduce internal `infer/src` flatness by introducing a dedicated `src/cli/` package and align infer test layout with sibling package patterns (area-based folders).

### TDD record

1. Extended IA contract test with two new assertions:
   - `src/cli/` package must exist with `app.py`, `console.py`, `builders.py`, `ingest.py`, `requests.py`.
   - `tests/` must be grouped by area (`cli`, `runtime`, `contracts`, `docs`, `package`).
2. Confirmed red state:
   - `uv run pytest -q src/dnadesign/infer/tests/package/test_source_tree_contracts.py`
3. Implemented structural moves and import rewiring.
4. Re-ran targeted and full suites to green.

### Implemented changes

- Internal CLI modularization:
  - moved `src/cli.py` -> `src/cli/app.py`
  - moved `src/_console.py` -> `src/cli/console.py`
  - moved `src/cli_builders.py` -> `src/cli/builders.py`
  - moved `src/cli_ingest.py` -> `src/cli/ingest.py`
  - moved `src/cli_requests.py` -> `src/cli/requests.py`
  - added `src/cli/__init__.py` as CLI package surface
- Fixed config discovery to continue honoring package-local fallback config at `infer/config.yaml` after CLI package move.
- Test modularization:
  - moved infer tests under area folders: `tests/cli`, `tests/runtime`, `tests/contracts`, `tests/docs`, `tests/package`.
- Documentation updates:
  - updated architecture map to represent `src/cli/*` package shape.
  - updated `src/README.md` internal source map for CLI package boundary.

### Verification evidence

- `uv run pytest -q src/dnadesign/infer/tests/package/test_source_tree_contracts.py src/dnadesign/infer/tests/cli/test_builders.py src/dnadesign/infer/tests/cli/test_ingest.py src/dnadesign/infer/tests/cli/test_requests.py src/dnadesign/infer/tests/package/test_wrapper_contracts.py`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run pytest -q src/dnadesign/notify/tests/test_events_source.py src/dnadesign/notify/tests/test_workspace_source.py src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py`
- `uv run infer --help`
- `uv run python -m dnadesign.infer --help`

### Notes / next opportunities

- `src/cli/app.py` remains a large command file; next slice should split command handlers by command group into `src/cli/commands/` while preserving current CLI signatures and exit-code behavior.
- `activations_debug.log` remains at infer package root and should be moved under workspace-local logs by explicit policy.
- Added infer CLI wrapper `main()` in `src/dnadesign/infer/cli.py` and added package wrapper contract test to keep a callable entrypoint alias available alongside `app`.

## 2026-03-06 - Phase 2 Slice D (CLI Command Group Split)

### Goal

Further reduce CLI monolith coupling by splitting `src/cli/app.py` command bodies into command-group modules under `src/cli/commands/` while preserving command signatures and exit semantics.

### TDD record

1. Added failing IA contract for command-group split:
   - `test_infer_cli_commands_are_split_by_group`
2. Confirmed red state:
   - `uv run pytest -q src/dnadesign/infer/tests/package/test_source_tree_contracts.py::test_infer_cli_commands_are_split_by_group`
3. Implemented command-group extraction and app wiring reduction.
4. Re-ran targeted and full suites to green.

### Implemented changes

- Added shared CLI helper module:
  - `src/dnadesign/infer/src/cli/common.py` (`exit_for`, `raise_cli_error`, `discovery_config`, `guard_pickle`).
- Added command-group modules:
  - `src/dnadesign/infer/src/cli/commands/run.py`
  - `src/dnadesign/infer/src/cli/commands/extract.py`
  - `src/dnadesign/infer/src/cli/commands/generate.py`
  - `src/dnadesign/infer/src/cli/commands/presets.py`
  - `src/dnadesign/infer/src/cli/commands/adapters.py`
  - `src/dnadesign/infer/src/cli/commands/validate.py`
  - `src/dnadesign/infer/src/cli/commands/__init__.py` (`register_all`).
- Reduced `src/dnadesign/infer/src/cli/app.py` to:
  - Typer app creation
  - root callback (logging/trace flags)
  - command-group registration call.

### Verification evidence

- `uv run pytest -q src/dnadesign/infer/tests/package/test_source_tree_contracts.py src/dnadesign/infer/tests/cli src/dnadesign/infer/tests/package/test_wrapper_contracts.py`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run pytest -q src/dnadesign/notify/tests/test_events_source.py src/dnadesign/notify/tests/test_workspace_source.py src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py`
- `uv run infer --help`
- `uv run python -m dnadesign.infer --help`

### Notes / next opportunities

- Runtime core is still relatively flat (`adapter_*`, `*execution`, `writeback_*` at one level); next IA slice should evaluate introducing `src/runtime/` subpackages without changing behavior.
- `activations_debug.log` remains at package root and should be addressed by explicit hygiene policy.

## 2026-03-06 - Phase 2 Slice E (Runtime Package + Log Hygiene)

### Goal

Apply next two IA quality increments:
1. group runtime modules under `src/runtime/`
2. enforce root log-artifact hygiene (`no tracked *.log` at infer package root).

### TDD record

1. Added failing IA contracts:
   - `test_infer_runtime_modules_are_grouped_under_runtime_package`
   - `test_infer_root_does_not_track_runtime_log_artifacts`
2. Confirmed red state:
   - `uv run pytest -q src/dnadesign/infer/tests/package/test_source_tree_contracts.py::test_infer_runtime_modules_are_grouped_under_runtime_package src/dnadesign/infer/tests/package/test_source_tree_contracts.py::test_infer_root_does_not_track_runtime_log_artifacts`
3. Implemented runtime package move + import rewiring + log artifact removal.
4. Re-ran targeted and full suites to green.

### Implemented changes

- Runtime package created:
  - `src/dnadesign/infer/src/runtime/__init__.py`
- Moved runtime modules into `src/runtime/`:
  - `adapter_dispatch.py`, `adapter_runtime.py`, `batch_policy.py`, `extract_chunk_writeback.py`, `extract_execution.py`, `generate_execution.py`, `progress.py`, `resume_planner.py`, `writeback_dispatch.py`
- Updated import graph:
  - `engine.py` now imports runtime modules from `src.runtime.*`
  - CLI builders now import `ProgressFactory` from `src.runtime.progress`
  - runtime tests now import and monkeypatch `dnadesign.infer.src.runtime.*`
- Removed tracked root log artifact:
  - deleted `src/dnadesign/infer/activations_debug.log`
- Updated architecture/docs maps for runtime package shape.

### Verification evidence

- `uv run pytest -q src/dnadesign/infer/tests/package/test_source_tree_contracts.py src/dnadesign/infer/tests/runtime src/dnadesign/infer/tests/contracts src/dnadesign/infer/tests/cli src/dnadesign/infer/tests/package/test_wrapper_contracts.py`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run pytest -q src/dnadesign/notify/tests/test_events_source.py src/dnadesign/notify/tests/test_workspace_source.py src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py`
- `uv run infer --help`
- `uv run python -m dnadesign.infer --help`

### Notes / next opportunities

- `engine.py` still centralizes orchestration + ingest/load concerns and is the next decoupling hotspot.
- Logging setup remains duplicated between `src/_logging.py` and `src/cli/console.py`; a single policy owner module would reduce drift risk.

## 2026-03-06 - Phase 2 Slice F (Engine Ingest Boundary + Shared Logging Policy)

### Goal

Apply the next two decoupling increments:
1. move ingest loading contracts out of `engine.py` into a dedicated runtime module.
2. establish one logging policy owner used by CLI and runtime paths.

### TDD record

1. Added failing tests:
   - source-tree contract update requiring `src/runtime/ingest_loading.py`.
   - ingest-helper tests switched to `dnadesign.infer.src.runtime.ingest_loading`.
   - new CLI test `test_console_setup_logging_uses_shared_policy_owner`.
2. Confirmed red state:
   - `uv run pytest -q src/dnadesign/infer/tests/package/test_source_tree_contracts.py src/dnadesign/infer/tests/runtime/test_extract_ingest_helper.py src/dnadesign/infer/tests/runtime/test_generate_ingest_helper.py src/dnadesign/infer/tests/cli/test_console.py`
   - initial failure: `ModuleNotFoundError` for `dnadesign.infer.src.runtime.ingest_loading`.
3. Implemented runtime ingest module extraction + shared logging delegation.
4. Re-ran targeted and full suites to green.

### Implemented changes

- Added runtime ingest boundary:
  - `src/dnadesign/infer/src/runtime/ingest_loading.py`
  - `ExtractIngestPayload`, `load_extract_ingest(...)`, `load_generate_ingest(...)`
- Rewired engine orchestration:
  - `src/dnadesign/infer/src/engine.py` now imports ingest-loading helpers from `src/runtime/ingest_loading.py`.
  - removed ingest-source loader branching from `engine.py`.
- Consolidated logging setup ownership:
  - `src/dnadesign/infer/src/_logging.py` now accepts optional `rich_console` and owns root logging setup policy.
  - `src/dnadesign/infer/src/cli/console.py` now delegates `setup_console_logging(...)` to shared logging policy.
- Updated docs maps:
  - `src/dnadesign/infer/docs/architecture/README.md`
  - `src/dnadesign/infer/src/README.md`

### Verification evidence

- `uv run pytest -q src/dnadesign/infer/tests/package/test_source_tree_contracts.py src/dnadesign/infer/tests/runtime/test_extract_ingest_helper.py src/dnadesign/infer/tests/runtime/test_generate_ingest_helper.py src/dnadesign/infer/tests/cli/test_console.py`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run pytest -q src/dnadesign/notify/tests/test_events_source.py src/dnadesign/notify/tests/test_workspace_source.py src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py`
- `uv run infer --help`
- `uv run python -m dnadesign.infer --help`

### Notes / next opportunities

- `engine.py` remains the main orchestration surface; further decomposition should preserve explicit contracts and keep API signatures stable.
- Runtime module grouping can continue if additional hotspots emerge, but current module seams are now explicit for ingest and logging policy ownership.

## 2026-03-06 - Phase 2 Slice G (Maintainer Audit Hardening)

### Goal

Run an end-to-end maintainer audit for infer shipping readiness, then harden highest-impact correctness and UX footguns with adversarial coverage.

### Baseline evidence

- Baseline suites:
  - `uv run pytest -q src/dnadesign/infer/tests`
  - `uv run pytest -q src/dnadesign/notify/tests/test_events_source.py src/dnadesign/notify/tests/test_workspace_source.py src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py`
- Primary CLI usage-path checks:
  - `uv run infer validate config --config src/dnadesign/infer/config.yaml`
  - `uv run infer extract --preset evo2/extract_logits_ll --seq ACGT --dry-run`
- Baseline adversarial probe before hardening:
  - unknown model key accepted by `validate config` (`/tmp/infer_bad_extra.yaml`) with exit code `0` (unexpected permissive behavior).

### Prioritized findings

1. High: config schema accepted unknown keys (silent drift risk).
   - Evidence: `validate config` accepted `model.typo_field` with exit code `0`.
   - Impact: typoed or stale fields silently ignored, producing hidden misconfiguration.
2. High: USR write-back overwrite contract was ambiguous.
   - Evidence: `writers/usr.py` accepted `overwrite` parameter but attach semantics required `allow_overwrite=True` for chunk append safety.
   - Impact: direct write-back flows could accidentally replace existing values without explicit guard behavior.
3. Medium: CLI exit-code ergonomics for schema validation.
   - Evidence: Pydantic schema failures were not mapped to config exit code by contract.
   - Impact: operators saw generic non-specific failure code for config validation errors.

### TDD and implementation

1. Added failing tests:
   - `src/dnadesign/infer/tests/cli/test_validate_command.py`
     - unknown fields must fail with exit `2`
     - wrong type must fail with exit `2`
   - `src/dnadesign/infer/tests/contracts/test_usr_writeback_contract.py`
     - `overwrite=false` must fail fast when attempting to replace existing infer values.
2. Implemented fixes:
   - strict config parsing:
     - `src/dnadesign/infer/src/config.py`
     - introduced `StrictConfigModel` (`extra="forbid"`) for root and nested schema models.
   - CLI config error mapping:
     - `src/dnadesign/infer/src/cli/common.py`
     - map `pydantic.ValidationError` to config exit code (`2`).
   - write-back overwrite guard with append-safe chunk semantics:
     - `src/dnadesign/infer/src/writers/usr.py`
     - keep `allow_overwrite=True` for chunk append behavior, but reject value replacement when `overwrite=false` by scanning existing infer overlay ids/columns.
3. Updated docs:
   - `src/dnadesign/infer/docs/reference/command-contracts.md`
   - documented strict validate-config behavior and overwrite=false replacement guard.

### Adversarial pressure-test outcomes

- Schema mutation:
  - unknown model key (`model.typo_field`) now fails with explicit schema error and exit code `2`.
  - wrong type (`model.batch_size: not_an_int`) now fails with explicit schema error and exit code `2`.
- State/flow abuse:
  - repeated write-back on same id/column with `overwrite=false` now fails fast with `WriteBackError`.
  - chunked append on non-overlapping ids remains functional and resume-safe.

### Verification evidence

- `uv run pytest -q src/dnadesign/infer/tests/cli/test_validate_command.py src/dnadesign/infer/tests/contracts/test_usr_writeback_contract.py`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run pytest -q src/dnadesign/notify/tests/test_events_source.py src/dnadesign/notify/tests/test_workspace_source.py src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py`
- `uv run infer validate config --config src/dnadesign/infer/config.yaml`
- `uv run infer extract --preset evo2/extract_logits_ll --seq ACGT --dry-run`
- `uv run infer validate config --config /tmp/infer_bad_extra.yaml` (now explicit failure, exit `2`)

### Notes / next opportunities

- Full positive-path runtime extraction/generation still depends on model/runtime availability; current hardening covered schema/CLI contracts, overlay safety, and dry-run/runtime-invariant surfaces.

## 2026-03-06 - Phase 2 Slice H (Explicit Registry Bootstrap + Config Discovery Hardening)

### Goal

Reduce implicit coupling in adapter registration and remove hidden config fallback behavior to keep infer fail-fast and easier to reason about.

### TDD record

1. Added failing tests:
   - `src/dnadesign/infer/tests/package/test_registry_bootstrap_contracts.py`
   - `src/dnadesign/infer/tests/cli/test_adapters_commands.py`
   - `src/dnadesign/infer/tests/cli/test_validate_command.py::test_validate_config_requires_explicit_path_or_cwd_config`
2. Confirmed red state:
   - `uv run pytest -q src/dnadesign/infer/tests/package/test_registry_bootstrap_contracts.py src/dnadesign/infer/tests/cli/test_validate_command.py src/dnadesign/infer/tests/cli/test_adapters_commands.py`
3. Implemented explicit bootstrap boundary and config discovery hardening.
4. Re-ran targeted and full suites to green.

### Implemented changes

- Added explicit registry bootstrap module:
  - `src/dnadesign/infer/src/bootstrap.py` (`initialize_registry()` idempotent contract).
- Refactored adapter registration to explicit function:
  - `src/dnadesign/infer/src/adapters/__init__.py` now exposes `register_defaults()` instead of import-time registration.
- Removed package import side-effect registration:
  - `src/dnadesign/infer/__init__.py` no longer imports adapters for registration side effects.
- Wired explicit bootstrap at runtime boundaries:
  - CLI root callback (`src/dnadesign/infer/src/cli/app.py`)
  - API entrypoints (`src/dnadesign/infer/src/api.py`)
  - Output function validation (`src/dnadesign/infer/src/config.py`)
- Hardened config discovery:
  - `src/dnadesign/infer/src/cli/common.py` now allows only `--config` or cwd `config.yaml` (no module-local fallback path).
- Added IA contract guard:
  - `src/dnadesign/infer/tests/package/test_source_tree_contracts.py` now requires `src/bootstrap.py`.
- Updated docs:
  - `src/dnadesign/infer/docs/architecture/README.md`
  - `src/dnadesign/infer/src/README.md`
  - `src/dnadesign/infer/docs/reference/command-contracts.md`

### Verification evidence

- `uv run pytest -q src/dnadesign/infer/tests/package/test_registry_bootstrap_contracts.py src/dnadesign/infer/tests/cli/test_validate_command.py src/dnadesign/infer/tests/cli/test_adapters_commands.py`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run pytest -q src/dnadesign/notify/tests/test_events_source.py src/dnadesign/notify/tests/test_workspace_source.py src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py`
- `uv run infer validate config --config src/dnadesign/infer/docs/operations/examples/pressure_test_infer_config.yaml`
- `uv run infer adapters list`

### Notes / next opportunities

- `src/dnadesign/infer/src/engine.py` remains the highest coupling point and next candidate for further orchestration decomposition.
- A dedicated `infer validate runtime` preflight command could expose a deterministic operator check bundle for scheduler workflows.

## 2026-03-06 - Phase 2 Slice I (Workspace Harness + Scaffold Contracts)

### Goal

Add an explicit infer workspace harness path so pressure-test runs are easier to start, fail-fast, and consistent with sibling package IA patterns.

### TDD record

1. Added failing tests:
   - `src/dnadesign/infer/tests/cli/test_workspace_command.py`
   - `src/dnadesign/infer/tests/package/test_source_tree_contracts.py` (workspace scaffold assertions)
2. Confirmed red state:
   - `uv run pytest -q src/dnadesign/infer/tests/cli/test_workspace_command.py src/dnadesign/infer/tests/package/test_source_tree_contracts.py`
3. Implemented workspace resolution/scaffold module + CLI command group + docs/workspaces scaffold.
4. Re-ran targeted and full suites to green.

### Implemented changes

- Added workspace contract module:
  - `src/dnadesign/infer/src/workspace.py`
  - explicit contracts:
    - `--root` override, then `INFER_WORKSPACE_ROOT`, then repo-default root.
    - fail-fast workspace id validation (no path-like ids).
    - fail-fast template existence checks.
- Added CLI command group:
  - `src/dnadesign/infer/src/cli/commands/workspace.py`
  - `infer workspace where`
  - `infer workspace init --id <name> [--root] [--template]`
- Registered command group:
  - `src/dnadesign/infer/src/cli/commands/__init__.py`
- Added infer workspace scaffold directory:
  - `src/dnadesign/infer/workspaces/README.md`
- Updated IA/docs references:
  - `src/dnadesign/infer/README.md`
  - `src/dnadesign/infer/docs/README.md`
  - `src/dnadesign/infer/docs/index.md`
  - `src/dnadesign/infer/docs/architecture/README.md`
  - `src/dnadesign/infer/src/README.md`
  - `src/dnadesign/infer/docs/reference/command-contracts.md`
  - `src/dnadesign/infer/docs/operations/pressure-test-agnostic-models.md`
  - `src/dnadesign/infer/docs/tutorials/demo_pressure_test_usr_ops_notify.md`
- Added/updated tests:
  - `src/dnadesign/infer/tests/cli/test_workspace_command.py`
  - `src/dnadesign/infer/tests/package/test_source_tree_contracts.py`
  - `src/dnadesign/infer/tests/docs/test_information_architecture_contracts.py`
  - `src/dnadesign/infer/tests/docs/test_pressure_runbook_docs_contract.py`

### Verification evidence

- `uv run pytest -q src/dnadesign/infer/tests/cli/test_workspace_command.py src/dnadesign/infer/tests/package/test_source_tree_contracts.py src/dnadesign/infer/tests/docs/test_information_architecture_contracts.py src/dnadesign/infer/tests/docs/test_pressure_runbook_docs_contract.py`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run pytest -q src/dnadesign/notify/tests/test_events_source.py src/dnadesign/notify/tests/test_workspace_source.py src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py`
- `uv run infer workspace where`
- `uv run infer workspace init --id demo_tmp --root <tmp>`
- `uv run infer validate config --config src/dnadesign/infer/docs/operations/examples/pressure_test_infer_config.yaml`

### Notes / next opportunities

- Infer now has a first-class `workspaces/` root and deterministic init path, matching sibling package structure more closely.
- Next incremental hardening candidate: add `infer validate workspace` to check workspace config + USR accessibility in one deterministic preflight command for scheduler workflows.

## 2026-03-06 - Phase 2 Slice J (Maintainer Audit: Demo Workspace Intent + Legacy Cruft Hardening)

### Goal

Confirm workspace intent for end-to-end pressure-test demos and remove high-signal legacy cruft from active infer source/docs surfaces.

### Audit scope and baseline

- Scope: `src/dnadesign/infer` (`src/`, `docs/`, `tests/`, CLI user path).
- Baseline verification before fixes:
  - `uv run pytest -q src/dnadesign/infer/tests`
  - `uv run pytest -q src/dnadesign/notify/tests/test_events_source.py src/dnadesign/notify/tests/test_workspace_source.py src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py`
- Baseline cruft scan:
  - 35 infer `src/*.py` headers still contained `<dnadesign project>` template token.
  - docs outside `docs/dev/journal.md` were already mostly aligned with new module layout.

### Prioritized findings

1. Medium: source-header template token remained across infer runtime modules.
   - Impact: avoidable maintainability/documentation noise and inconsistent file identity metadata.
2. Medium: no machine-checkable contract prevented legacy flat-module references from reappearing in active docs.
   - Impact: drift risk in operator-facing docs over future increments.
3. Low: workspace intent ambiguity for demo pressure tests needed explicit CLI/docs confirmation.
   - Impact: small UX ambiguity during onboarding.

### TDD record

1. Added failing test:
   - `src/dnadesign/infer/tests/package/test_source_tree_contracts.py::test_infer_src_headers_do_not_use_template_project_placeholder`
2. Added docs guard test:
   - `src/dnadesign/infer/tests/docs/test_information_architecture_contracts.py::test_infer_docs_excluding_journal_avoid_legacy_flat_module_paths`
3. Confirmed red state for header-placeholder contract.
4. Replaced placeholder token in infer `src/*.py` headers and re-verified.

### Changes applied

- Added source-tree hardening contract:
  - no `<dnadesign project>` placeholder token allowed under infer `src/` python files.
- Added docs hardening contract:
  - docs excluding `docs/dev/journal.md` must not reference legacy flat-module/test paths.
- Cleaned infer `src/*.py` header template token to `dnadesign`.
- Confirmed workspace demo intent with real user flow:
  - `infer workspace init --id test_stress_ethanol --root <tmp>`
  - `infer validate config --config <tmp>/test_stress_ethanol/config.yaml`

### Adversarial / pressure evidence

- Workspace id/path abuse checks (existing contracts) remain enforced:
  - path-like workspace ids fail fast.
  - existing workspace directory fails fast.
- Demo workspace scaffold pressure path succeeded and produced valid config + expected folder structure.

### Verification evidence

- `uv run pytest -q src/dnadesign/infer/tests/package/test_source_tree_contracts.py::test_infer_src_headers_do_not_use_template_project_placeholder src/dnadesign/infer/tests/docs/test_information_architecture_contracts.py::test_infer_docs_excluding_journal_avoid_legacy_flat_module_paths`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run pytest -q src/dnadesign/notify/tests/test_events_source.py src/dnadesign/notify/tests/test_workspace_source.py src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py`
- `uv run infer workspace init --id test_stress_ethanol --root <tmp>`
- `uv run infer validate config --config <tmp>/test_stress_ethanol/config.yaml`

### Notes / next opportunities

- Historical journal entries intentionally retain prior file paths as time-stamped evidence; active docs now carry no legacy flat-path references under current contracts.
- Next increment: add `infer validate workspace` for a single command that checks workspace config + USR accessibility + dry-run readiness.

## 2026-03-06 - Phase 2 Slice K (UX Audit: Local Workspace IO + Cruft Pass)

### Goal

Clarify infer UX for workspace-local datasets (not only USR) and harden config-driven IO behavior for `infer run`.

### Audit findings

1. High UX gap: `infer extract`/`infer generate` already supported local workspace files, but `infer run --config` was effectively USR-centric unless inputs were injected programmatically.
2. Medium cruft: `src/ingest/sources.py` module header still carried stale copied description text.
3. Medium cruft-control gap: no automated contract for config-driven local ingest path behavior in `run` command.

### Changes applied

- Added config-driven local ingest resolution module:
  - `src/dnadesign/infer/src/cli/config_inputs.py`
  - supports `ingest.path` for `sequences`, `records`, and `pt_file` in config workflows.
  - relative `ingest.path` values resolve against config directory.
- Updated config schema:
  - `src/dnadesign/infer/src/config.py`
  - added `ingest.path` field.
  - fail-fast guard: `ingest.path` is invalid for `source='usr'`.
- Wired `infer run` to config-driven input resolver:
  - `src/dnadesign/infer/src/cli/commands/run.py`
- Added tests:
  - `src/dnadesign/infer/tests/cli/test_config_inputs.py`
  - `src/dnadesign/infer/tests/cli/test_run_command_config_inputs.py`
  - `src/dnadesign/infer/tests/cli/test_validate_command.py` (USR path rejection contract)
- Docs updates:
  - `src/dnadesign/infer/docs/reference/command-contracts.md`
  - `src/dnadesign/infer/workspaces/README.md`
- Cruft cleanup:
  - `src/dnadesign/infer/src/ingest/sources.py` header description corrected.

### UX contract (current)

- Ad-hoc commands:
  - `extract`: `--seq-file`, `--records-jsonl`, `--pt`, or `--usr`.
  - `generate`: `--prompt-file` or `--usr`.
- Config workflows (`infer run --config`):
  - `ingest.source: usr` via dataset/root fields.
  - `ingest.source: sequences|records|pt_file` via `ingest.path` (workspace-local path supported).

### Verification evidence

- `uv run pytest -q src/dnadesign/infer/tests/cli/test_config_inputs.py src/dnadesign/infer/tests/cli/test_validate_command.py`
- `uv run pytest -q src/dnadesign/infer/tests/cli/test_run_command_config_inputs.py`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run pytest -q src/dnadesign/notify/tests/test_events_source.py src/dnadesign/notify/tests/test_workspace_source.py src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py`

### Notes / next opportunities

- Next hardening step: add `infer validate workspace --config <path>` to combine config, local ingest.path readability, and USR dataset preflight into one deterministic command.
