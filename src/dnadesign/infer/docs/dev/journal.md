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

## 2026-03-06 - Phase 2 Slice L (PR Critique Closure + Run-Mode UX + Root IA Tightening)

### Goal

Address open infer Codex PR critique(s), harden `infer run` mode ergonomics with explicit fail-fast contracts, and remove top-level package cruft that increased IA sprawl.

### Codex critique addressed

- PR thread item: `src/dnadesign/infer/src/writers/usr.py`
  - `_guard_usr_overwrite` failed when an existing infer overlay did not yet contain newly requested output columns.
  - Root cause: Parquet read requested all target columns unconditionally, causing `ArrowInvalid` before column-presence checks.

### TDD record

1. Added failing regression test:
   - `src/dnadesign/infer/tests/contracts/test_usr_writeback_contract.py::test_write_back_usr_overwrite_guard_allows_new_columns_missing_from_existing_overlay`
2. Confirmed red state:
   - `uv run pytest -q src/dnadesign/infer/tests/contracts/test_usr_writeback_contract.py -k missing_from_existing_overlay`
3. Implemented schema-aware overwrite guard.
4. Added failing `run` mode-contract tests:
   - mixed `--config` and `--preset`
   - preset-only flags used in config mode
5. Confirmed red state:
   - `uv run pytest -q src/dnadesign/infer/tests/cli/test_run_command_config_inputs.py`
6. Implemented explicit run-mode contract validator and re-verified green.
7. Added failing IA contract for minimal infer root surface and removed stale `infer/config.yaml`.

### Changes applied

- Write-back guard hardening:
  - `src/dnadesign/infer/src/writers/usr.py`
  - infer overlay schema is inspected first; overwrite guard reads only columns present in schema.
  - fail-fast if overlay schema is unreadable or missing `id`.
- Run command UX hardening:
  - `src/dnadesign/infer/src/cli/commands/run.py`
  - added explicit mode-contract checks:
    - reject `--config` + `--preset` together.
    - reject preset-only flags in config mode (`--usr`, `--field` when non-default, `--ids`, `--usr-root`, `--write-back`).
    - reject `--job` in preset mode.
- IA/cruft tightening:
  - removed stale root file: `src/dnadesign/infer/config.yaml`
  - added minimal top-level surface contract:
    - `src/dnadesign/infer/tests/package/test_source_tree_contracts.py::test_infer_root_keeps_minimal_top_level_surface`

### Verification evidence

- `uv run pytest -q src/dnadesign/infer/tests/contracts/test_usr_writeback_contract.py -k missing_from_existing_overlay`
- `uv run pytest -q src/dnadesign/infer/tests/contracts/test_usr_writeback_contract.py`
- `uv run pytest -q src/dnadesign/infer/tests/cli/test_run_command_config_inputs.py`
- `uv run pytest -q src/dnadesign/infer/tests/package/test_source_tree_contracts.py -k minimal_top_level_surface`
- `uv run pytest -q src/dnadesign/infer/tests/contracts/test_usr_writeback_contract.py src/dnadesign/infer/tests/cli/test_run_command_config_inputs.py src/dnadesign/infer/tests/cli/test_workspace_command.py src/dnadesign/infer/tests/package/test_source_tree_contracts.py`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run pytest -q src/dnadesign/notify/tests/test_events_source.py src/dnadesign/notify/tests/test_workspace_source.py src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py -k infer`

### Notes / next opportunities

- Remaining UX hardening candidate: add CLI characterization tests for broader `run` error-envelope mapping (exit code + message stability per mode).
- Remaining IA candidate: consolidate any future package examples under `docs/operations/examples/` or `workspaces/` only (avoid adding root-level samples).

## 2026-03-06 - Phase 2 Slice M (Workspace Profile Contract + Portable Local Scaffold)

### Goal

Remove workspace-init portability footguns by making local-file scaffolds the default while keeping explicit pressure-test template routing for USR workflows.

### Audit findings

1. High UX footgun: `infer workspace init` defaulted to a USR pressure template with environment-specific root and dataset assumptions.
2. Medium docs drift: pressure-test docs implied default workspace init was always correct for USR route without stating template profile contract.
3. Medium contract gap: no CLI test enforced profile-based template selection behavior.

### TDD record

1. Added failing workspace command tests for:
   - default scaffold content (`source: records`, `path: inputs/records.jsonl`)
   - explicit `--profile usr-pressure`
   - invalid profile rejection contract.
2. Added failing docs contract checks requiring pressure-test walkthroughs to call `workspace init` with `--profile usr-pressure`.
3. Confirmed red state on targeted suite.
4. Implemented profile-based template resolution and command wiring.
5. Re-ran targeted suite and confirmed green.

### Changes applied

- Workspace profile contract:
  - `src/dnadesign/infer/src/workspace.py`
  - `resolve_workspace_template(..., profile=...)` supports:
    - `local` -> `workspace_local_records_config.yaml`
    - `usr-pressure` -> `pressure_test_infer_config.yaml`
  - invalid profile fails fast with explicit choices.
  - `init_workspace(..., profile=...)` now uses profile resolver.
- CLI workspace ergonomics:
  - `src/dnadesign/infer/src/cli/commands/workspace.py`
  - added `--profile` to `workspace where` and `workspace init`.
  - `workspace where` now prints `workspace_profile`.
  - `workspace init` now prints selected profile and warns in `usr-pressure` mode to review `ingest.dataset` and `ingest.root`.
- New portable template:
  - `src/dnadesign/infer/docs/operations/examples/workspace_local_records_config.yaml`
  - default scaffold uses local JSONL records in `inputs/records.jsonl`.
- Added CLI characterization coverage:
  - `src/dnadesign/infer/tests/cli/test_workspace_command.py::test_workspace_local_profile_supports_validate_and_dry_run`
  - verifies `workspace init` -> `validate config` -> `run --dry-run` local profile flow.
- Docs alignment:
  - `src/dnadesign/infer/workspaces/README.md`
  - `src/dnadesign/infer/docs/operations/README.md`
  - `src/dnadesign/infer/docs/reference/command-contracts.md`
  - `src/dnadesign/infer/docs/operations/pressure-test-agnostic-models.md`
  - `src/dnadesign/infer/docs/tutorials/demo_pressure_test_usr_ops_notify.md`

### Verification evidence

- `uv run pytest -q src/dnadesign/infer/tests/cli/test_workspace_command.py src/dnadesign/infer/tests/docs/test_pressure_runbook_docs_contract.py src/dnadesign/infer/tests/docs/test_information_architecture_contracts.py -k "workspace_init or pressure_runbook_docs_include_standalone_and_ops_paths or infer_pressure_test_tutorial_covers_local_and_ops_paths"`
- `uv run infer workspace where`
- `uv run infer workspace where --profile usr-pressure`
- `uv run pytest -q src/dnadesign/infer/tests`
- `uv run pytest -q src/dnadesign/notify/tests/test_events_source.py src/dnadesign/notify/tests/test_workspace_source.py src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py -k infer`

### Notes / next opportunities

- Add `infer workspace doctor --config <path>` to validate scaffold files exist (`inputs/records.jsonl` etc.) before runtime.
- Add docs snippet with minimal JSONL example under `docs/getting-started/`.

## 2026-03-06 - Phase 2 Slice N (40B Pressure-Test Readiness + 3-Hour Infer Batch Plan)

### Goal

Record concrete SCC findings for current infer behavior (7B baseline and desired 40B pressure path), then define an execution-ready plan for promptable 3-hour infer jobs with explicit fresh/reset/resume semantics.

### Read-only scheduler and storage snapshot

- Scheduler snapshot (`qstat -u "$USER"` and `uv run python -m dnadesign.ops.orchestrator.gates session-counts`):
  - running jobs: `2`
  - queued jobs: `1`
  - Eqw jobs: `0`
- Storage snapshot (`df -h /projectnb/dunlop/esouth /project/dunlop/esouth /scratch/$USER`):
  - `/projectnb`: `91%` used (`799G/880G`)
  - `/project`: `44%` used (`96G/220G`)
  - `/scratch`: `14%` used (`112G/876G`)

### Findings (current-state audit)

1. High: infer model registry is still 7B-centric.
   - Registered model ids are `evo2_7b` and `evo2_1b_base`; `evo2_40b` is not registered.
   - Practical effect: `model.id: evo2_40b` currently fails fast at registry resolution before adapter instantiation.
2. High: ops infer submit path does not execute infer runtime.
   - Runbook submit injects `INFER_CONFIG=...` into `qsub -v`, but `docs/bu-scc/jobs/evo2-gpu-infer.qsub` is still a GPU smoke shell and does not call `uv run infer run --config "$INFER_CONFIG"`.
   - Practical effect: a successful submit today can complete without writing infer columns.
3. High: existing 40B model cache is large and concentrated on `/projectnb`.
   - `models--arcinstitute--evo2_40b` exists under `/projectnb/dunlop/esouth/cache/huggingface/hub/` at ~`77G`.
   - Snapshot contains `evo2_40b.pt.part0` and `evo2_40b.pt.part1` symlinked to two ~`39G` blobs.
4. Medium: infer mode semantics in ops are not yet hard-gated like densegen.
   - infer has no explicit resume-readiness policy in `_contracts.resume_readiness`.
   - `--mode auto` can choose `resume` from generic artifact markers, but `--mode fresh` is still accepted even when artifacts exist (no `--allow-fresh-reset` enforcement for infer).
5. Medium: infer overlay-part preflight guard is explicit skip.
   - `usr-overlay-guard --tool infer_evo2` returns `guard_status=skipped` because infer_evo2 contract reports no overlay-part emission.
6. Medium: reset/prune ergonomics are missing for infer-only columns.
   - There is no infer CLI command to prune only infer namespaced columns for a dataset/workspace.
   - Current behavior is limited to write-back `overwrite` semantics plus resume-by-missing-column logic.

### Real-run behavior (if executed today)

1. Ops path (`uv run ops runbook execute --workflow infer ... --submit`):
   - preflight passes (dir checks, config validate, qsub verify, shape advisor/operator brief).
   - submit command launches `evo2-gpu-infer.qsub`.
   - job script does module loads + CUDA smoke print, then exits.
   - infer outputs (`infer__...`) are not produced because infer runtime is not invoked.
2. Direct infer path with current pressure template (`model.id: evo2_7b`):
   - `uv run infer run --config ... --job pressure_evo2_logits_llr` can write `logits_mean` and `llr_mean` columns for USR datasets when run on a GPU-capable node.
3. Direct infer path with `model.id: evo2_40b` right now:
   - fails fast with unknown model id contract unless infer registry is extended for `evo2_40b`.

### Proposed transient layout for 40B pressure runs

- Keep heavy model shards read-only in existing `/projectnb` Hugging Face hub cache.
- Route runtime transients to `/project` (or job-local scratch) to avoid additional `/projectnb` sprawl:
  - `TMPDIR=<workspace>/outputs/runtime/tmp/$JOB_ID`
  - `UV_CACHE_DIR=<workspace>/outputs/runtime/uv-cache`
  - `TORCH_EXTENSIONS_DIR=<workspace>/outputs/runtime/torch-extensions`
  - `TRITON_CACHE_DIR=<workspace>/outputs/runtime/triton-cache`
  - `PYTHONPYCACHEPREFIX=<workspace>/outputs/runtime/pycache`
- Keep runbook/audit/stdout artifacts under workspace:
  - `<workspace>/outputs/logs/ops/runbooks/`
  - `<workspace>/outputs/logs/ops/audit/`
  - `<workspace>/outputs/logs/ops/sge/<runbook-id>/`

### Create-Plan Artifact: 3-Hour Infer Promptable Batch Flow

#### Plan intent summary

Enable a user prompt like: "run a 3 hour batch job for infer on workspace X" with deterministic behavior for:

- fresh infer-column addition,
- infer-column-only reset/prune,
- resume after interruption,

while preserving 7B support and allowing 40B pressure tests.

#### Explicit scope

- In scope:
  - infer 40B model-id readiness in infer registry/contracts.
  - infer GPU qsub template execution correctness (`INFER_CONFIG` must be consumed).
  - explicit mode contract (`fresh`, `reset-prune`, `resume`) for infer ops route.
  - transient path policy to keep runtime artifacts off near-full `/projectnb`.
  - docs/runbook updates for low-friction `uv` environment/bootstrap.
- Out of scope:
  - workspace deletion flows from infer.
  - silent fallback behaviors.
  - changing densegen/usr/notify core runtime behavior beyond infer-facing contract additions.

#### Ordered action checklist

1. Add infer 40B registry contract under TDD:
   - failing tests for `adapters list` + config validation with `model.id=evo2_40b`.
   - register `evo2_40b` while keeping existing 7B/1B ids unchanged.
2. Replace infer GPU qsub smoke body with real infer execution contract:
   - require non-empty `INFER_CONFIG`.
   - run `uv run infer validate config --config "$INFER_CONFIG"` then `uv run infer run --config "$INFER_CONFIG"`.
   - keep CUDA/GCC module loading and fail-fast shell options.
3. Add infer mode contract in ops plan/execute:
   - `fresh`: append missing infer outputs only.
   - `resume`: continue from partial infer outputs.
   - `reset-prune`: explicit infer-column prune step, then fresh run.
4. Implement infer-column-only prune command (no workspace deletion):
   - dataset + job/model/output scoped.
   - removes only `infer__<model>__<job>__<out>` targets.
   - guarded by explicit confirmation flag.
5. Add runbook fields for infer mode wiring:
   - infer reset/prune enablement and target scope.
   - explicit transient path block for qsub environment exports.
6. Add no-submit integration tests for prompt-like 3-hour flow:
   - runbook init (`h_rt=03:00:00`) -> plan -> execute `--no-submit`.
   - assert infer submit command consumes `INFER_CONFIG` and mode wiring.
7. Add docs updates:
   - infer pressure-test guide (7B + 40B options),
   - SCC install/quickstart env recommendations for transient isolation,
   - ops runbook docs for infer fresh/reset/resume behavior.
8. Validate on GPU batch session (not login node):
   - run one bounded infer job against `test_stress_ethanol`.
   - verify infer namespaced columns for `logits` + `llr`, plus resume/reset-prune behavior.

#### Validation and risk handling

- Validation gates per increment:
  - `uv run pytest -q src/dnadesign/infer/tests`
  - `uv run pytest -q src/dnadesign/ops/tests/test_runbook_orchestrator.py src/dnadesign/ops/tests/test_sge_gates.py -k infer`
  - `uv run pytest -q src/dnadesign/notify/tests/test_workspace_source.py src/dnadesign/notify/tests/test_events_source.py -k infer`
- Risk controls:
  - keep infer reset strictly column-scoped and opt-in.
  - no implicit mode fallback.
  - no scheduler mutation during planning (`--no-submit` first).

#### Open blockers (true blockers only)

1. Confirm canonical persistent location for 40B shard cache (keep in `/projectnb` vs relocate).
2. Confirm whether infer reset-prune should live under `infer` CLI, `usr` CLI, or both with one canonical implementation.
3. Confirm required notify orchestration events for infer reset-prune runs (if any).

### Evidence commands executed

- `qstat -u "$USER"`
- `uv run python -m dnadesign.ops.orchestrator.gates session-counts`
- `df -h /projectnb/dunlop/esouth /project/dunlop/esouth /scratch/$USER`
- `du -sh /projectnb/dunlop/esouth/cache/huggingface/hub/models--arcinstitute--evo2_40b`
- `ls -lah /projectnb/dunlop/esouth/cache/huggingface/hub/models--arcinstitute--evo2_40b/snapshots/*`
- `uv run ops runbook init --workflow infer --h-rt 03:00:00 ... --no-notify`
- `uv run ops runbook plan --runbook ... --mode auto|fresh|resume`
- `uv run ops runbook execute --runbook ... --audit-json ... --no-submit`

## 2026-03-06 - Phase 2 Slice O (GPU Env Build Execution Plan + Evo2 Upstream Delta)

### Goal

Create an execution-ready, low-fanout plan to build and validate the SCC GPU infer environment, grounded in current repo contracts and current Evo2 upstream state.

### Route + risk posture (SGE ops)

- workflow_id: `infer_batch_submit`
- execution_locus: `scc_login_shell` (`hostname=scc1`)
- queue posture snapshot:
  - running jobs: `2`
  - queued jobs: `1`
  - Eqw jobs: `0`
- advisor output:
  - `advisor=single`
  - `recommended_action=submit_single`
  - `submit_gate=ready`
- risk posture: low fanout (single submit path only), no arrays, enforce `--no-submit` preflight before any real submit.

### Current evidence snapshot

1. Local dev env currently does not have Evo2 extras installed:
   - `evo2`, `transformer-engine`, `flash-attn`, `vtx` were reported as not installed.
   - `torch` is present at `2.8.0+cu128`.
2. Storage pressure remains asymmetric:
   - `/projectnb/dunlop/esouth` is at `91%`.
   - `/project/dunlop/esouth` is at `44%`.
3. 40B shard cache exists and is large under `/projectnb`:
   - model cache directory size ~`77G`
   - two shard blobs (`part0`, `part1`) are ~`39G` each.

### Docs and packaging audit findings

1. `pyproject.toml` uses `infer-evo2` extra with:
   - `torch>=2.8,<2.9` (Linux x86_64 via PyTorch cu128 index),
   - `flash-attn>=2.8.0.post2,<3`,
   - `transformer-engine[pytorch]>=2.0,<3`,
   - `evo2`.
2. High-level install docs are coherent but need one explicit hardening addition:
   - they describe `UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR`, and `HF_HOME`,
   - they do not yet prescribe a full transient-path bundle for infer batch jobs (`TMPDIR`, `TORCH_EXTENSIONS_DIR`, `TRITON_CACHE_DIR`, `PYTHONPYCACHEPREFIX`) to keep build/runtime churn off near-full `/projectnb`.
3. Contract mismatch remains in infer GPU submit template:
   - `ops` injects `INFER_CONFIG=...` into infer submit commands,
   - `docs/bu-scc/jobs/evo2-gpu-infer.qsub` still executes CUDA smoke only and does not run `infer validate config` + `infer run --config`.

### Evo2 upstream delta (as of 2026-03-06)

Upstream probes against `https://github.com/ArcInstitute/evo2` show:

1. Default branch head: `bc7e7bf...`.
2. Released tag observed: `v0.5.0`.
3. Current package metadata on `main` reports `version = "0.5.3"` and includes dependency `vtx>=0.0.8`.
4. README now explicitly calls out `evo2_20b` release and model set:
   - `evo2_7b`, `evo2_20b`, `evo2_40b`, base variants.
5. README states FP8/Transformer Engine requirement for `40B`, `20B`, and `1B`; 7B can run without TE on supported GPUs.

### Update needs inferred from upstream delta

1. Infer adapter/model registry should be updated from current 7B/1B-only registration to include at least `evo2_40b` and likely `evo2_20b` (under TDD) so runbook/model contracts align with available upstream checkpoints.
2. SCC install docs should explicitly separate:
   - 7B lane (lighter, non-TE-capable fallback where valid),
   - FP8 lane (20B/40B/1B with Hopper + TE constraints).
3. Dependency/compatibility note should be tightened in docs:
   - dnadesign currently pins Torch 2.8 for infer extra, while upstream README recommends Torch 2.6/2.7.
   - this should be captured as an explicit compatibility test gate rather than implicit assumption.

### Create-Plan artifact (execution plan)

#### Plan intent summary

Build the infer GPU environment deterministically, then run one bounded pressure-test path that is scheduler-safe and validates infer write-back behavior before scaling out.

#### Explicit scope

- In scope:
  - environment bootstrap + verification for Evo2 infer on SCC,
  - infer submit-template correctness (`INFER_CONFIG` consumption),
  - infer model-id readiness for large-model pressure tests,
  - docs hardening for repeatable SCC GPU setup and transient-path policy.
- Out of scope:
  - multi-submit fanout or arrays in this phase,
  - workspace deletion semantics,
  - silent fallback behavior.

#### Ordered action checklist

1. Lock transience policy in docs/runbook contract:
   - keep model shards where they are,
   - route runtime/build caches to `/project` (or job scratch) with explicit env exports.
2. TDD: harden infer qsub template contract:
   - failing tests for infer submit command expectation and template semantics,
   - enforce required `INFER_CONFIG`,
   - run `uv run infer validate config --config "$INFER_CONFIG"` then `uv run infer run --config "$INFER_CONFIG"`.
3. TDD: extend infer model registration:
   - add failing tests for adapter list/config acceptance of new model ids,
   - register `evo2_40b`, evaluate `evo2_20b` inclusion in same slice if tests remain tight.
4. TDD: add infer mode hardening in ops:
   - codify fresh/resume/reset-prune behavior for infer columns only.
5. Add infer column-prune command contract (no workspace delete path):
   - explicit scope + explicit confirmation flag,
   - no fallback behavior.
6. Harden install docs:
   - add GPU lane matrix (7B vs FP8 models),
   - add deterministic env exports for transients/caches,
   - add compatibility test section tied to current pinned torch + evo2 stack.
7. Execute no-submit workflow only:
   - `ops runbook init --workflow infer --h-rt 03:00:00 --no-notify`,
   - `ops runbook plan`,
   - `ops runbook execute --no-submit`.
8. Submit exactly one job after green gates:
   - `ops runbook execute --submit`,
   - verify `infer__...` namespaced columns (llr/logits) on target dataset.

#### Validation and risk handling

- Required checks before submit:
  1. `qstat -u "$USER"`
  2. `uv run ops runbook plan --runbook <path>`
  3. `uv run ops runbook execute --runbook <path> --audit-json <path> --no-submit`
  4. template verify gate (`qsub -verify`) through ops preflight
- Hard-stop conditions:
  1. any preflight non-zero,
  2. any `Eqw` job appears,
  3. running jobs exceed `3` and new submissions are requested.

#### Open blockers (max 3)

1. Confirm canonical long-term location for 40B model cache (`/projectnb` retained vs migration plan).
2. Confirm whether infer reset-prune command should be owned by infer CLI, usr CLI, or both delegating to one implementation.
3. Confirm first pressure-test sequence: 7B smoke-first vs direct 40B path.

### Commands executed for this slice

- `skills-preflight --json --strict --ensure-fresh --require-hooks`
- `hostname`
- `qstat -u "$USER"`
- `uv run python -m dnadesign.ops.orchestrator.gates session-counts`
- `uv run python -m dnadesign.ops.orchestrator.gates submit-shape-advisor --planned-submits 1 --warn-over-running 3`
- `uv run python -m dnadesign.ops.orchestrator.gates operator-brief --planned-submits 1 --warn-over-running 3`
- `sed -n '1,260p' pyproject.toml`
- `sed -n '1,260p' docs/installation.md`
- `sed -n '1,340p' docs/bu-scc/install.md`
- `sed -n '1,320p' docs/bu-scc/quickstart.md`
- `sed -n '1,260p' docs/bu-scc/jobs/evo2-gpu-infer.qsub`
- `sed -n '1,340p' docs/operations/orchestration-runbooks.md`
- `git ls-remote --heads --tags https://github.com/ArcInstitute/evo2.git`
- `curl -L https://raw.githubusercontent.com/ArcInstitute/evo2/main/README.md`
- `curl -L https://raw.githubusercontent.com/ArcInstitute/evo2/main/pyproject.toml`
- `uv run python - <<'PY' ... importlib.metadata ... PY`
- `df -h /projectnb/dunlop/esouth /project/dunlop/esouth /scratch/$USER`
- `du -sh /projectnb/dunlop/esouth/cache/huggingface/hub/models--arcinstitute--evo2_40b`

## 2026-03-06 - Phase 2 Slice P (Infer-Evo2 Local Env Build Attempt + Interactive Session Gate)

### Goal

Attempt a real `uv sync --locked --extra infer-evo2` with transient/caches routed to `/project`, then record concrete blockers and the required interactive-session bootstrap contract.

### Build attempt executed

1. Created a dedicated env and runtime cache roots under `/project/dunlop/esouth/dnadesign/`:
   - `.venv-infer-evo2-attempt`
   - `runtime/infer-evo2-attempt/{tmp,uv-cache,torch-extensions,triton-cache,pycache}`
2. Ran:
   - `uv sync --locked --extra infer-evo2`
3. Result:
   - dependency resolution + large wheel downloads completed,
   - build failed at `transformer-engine-torch==2.11.0`.

### Root cause evidence

1. Missing CUDA toolchain headers on login host path:
   - repeated compile error: `fatal error: crt/host_defines.h: No such file or directory`
2. Host compiler too old for torch extension build:
   - error: `We need GCC 9 or later.`
3. Host checks during failure window:
   - `gcc (GCC) 8.5.0 ...`
   - `nvcc=missing`

### Operational conclusion

Yes, a proper SCC toolchain context is required for reliable infer-evo2 build completion. A GPU allocation is not required for dependency resolution itself, but practical TE/flash-attn build + runtime validation should be performed in an interactive or batch environment with the correct CUDA/GCC modules loaded.

### Interactive-session bootstrap checklist (next attempt)

1. Start a bounded interactive session with GPU resources.
2. Load explicit modules before `uv sync`:
   - `module purge`
   - `module load cuda/<version>`
   - `module load gcc/<version>`
3. Export compiler/CUDA pointers:
   - `CC`, `CXX`, `CUDAHOSTCXX`, `CUDA_HOME`
4. Keep transient/caches under `/project` or job scratch:
   - `TMPDIR`, `UV_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, `TRITON_CACHE_DIR`, `PYTHONPYCACHEPREFIX`
5. Re-run:
   - `uv sync --locked --extra infer-evo2`
6. Validate after sync:
   - torch CUDA visibility,
   - `transformer_engine`, `flash_attn`, `evo2`, `vtx` imports,
   - infer adapter listing and infer config validation commands.

### Commands executed for this slice

- `uv sync --locked --extra infer-evo2` (with transient/cache exports to `/project`)
- `gcc --version`
- `command -v nvcc`

### Implementation changes landed in this slice

1. Infer model-id readiness:
   - registered `evo2_20b` and `evo2_40b` in infer adapter defaults.
2. Infer GPU qsub contract:
   - `docs/bu-scc/jobs/evo2-gpu-infer.qsub` now requires `INFER_CONFIG`, runs infer config validation, then runs infer config execution.
3. SCC docs hardening:
   - added transient-path guidance and model-lane guidance to `docs/bu-scc/install.md`,
   - updated infer submit examples in `docs/bu-scc/quickstart.md`, `docs/bu-scc/jobs/README.md`, and `docs/bu-scc/batch-notify.md`.

### Verification evidence for landed changes

- `uv run pytest -q src/dnadesign/infer/tests/cli/test_adapters_commands.py src/dnadesign/densegen/tests/docs/test_bu_scc_docs_contracts.py -k "adapters_list_reports_registered_default_model_ids or evo2_qsub_template_requires_infer_config_and_runs_infer_cli or bu_scc_install_doc_includes_infer_transient_path_policy_and_model_lanes"`
- `uv run pytest -q src/dnadesign/infer/tests/cli/test_adapters_commands.py src/dnadesign/densegen/tests/docs/test_bu_scc_docs_contracts.py src/dnadesign/infer/tests/docs/test_information_architecture_contracts.py src/dnadesign/infer/tests/docs/test_pressure_runbook_docs_contract.py`
- `uv run pytest -q src/dnadesign/ops/tests/test_runbook_orchestrator.py -k "infer_runbook_uses_gpu_submit_template_and_filters or infer_batch_submit_without_notify_skips_notify_phase"`

## 2026-03-07 - Phase 2 Slice Q (Top-Level Docs + pyproject Sequence Audit During Interactive GPU Build)

### Goal

Align real SCC GPU build behavior with documented install flow and `pyproject.toml` packaging contracts, then capture an execution sequence that avoids known build/runtime failure modes.

### Top-level sequence audit (docs + pyproject)

1. Top-level route map is coherent:
   - `README.md` -> `docs/README.md` -> `docs/installation.md` -> `docs/bu-scc/{quickstart,install}.md`.
2. Packaging contract in `pyproject.toml` matches intent:
   - `infer-evo2` extra includes `torch` + `flash-attn` + `transformer-engine[pytorch]` + `evo2`.
   - uv is configured with `no-build-isolation-package = ["flash-attn", "transformer-engine-torch"]`.
3. SCC docs cover module/toolchain loading and transient-path routing, but interactive build findings exposed additional critical knobs needed for deterministic success on this host.

### Interactive build findings (real host evidence)

1. Host C library is `glibc 2.28` (`ldd --version`), while installed `flash_attn_2_cuda*.so` required `GLIBC_2.32`.
   - implication: prebuilt `flash-attn` binaries can be incompatible on this SCC host; source build is required.
2. Correct flash-attn force-build env var is:
   - `FLASH_ATTENTION_FORCE_BUILD=TRUE`
   - not `FLASH_ATTN_FORCE_BUILD`.
3. `flash-attn` arch fanout is controlled by:
   - `FLASH_ATTN_CUDA_ARCHS` (defaults to `80;90;100;120` in upstream setup.py),
   - not `TORCH_CUDA_ARCH_LIST` for this package/version.
4. Transformer Engine include discovery can fail on `nccl.h` when include paths are narrowed.
   - using only `NVTE_CUDA_INCLUDE_PATH=$CUDA_HOME/include` can hide NCCL headers from nvidia wheel includes.
   - stable fix: compose `CPATH`/`CPLUS_INCLUDE_PATH` with both `$CUDA_HOME/include` and venv `site-packages/nvidia/*/include`.

### Recommended sequence of events (SCC interactive build lane)

1. Load toolchain first (`module purge`, `module load cuda/...`, `module load gcc/...`), export `CC/CXX/CUDAHOSTCXX/CUDA_HOME`.
2. Pin env + transients to project paths (`UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR`, `TMPDIR`, `TORCH_EXTENSIONS_DIR`, `TRITON_CACHE_DIR`, `PYTHONPYCACHEPREFIX`).
3. Build include path from:
   - `$CUDA_HOME/include`
   - venv `site-packages/nvidia/*/include`
4. Force source build for flash-attn:
   - `FLASH_ATTENTION_FORCE_BUILD=TRUE`
   - `FLASH_ATTN_CUDA_ARCHS=80` (single-arch compile path for this host/toolchain)
5. Install/repair:
   - `uv sync --locked --extra infer-evo2` (or targeted `uv sync --reinstall-package flash-attn ...` for remediation).
6. Verify binary compatibility and runtime imports:
   - inspect `flash_attn_2_cuda*.so` GLIBC symbols,
   - run `uv run python` imports for `torch`, `transformer_engine`, `flash_attn`, `evo2`, `vtx`,
   - run `uv run infer adapters list`.

### Current status (slice checkpoint)

1. Interactive rebuild rerun is active with:
   - `FLASH_ATTENTION_FORCE_BUILD=TRUE`
   - `FLASH_ATTN_CUDA_ARCHS=80`
2. Compile invocation confirms single-arch `-gencode arch=compute_80,code=sm_80` path.
3. Final import/GLIBC verification remains pending on completion of this build step.

### Commands executed for this slice

- `sed -n ... README.md docs/README.md docs/installation.md docs/bu-scc/install.md docs/bu-scc/quickstart.md docs/bu-scc/jobs/README.md`
- `sed -n ... pyproject.toml`
- `nl -ba ... pyproject.toml docs/installation.md docs/bu-scc/install.md docs/bu-scc/quickstart.md`
- `ldd --version`
- `getconf GNU_LIBC_VERSION`
- `strings .venv-infer-evo2-gpu/lib/python3.12/site-packages/flash_attn_2_cuda*.so | rg GLIBC_`
- `rg -n 'FLASH_ATTENTION_FORCE_BUILD|FLASH_ATTN_CUDA_ARCHS|gencode' <flash-attn setup.py in uv sdist temp>`
- `uv pip install --python .venv-infer-evo2-gpu/bin/python --reinstall-package flash-attn --no-binary flash-attn --no-build-isolation --refresh-package flash-attn --refresh --no-cache flash-attn==2.8.3`

## 2026-03-07 - Phase 2 Slice R (Deterministic GPU Docs Hardening: Runbook + Didactic Lane)

### Goal

Harden documentation so a human or machine can deterministically bootstrap Evo2 GPU infer on SCC with clear UV lane semantics, explicit export ordering, and infer capability verification.

### Audit summary

1. Top-level docs already route correctly (`README` -> `docs/README` -> `docs/installation` -> `docs/bu-scc/*`) but lacked one explicit deterministic GPU lane with the discovered flash-attn/TE controls.
2. Infer docs had pressure-test and demo runbooks but lacked a dedicated SCC GPU environment runbook tied to infer capability checks.
3. Existing docs tests covered SCC install transient/model-lane sections but did not enforce deterministic flash-attn source-build controls or infer runbook discoverability for this lane.

### Changes applied

1. Added infer deterministic runbook:
   - `src/dnadesign/infer/docs/operations/scc-evo2-gpu-uv-runbook.md`
   - includes:
     - machine lane (copy/paste deterministic command sequence),
     - didactic lane (UV lanes, extras/groups, exports rationale),
     - infer capability checks (`infer adapters list`, `infer validate config`).
2. Updated infer docs IA routing:
   - `src/dnadesign/infer/docs/operations/README.md`
   - `src/dnadesign/infer/docs/README.md`
   - `src/dnadesign/infer/docs/index.md`
   - `src/dnadesign/infer/docs/getting-started/README.md`
   - `src/dnadesign/infer/README.md`
   - `src/dnadesign/infer/docs/operations/pressure-test-agnostic-models.md`
3. Hardened BU SCC + top-level docs:
   - `docs/bu-scc/install.md`
     - added `### Deterministic GPU build runbook (copy/paste lane)`
     - added didactic rationale section
     - documented `FLASH_ATTENTION_FORCE_BUILD`, `FLASH_ATTN_CUDA_ARCHS`, `CPATH`, `CPLUS_INCLUDE_PATH`
   - `docs/installation.md`
     - added explicit UV lane model (`default-groups`, `--group dev`, `--extra infer-evo2`)
     - linked deterministic SCC GPU lane + infer runbook
   - cross-link updates:
     - `docs/README.md`
     - `docs/bu-scc/README.md`
     - `docs/bu-scc/quickstart.md`
     - `docs/bu-scc/jobs/README.md`
     - `docs/bu-scc/batch-notify.md`
4. Added/updated doc contract tests (TDD):
   - `src/dnadesign/densegen/tests/docs/test_bu_scc_docs_contracts.py`
     - new deterministic flash-attn control assertions
   - `src/dnadesign/infer/tests/docs/test_information_architecture_contracts.py`
     - operations index must link SCC GPU runbook
   - `src/dnadesign/infer/tests/docs/test_scc_gpu_env_docs_contract.py`
     - new infer SCC GPU runbook content contract

### Verification evidence

- Failing test gate (before docs changes):
  - deterministic SCC install section missing,
  - infer operations index missing new runbook link,
  - runbook file missing.
- Green test gate (after docs changes):
  - `uv run pytest -q src/dnadesign/densegen/tests/docs/test_bu_scc_docs_contracts.py src/dnadesign/infer/tests/docs/test_information_architecture_contracts.py src/dnadesign/infer/tests/docs/test_pressure_runbook_docs_contract.py src/dnadesign/infer/tests/docs/test_scc_gpu_env_docs_contract.py`
  - result: pass (`21` tests).

### Runtime build status snapshot

1. Active flash-attn source build remains in progress in `.venv-infer-evo2-gpu`.
2. Current compile path confirms bounded arch control (`-gencode arch=compute_80,code=sm_80`).
3. Final runtime import verification is still pending completion of that build command.

## 2026-03-07 - Phase 2 Slice S (UV Workflow Policy Correction: No uv pip in Canonical Path)

### Goal

Correct the environment-repair workflow to stay on canonical UV project commands and encode the policy in docs/contracts.

### Critical evaluation

1. `uv add` / `uv remove` are the correct commands for dependency declaration changes because they update `pyproject.toml` and `uv.lock`.
2. `uv sync` is the correct command for environment realization and rebuilds when dependency declarations are unchanged.
3. `uv pip` is not the preferred path for this repo's canonical runbooks and was removed from active guidance.

### Corrective actions applied

1. Stopped the in-flight `uv pip` flash-attn remediation process and orphaned compile workers.
2. Replaced active rebuild path with lock-driven sync:
   - `uv sync --locked --extra infer-evo2 --reinstall-package flash-attn --refresh --refresh-package flash-attn --no-binary-package flash-attn --no-cache`
3. Updated docs to encode UV policy explicitly:
   - `docs/installation.md`
   - `docs/bu-scc/install.md`
   - `src/dnadesign/infer/docs/operations/scc-evo2-gpu-uv-runbook.md`
4. Updated docs contract tests to enforce UV policy wording:
   - `src/dnadesign/densegen/tests/docs/test_bu_scc_docs_contracts.py`
   - `src/dnadesign/infer/tests/docs/test_scc_gpu_env_docs_contract.py`

### Verification evidence

- `uv run pytest -q src/dnadesign/densegen/tests/docs/test_bu_scc_docs_contracts.py src/dnadesign/infer/tests/docs/test_information_architecture_contracts.py src/dnadesign/infer/tests/docs/test_pressure_runbook_docs_contract.py src/dnadesign/infer/tests/docs/test_scc_gpu_env_docs_contract.py`
- result: pass (`21` tests).

### Runtime status

1. Canonical `uv sync` source-build path is active for flash-attn rebuild.
2. Import-level compatibility verification remains pending completion.

## 2026-03-07 - Phase 2 Slice T (Docs Language Hardening: Direct, Plain, Single-Path Guidance)

### Goal

Rewrite installation and infer SCC documentation to use plain, direct language with clear sequential instructions, no human/agent phrasing, and no divergent path framing.

### Changes applied

1. Updated top-level installation guidance:
   - `docs/installation.md`
   - renamed lane/branch framing to direct section titles (`Platform support`, `UV dependency model`, `Development tools (when needed)`).
   - preserved UV policy clarity (`uv add/remove` for declaration changes, `uv sync` for realization/rebuilds).
2. Updated BU SCC install/runbook wording:
   - `docs/bu-scc/install.md`
   - replaced `At a glance`/`Choose your path` with direct scope language.
   - renamed runbook sections to:
     - `GPU setup and verification runbook`
     - `Why these settings are required`
   - replaced model-lane wording with model-support wording.
3. Updated infer SCC runbook wording:
   - `src/dnadesign/infer/docs/operations/scc-evo2-gpu-uv-runbook.md`
   - replaced `machine lane`/`human lane` headings with:
     - `Setup and verification steps`
     - `Why this setup works`
4. Updated cross-links to new BU SCC install anchor:
   - `docs/README.md`
   - `docs/bu-scc/README.md`
   - `docs/bu-scc/quickstart.md`
   - `docs/bu-scc/batch-notify.md`
   - `docs/bu-scc/jobs/README.md`
5. Added/updated docs contracts to enforce style and structure:
   - `src/dnadesign/densegen/tests/docs/test_bu_scc_docs_contracts.py`
   - `src/dnadesign/infer/tests/docs/test_scc_gpu_env_docs_contract.py`
   - updated expectations for new section names and removed phrasing.

### Verification evidence

- `uv run pytest -q src/dnadesign/densegen/tests/docs/test_bu_scc_docs_contracts.py src/dnadesign/infer/tests/docs/test_information_architecture_contracts.py src/dnadesign/infer/tests/docs/test_pressure_runbook_docs_contract.py src/dnadesign/infer/tests/docs/test_scc_gpu_env_docs_contract.py`
- result: pass (`22` tests).

### Runtime note

1. Stopped the in-progress `uv sync --reinstall-package flash-attn` build process after docs pass completion to avoid leaving background compile load running.

## 2026-03-07 - Phase 2 Slice U (Documentation Tone + Single-Sequence Runbook Pass)

### Goal

Run a strict language pass on installation + BU SCC + infer pressure-test docs so the instructions stay plain, direct, and sequential without branch-labeled runbook paths.

### Changes applied

1. Tightened docs style contract tests first (red -> green):
   - `src/dnadesign/densegen/tests/docs/test_bu_scc_docs_contracts.py`
   - `src/dnadesign/infer/tests/docs/test_pressure_runbook_docs_contract.py`
   - Added forbidden phrasing gates for lane/agent labels and branch-labeled `Path A/B/C` sections.
2. Rewrote wording in installation and BU SCC index/bootstrap docs:
   - `docs/installation.md`
   - `docs/bu-scc/README.md`
   - `docs/bu-scc/install.md`
   - `docs/bu-scc/batch-notify.md`
   - `docs/README.md`
   - renamed `docs/bu-scc/agent-cheatsheet.md` -> `docs/bu-scc/submission-reference.md` and updated links.
3. Rewrote infer docs wording and pressure runbook structure:
   - `src/dnadesign/infer/docs/README.md`
   - `src/dnadesign/infer/docs/operations/pressure-test-agnostic-models.md`
   - Converted branch labels into one ordered procedure (`1..9`) while preserving all required commands.

### Verification evidence

- `uv run pytest -q src/dnadesign/densegen/tests/docs/test_bu_scc_docs_contracts.py src/dnadesign/infer/tests/docs/test_information_architecture_contracts.py src/dnadesign/infer/tests/docs/test_pressure_runbook_docs_contract.py src/dnadesign/infer/tests/docs/test_scc_gpu_env_docs_contract.py`
- result: pass (`22` tests).

## 2026-03-07 - Phase 2 Slice V (GPU Env Pressure Test Audit: Real Build Evidence + Runbook Hardening)

### Goal

Pressure-test the installation and infer BU SCC GPU docs as a first-time operator workflow and verify whether the deterministic runbook builds a usable Evo2 infer environment end-to-end.

### Environment evidence

1. Host: `scc-505` (interactive GPU session).
2. GPUs: `NVIDIA L40S` (multiple devices visible via `nvidia-smi -L`).
3. Modules used: `cuda/12.8`, `gcc/13.2.0`.
4. UV contract: `uv 0.9.18`, project requires Python `3.12`.

### Pressure-test execution summary

1. Base lock sync completed successfully in fresh env root:
   - `uv sync --locked` completed and installed torch CUDA stack (`torch 2.8.0+cu128`) and base dependencies.
2. Evo2 extra sync entered source build path for flash-attn:
   - `uv sync --locked --extra infer-evo2` triggered `flash-attn` CUDA compilation.
   - Build inventory showed `72` CUDA translation units in flash-attn source tree.
   - Runtime observation: with constrained build controls (`MAX_JOBS=1`, `CMAKE_BUILD_PARALLEL_LEVEL=1`), compile wall-clock extended beyond 30 minutes and was still in-progress.
3. Partial-state footgun observed after interrupted compile:
   - Dist metadata contained `transformer-engine` but missing runtime torch extension (`transformer_engine.pytorch` import failed).
   - `flash-attn` package remained missing.
   - `infer validate config --config ...` still passed, which can mask unusable runtime extension state.

### Root-cause findings

1. Lockfile expectation mismatch in docs:
   - `flash-attn` entry in `uv.lock` is sdist-only (no wheel entries), so source compilation is expected in current lock state.
2. Runbook verification was not fail-fast enough:
   - previous verification printed versions but did not fail when required runtime imports were absent.
3. Efficiency posture was too conservative by default:
   - hard-coded single-thread build caps maximize determinism but can make first build impractically long on available GPU nodes.

### Hardening changes applied

1. Added explicit lockfile expectation to runbooks:
   - `flash-attn is sdist-only in uv.lock` now called out in infer SCC runbook and BU SCC install runbook.
2. Added fail-fast runtime verification gate:
   - runbook now emits `MISSING_REQUIRED` and exits non-zero (`raise SystemExit(1)`) when required dist/module imports fail.
3. Added deterministic recovery command for partial installs:
   - `uv sync --locked --extra infer-evo2 --reinstall-package flash-attn --reinstall-package transformer-engine-torch`.
4. Updated build-control guidance:
   - parallel defaults for normal runs, single-thread caps reserved for memory-constrained fallback.

### Verification evidence

- Docs contract tests after hardening:
  - `uv run pytest -q src/dnadesign/densegen/tests/docs/test_bu_scc_docs_contracts.py src/dnadesign/infer/tests/docs/test_information_architecture_contracts.py src/dnadesign/infer/tests/docs/test_pressure_runbook_docs_contract.py src/dnadesign/infer/tests/docs/test_scc_gpu_env_docs_contract.py`
  - result: pass.

## 2026-03-07 - Phase 2 Slice W (Resource Budget Gate: 4 cores + 1x L40S)

### Goal

Add a deterministic, budget-aware preflight so build and run decisions are mechanical for constrained SCC sessions and do not rely on ad-hoc tuning.

### Session evidence

1. GPU query:
   - `nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader`
   - observed: `NVIDIA L40S, 46068 MiB, 8.9`
2. Scheduler/session variables:
   - `NSLOTS=4`
   - `OMP_NUM_THREADS=1` (default shell state can underutilize allocated cores unless explicitly set)
   - `TMPDIR=/scratch/<jobid>.1.l40s`
3. Python/runtime precision contract in infer:
   - only `fp32`, `fp16`, `bf16` are accepted (`src/dnadesign/infer/src/config.py`).
   - no quantized/offloaded inference path is exposed in infer CLI/config.

### Capacity calculation used for preflight

- weight memory estimate: `params * bytes_per_param`
- guard band: `required_gib = weight_gib * 1.25`
- GPU usable headroom: `gpu_total_gib * 0.90`
- fail condition: `required_gib > gpu_usable_gib`

Approximate bf16/fp16 values:

1. `evo2_7b`: weights `13.0 GiB`, guarded `16.3 GiB`
2. `evo2_20b`: weights `37.3 GiB`, guarded `46.6 GiB`
3. `evo2_40b`: weights `74.5 GiB`, guarded `93.1 GiB`
4. `400B` class: weights `745.1 GiB`, guarded `931.3 GiB`

Implication on one L40S (`~45.0 GiB` total):

1. `evo2_7b`: expected fit.
2. `evo2_20b`: borderline-to-fail under guard-band policy.
3. `evo2_40b`: fail.
4. `400B`: out of scope by both model-id support and capacity.

### Hardening changes applied

1. Added `Capacity and build profile gate` to:
   - `src/dnadesign/infer/docs/operations/scc-evo2-gpu-uv-runbook.md`
2. Added mirrored `6.4 Capacity gate and resource profile` to:
   - `docs/bu-scc/install.md`
3. Gate behavior:
   - derives `FLASH_ATTN_CUDA_ARCHS` from detected compute capability (`8.9 -> 89`),
   - derives build jobs from `NSLOTS` (`1/2/4` cap),
   - exports deterministic build knobs,
   - emits `RESOURCE_GATE_OK` on pass,
   - emits `RUN_CAPACITY_FAIL` and exits non-zero on fail.
4. Added explicit model-id scope note:
   - infer supports `evo2_7b`, `evo2_20b`, `evo2_40b`;
   - 400B is not a supported infer `model.id`.

### Test hardening updates

1. Updated infer docs contract:
   - `src/dnadesign/infer/tests/docs/test_scc_gpu_env_docs_contract.py`
   - now asserts presence of gate tokens (`TARGET_MODEL_ID`, `RUN_CAPACITY_FAIL`, `RESOURCE_GATE_OK`).
2. Updated BU SCC docs contract:
   - `src/dnadesign/densegen/tests/docs/test_bu_scc_docs_contracts.py`
   - now asserts install doc includes capacity gate section + 400B scope note.

### Verification evidence

- `uv run pytest -q src/dnadesign/densegen/tests/docs/test_bu_scc_docs_contracts.py src/dnadesign/infer/tests/docs/test_scc_gpu_env_docs_contract.py src/dnadesign/infer/tests/docs/test_information_architecture_contracts.py src/dnadesign/infer/tests/docs/test_pressure_runbook_docs_contract.py`

## 2026-03-07 - Phase 2 Slice X (Infer Multi-GPU Contract Hardening: Topology + Capacity Preflight)

### Goal

Land the first implementation increment for flexible multi-GPU infer behavior with strict fail-fast contract checks, while preserving current single-GPU 7B behavior.

### Scope of implementation

1. Add explicit model topology contract to infer config.
2. Add runtime GPU inventory probe and capacity validator.
3. Wire fail-fast capacity checks into:
   - `infer validate config`
   - `infer run --dry-run` (and `infer run` preflight path)
4. Keep current CLI surfaces unchanged.

### Changes applied

1. Config contract update:
   - `src/dnadesign/infer/src/config.py`
   - added `ModelParallelismConfig` with:
     - `strategy`: `single_device | multi_gpu_vortex`
     - `min_gpus`
     - `gpu_ids`
   - added `model.parallelism` to `ModelConfig` with strict validation.
2. Runtime probe module:
   - `src/dnadesign/infer/src/runtime/hardware_probe.py`
   - provides `GpuDeviceInfo`, `GpuInventory`, `probe_gpu_inventory()`.
3. Runtime capacity planner:
   - `src/dnadesign/infer/src/runtime/capacity_planner.py`
   - computes guarded model memory requirement (`weights * 1.25`) for Evo2 ids and validates against usable GPU memory (`sum(vram * 0.90)` across required devices).
   - enforces topology contract (`required_gpus`, valid `gpu_ids`, no multi-gpu strategy on non-cuda device).
   - emits explicit `ValidationError` messages prefixed with `CAPACITY_FAIL`.
4. CLI fail-fast wiring:
   - `src/dnadesign/infer/src/cli/commands/validate.py`
   - `src/dnadesign/infer/src/cli/commands/run.py`
   - both now call `validate_model_hardware_contract(..., inventory=probe_gpu_inventory())` before success summary or execution.
5. Test hardening:
   - new: `src/dnadesign/infer/tests/runtime/test_capacity_planner.py`
   - updated:
     - `src/dnadesign/infer/tests/cli/test_validate_command.py`
     - `src/dnadesign/infer/tests/cli/test_run_command_config_inputs.py`
   - fixed stale registry expectation to avoid false failures as model registrations expanded:
     - `src/dnadesign/infer/tests/package/test_registry_bootstrap_contracts.py`

### Behavior outcomes

1. `evo2_40b` + one ~45 GiB GPU now fails during validate/dry-run with explicit `CAPACITY_FAIL`.
2. `evo2_7b` + one ~45 GiB GPU passes capacity checks.
3. Multi-GPU strategy requires explicit minimum GPU count and valid GPU id mapping; no silent fallback to single device.

### Verification evidence

1. Targeted TDD slice:
   - `uv run pytest -q src/dnadesign/infer/tests/runtime/test_capacity_planner.py src/dnadesign/infer/tests/cli/test_validate_command.py src/dnadesign/infer/tests/cli/test_run_command_config_inputs.py`
2. Full infer package tests:
   - `uv run pytest -q src/dnadesign/infer/tests`
3. Previously hardened docs contract tests remain green:
   - `uv run pytest -q src/dnadesign/densegen/tests/docs/test_bu_scc_docs_contracts.py src/dnadesign/infer/tests/docs/test_scc_gpu_env_docs_contract.py`

## 2026-03-07 - Phase 2 Slice Y (Ops Harness Hardening + Docs/Runbook Audit)

### Goal

1. Enforce infer topology/capacity contract compatibility in ops runbook planning before submit command rendering.
2. Audit and harden docs/runbooks so naive operators can build and preflight environments with canonical UV `sync --group/--extra` flows and explicit resource semantics.

### Harness lane and endpoints used

1. Primary lane: `autonomy-hardening`
2. Secondary lane: `knowledge-refresh`
3. Endpoints in scope:
   - `autonomy-capability`
   - `architecture-invariants`
   - `knowledge-integrity`

### Ops contract hardening applied

1. Added infer runbook resource contract checks in ops planner:
   - `src/dnadesign/ops/orchestrator/plan.py`
   - loads infer `model` contract from runbook config and validates it against runbook GPU resources before preflight command generation.
2. Added runbook resource field:
   - `resources.gpu_memory_gib` (optional, positive float)
   - schema: `src/dnadesign/ops/runbooks/schema.py`
   - densegen workflows now reject `resources.gpu_memory_gib` alongside other GPU-only keys.
3. Added infer runbook scaffold default:
   - `gpu_memory_gib: 45.0` for default infer scaffold
   - `src/dnadesign/ops/cli.py`
4. Infer planner capacity behavior:
   - uses `resources.gpu_memory_gib` when provided;
   - falls back to capability hints for known class (`gpu_capability=8.9 -> 45.0 GiB`);
   - fails fast when runbook resources are incompatible with infer model contract (`CAPACITY_FAIL` surface via runbook contract error).

### Infer CLI capacity-check context hardening

1. `infer validate config` now skips hardware-capacity enforcement when no local GPUs are visible and the model targets CUDA; it still validates schema/contracts and prints explicit guidance to use runbook planning for declared resources.
2. `infer run --dry-run` now skips hardware-capacity enforcement only when no local GPUs are visible; on GPU nodes it still enforces full capacity checks.

### Docs/runbook audit findings and fixes

1. Gap found:
   - Ops runbook docs did not describe `resources.gpu_memory_gib` or infer model.parallelism/resource preflight interaction.
2. Fixes:
   - `docs/operations/orchestration-runbooks.md`
     - added infer route support for `resources.gpu_memory_gib`;
     - added explicit infer planner contract rules for model.parallelism/resource validation and capability fallback.
   - `docs/bu-scc/quickstart.md`
     - added runbook guidance to set `resources.gpus`, `resources.gpu_capability`, and optional `resources.gpu_memory_gib` for deterministic infer preflight.
   - `src/dnadesign/infer/docs/operations/scc-evo2-gpu-uv-runbook.md`
     - clarified that `infer validate config` on GPU-less hosts reports capacity-check skip and points to `ops runbook plan` for declared scheduler-resource preflight.
3. UV canonical-flow audit:
   - current docs consistently use `uv sync --locked`, `uv sync --locked --group dev`, and `uv sync --locked --extra infer-evo2` for canonical environment realization;
   - no active install/runbook paths use `uv pip`.

### Test and evidence updates

1. Added runtime/CLI contract tests:
   - `src/dnadesign/infer/tests/runtime/test_capacity_planner.py`
   - `src/dnadesign/infer/tests/cli/test_validate_command.py`
   - `src/dnadesign/infer/tests/cli/test_run_command_config_inputs.py`
2. Added ops runbook planner coverage:
   - `src/dnadesign/ops/tests/test_runbook_orchestrator.py`
   - includes explicit failure test for `evo2_40b` with single-GPU runbook resources.
3. Updated docs contract checks:
   - `src/dnadesign/ops/tests/test_ops_docs_progressive_disclosure_contracts.py`
   - `src/dnadesign/infer/tests/docs/test_scc_gpu_env_docs_contract.py`
4. Verification commands:
   - `uv run pytest -q src/dnadesign/infer/tests src/dnadesign/ops/tests/test_runbook_orchestrator.py src/dnadesign/ops/tests/test_sge_gates.py src/dnadesign/ops/tests/test_ops_docs_progressive_disclosure_contracts.py src/dnadesign/densegen/tests/docs/test_bu_scc_docs_contracts.py src/dnadesign/infer/tests/docs/test_scc_gpu_env_docs_contract.py`
   - skill-harness audit: `/usr4/dl523/esouth/.agents/skills/harness-engineering/scripts/audit_skill_contracts.sh`

## 2026-03-07 - Phase 2 Slice Z (Canonical `.venv` GPU Build + Real Evo Smoke + Runtime Bug Fix)

### Goal

Pressure-test the documented SCC GPU build path in the canonical repo UV environment (`/project/dunlop/esouth/dnadesign/.venv`), run a real Evo infer smoke execution, and record operator-facing hardening updates.

### Session and resource evidence

1. Host: `scc-510` interactive session.
2. Scheduler/session vars: `NSLOTS=4`, `JOB_ID=3655013`, `CUDA_VISIBLE_DEVICES=0`.
3. GPU class: `NVIDIA L40S` (`46068 MiB`, compute capability `8.9`).
4. Storage posture:
   - `/projectnb/dunlop` at `91%` utilization.
   - `HF_HOME=/projectnb/dunlop/esouth/cache/huggingface`.
   - Runtime transients routed to `/project/dunlop/esouth/dnadesign/runtime/...`.

### Canonical build execution (actual run)

1. Environment target:
   - `UV_PROJECT_ENVIRONMENT=/project/dunlop/esouth/dnadesign/.venv`
2. Toolchain:
   - `module purge`
   - `module load cuda/12.8`
   - `module load gcc/13.2.0`
3. Build sequence:
   - `uv sync --locked`
   - `uv sync --locked --extra infer-evo2 --reinstall-package flash-attn --reinstall-package transformer-engine-torch`
4. Build evidence:
   - `flash-attn` built from source (sdist path in lock).
   - compile phase finished with: `Prepared 2 packages without build isolation in 69m 30s`.
   - runtime import gate passed:
     - `torch 2.8.0+cu128`
     - `transformer-engine 2.11.0`
     - `flash-attn 2.8.3`
     - `evo2 0.4.0`
     - `vtx 1.0.7`
     - `transformer_engine.pytorch import_ok`
     - `flash_attn import_ok`
     - `evo2 import_ok`
5. Build log path:
   - `/project/dunlop/esouth/dnadesign/runtime/infer-evo2-canonical-20260307-110609/build.log`

### Real infer smoke execution (model call)

1. Command:
   - `uv run infer extract --model-id evo2_7b --device cuda:0 --precision bf16 --alphabet dna --batch-size 1 --fn evo2.log_likelihood --format float --seq ACGTACGTACGT --no-progress`
2. Result:
   - succeeded (`Outputs for job 'adhoc_extract'`, `out` count `1`, type `float`).
3. Model cache outcome:
   - `evo2_7b` downloaded and cached:
     - `/projectnb/dunlop/esouth/cache/huggingface/hub/models--arcinstitute--evo2_7b` -> `13G`
   - existing `evo2_40b` cache retained:
     - `/projectnb/dunlop/esouth/cache/huggingface/hub/models--arcinstitute--evo2_40b` -> `77G`

### 40B capacity behavior in this current allocation

1. Validation check on `evo2_40b` with `cuda:0` in this session fails fast:
   - `CAPACITY_FAIL model_id=evo2_40b precision=bf16 required_gib=93.1 usable_gib=40.0`
   - `required_gpus=1 gpus_available=1`
2. Interpretation:
   - with current single visible L40S (`CUDA_VISIBLE_DEVICES=0`) this contract is intentionally blocked.
   - larger/multi-GPU topology must be requested and declared for 40B runtime.

### Runtime bug found during smoke and fixed (TDD)

1. Observed failure on first smoke attempt:
   - `get_adapter() takes 0 positional arguments but 1 was given`
2. Root cause:
   - `runtime.adapter_runtime.get_adapter` had keyword-only `model` while engine/existing call sites pass positional model.
3. TDD slice:
   - added failing regression test:
     - `src/dnadesign/infer/tests/runtime/test_adapter_runtime.py::test_get_adapter_accepts_positional_model_argument`
   - fixed signature:
     - `src/dnadesign/infer/src/runtime/adapter_runtime.py`
     - from `def get_adapter(*, model: ...)` to `def get_adapter(model: ...)`
   - reran targeted tests: pass.

### UV operator hardening finding

1. `uv sync --locked --group dev` without `--extra infer-evo2` removes GPU infer packages from the same environment realization.
2. Hardened guidance now requires combined command when both are needed:
   - `uv sync --locked --group dev --extra infer-evo2`

### Verification commands run

1. `uv run pytest -q src/dnadesign/infer/tests/runtime/test_adapter_runtime.py -k positional`
2. `uv run pytest -q src/dnadesign/infer/tests/runtime/test_adapter_runtime.py src/dnadesign/infer/tests/cli/test_run_command_config_inputs.py src/dnadesign/infer/tests/cli/test_validate_command.py`
3. real smoke command listed above (`infer extract` with `evo2_7b`).

## 2026-03-07 - Phase 2 Slice AA (Transient Path Policy Cleanup + Cache Placement Clarification)

### Goal

Enforce repo information architecture by removing top-level transient sprawl and hardening where infer GPU build/runtime artifacts are written.

### Findings

1. Top-level `runtime/` contained UV/flash-attn/triton transient artifacts from environment build and smoke runs.
2. Current model caches observed in this environment:
   - `evo2_7b` under current `HF_HOME` (~13G)
   - `evo2_40b` under current `HF_HOME` (~77G)
3. With near-full `/projectnb` posture, checkpoint location must be explicit before smoke/pressure runs.

### Decisions and changes

1. Top-level runtime sprawl guard:
   - added `/.gitignore` rule for `/runtime/`.
2. SCC install and infer runbook transient routing now uses workspace-scoped runtime paths:
   - `INFER_WORKSPACE_ROOT=.../src/dnadesign/infer/workspaces/test_stress_ethanol`
   - `INFER_RUNTIME_ROOT=$INFER_WORKSPACE_ROOT/outputs/runtime/evo2-gpu`
3. Model cache root is explicitly exported in docs (`HF_HOME`) so checkpoint placement is deterministic and auditable.
4. Removed repo-root `runtime/` directory from `dnadesign` working tree.

### Operator guidance for next smoke/pressure run

1. Set `HF_HOME` intentionally before run (`/project` vs `/projectnb`) and verify free space.
2. Keep build/runtime transients under workspace `outputs/runtime/...`.
3. Run 7B smoke first in the selected cache location before pressure tests.

### Verification notes

1. Repo-root layout now has no `runtime/` directory.
2. Documentation exports for infer GPU setup now route transients to workspace-scoped paths.

## 2026-03-07 - Phase 2 Slice AB (Canonical Cache Placement Hardening + Real 7B Smoke)

### Goal

Harden SCC docs and operator sequence so infer uses one canonical repo `.venv`, writes runtime transients under infer workspace outputs, and places 7B cache on `/project` deterministically.

### Root cause found during live smoke

1. Setting `HF_HOME` alone was not sufficient in this SCC session.
2. Inherited environment had `HF_HUB_CACHE` preset to `/projectnb/.../huggingface/hub`.
3. `huggingface_hub.constants.HF_HUB_CACHE` followed that inherited path, so Evo2 loaded from `/projectnb` even when `HF_HOME` was set to `/project`.

### Hardening change

1. Documentation now exports all cache controls together:
   - `HF_HOME`
   - `HF_HUB_CACHE`
   - `HUGGINGFACE_HUB_CACHE`
   - `TRANSFORMERS_CACHE`
2. Policy remains:
   - canonical uv env: `UV_PROJECT_ENVIRONMENT="$PWD/.venv"`
   - infer 7B cache on `/project` (`HF_HOME_7B`)
   - large external Evo2 artifacts on `/projectnb` (`HF_HOME_LARGE`)
   - runtime transients in infer workspace `outputs/runtime/...`

### Live verification (real run)

1. Command class:
   - `uv run infer extract --model-id evo2_7b --device cuda:0 --precision bf16 --alphabet dna --batch-size 1 --fn evo2.log_likelihood --format float --seq ACGTACGTACGT --no-progress`
2. Result:
   - execution succeeded and returned one float output.
3. Deterministic placement evidence:
   - model load path during run:
     - `/project/dunlop/esouth/cache/huggingface/evo2_7b/hub/models--arcinstitute--evo2_7b/.../evo2_7b.pt`
   - cache size after run:
     - `13G /project/dunlop/esouth/cache/huggingface/evo2_7b`

### Cleanup status

1. Removed leftover moved runtime trash directory:
   - `/project/dunlop/esouth/.trash/dnadesign-runtime-20260307-123847`
2. Repo root remains clean of top-level `runtime/` directory.

## 2026-03-07 - Phase 2 Slice AC (Evo2 API Pressure Pass + Pooling/Embedding Contracts)

### Goal

Pressure-test infer Evo2 usage for forward/logits, embeddings, and generation; harden contracts so behavior is explicit and easier to extend.

### Audit finding (root cause)

1. Pooling semantics drifted between batched and variable-length fallback paths in `Evo2Adapter`:
   - batched path pooled with `pool.dim` default `1` on `[B,L,V]` / `[B,L,D]`.
   - variable-length fallback pooled after `squeeze(0)` with default dim `0` on `[L,V]` / `[L,D]`.
2. Practical effect:
   - with explicit `pool.dim=1` (documented sequence-dimension pooling), fallback path pooled the feature axis instead of sequence axis.
3. Additional UX footguns:
   - `evo2.embedding` without `layer` could pass request assembly and fail later at adapter invocation.
   - `pool.dim=0` could consume batch axis and violate one-output-per-input contract.

### Changes applied

1. Refactored Evo2 pooling path to one contract:
   - added pool normalization + batch-preserving pooling helpers in `src/dnadesign/infer/src/adapters/evo2.py`.
   - both batched and fallback paths now interpret `pool.dim` against batched tensors.
   - fallback path pools before dropping batch axis, removing semantic drift.
2. Added explicit fail-fast pooling contract:
   - reject `pool.dim < 1` with clear `CapabilityError`.
3. Hardened embedding extraction contract:
   - CLI request assembly now fails fast when `fn=evo2.embedding` and `layer` is missing.
   - runtime adapter dispatch now enforces non-empty `params.layer` for `embedding`.
4. Added pressure-check runbook section:
   - `src/dnadesign/infer/docs/operations/scc-evo2-gpu-uv-runbook.md` now includes forward/logits, embeddings (layered), and generation checks.

### TDD evidence

1. Added failing tests first:
   - `src/dnadesign/infer/tests/runtime/test_evo2_adapter_pooling_contracts.py`
   - `src/dnadesign/infer/tests/runtime/test_adapter_dispatch.py` (embedding layer missing)
   - `src/dnadesign/infer/tests/cli/test_requests.py` (embedding layer requirement)
2. Implemented minimal fixes.
3. Re-ran targeted tests and full infer tests: pass.

### Live pressure checks (GPU)

1. Executed real 7B API checks on `cuda:0` in canonical `.venv`:
   - logits mean pooling (`evo2.logits`, `pool.dim=1`) for mixed-length inputs
   - embedding extraction (`evo2.embedding`, `layer=blocks.28.mlp.l3`, `pool.dim=1`)
   - generation (`max_new_tokens=4`)
2. Observed outputs:
   - `logits_widths [512, 512]`
   - `embedding_widths [4096, 4096]`
   - generation produced one continuation sequence.
3. Fail-fast adversarial checks:
   - `pool.dim=0` now raises explicit `CapabilityError`.
   - embedding request without `layer` now fails fast in CLI request validation.

### Verification commands

1. `uv run pytest -q src/dnadesign/infer/tests/runtime/test_evo2_adapter_pooling_contracts.py src/dnadesign/infer/tests/runtime/test_adapter_dispatch.py src/dnadesign/infer/tests/cli/test_requests.py`
2. `uv run pytest -q src/dnadesign/infer/tests`
3. `uv run pytest -q src/dnadesign/densegen/tests/docs/test_bu_scc_docs_contracts.py src/dnadesign/infer/tests/docs/test_scc_gpu_env_docs_contract.py src/dnadesign/infer/tests/docs/test_information_architecture_contracts.py src/dnadesign/infer/tests/docs/test_pressure_runbook_docs_contract.py`

## 2026-03-07 - Phase 2 Slice AD (Mean-Pooling Math Contract + Likelihood Reduction Fail-Fast)

### Goal

Validate mean pooling against explicit `1/n` averaging math and ensure Evo2 likelihood API usage has no silent reduction fallback.

### Findings and hardening

1. Mean pooling contract now has value-level tests (not shape-only) for logits and embeddings:
   - verifies pooled vectors equal explicit arithmetic mean over token positions.
2. Likelihood contract tightened:
   - previous behavior mapped unknown reductions to `sum` silently.
   - now invalid reductions fail fast with `CapabilityError`.
3. Evo2 API usage verified against runtime signature introspection:
   - `Evo2.__call__(..., return_embeddings=False, layer_names=None)`
   - `Evo2.score_sequences(..., reduce_method='mean', ...)`

### Verification commands

1. `uv run pytest -q src/dnadesign/infer/tests/runtime/test_evo2_adapter_pooling_contracts.py src/dnadesign/infer/tests/runtime/test_adapter_dispatch.py src/dnadesign/infer/tests/cli/test_requests.py src/dnadesign/infer/tests/docs/test_scc_gpu_env_docs_contract.py`
2. `uv run pytest -q src/dnadesign/infer/tests`

Passed.
