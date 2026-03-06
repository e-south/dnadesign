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
