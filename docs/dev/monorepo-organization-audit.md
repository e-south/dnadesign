## dnadesign Monorepo Organization Audit


### Contents
- [Executive summary](#executive-summary)
- [Current structure (observed)](#current-structure-observed)
- [Top-level](#top-level)
- [Tool directories under `src/dnadesign/`](#tool-directories-under-srcdnadesign)
- [Nested `src/` tool layout](#nested-src-tool-layout)
- [CLI entry points (from `pyproject.toml`)](#cli-entry-points-from-pyprojecttoml)
- [Dependency and environment management](#dependency-and-environment-management)
- [Pragmatic programming principles: assessment and recommendations](#pragmatic-programming-principles-assessment-and-recommendations)
- [1) Decoupled design](#1-decoupled-design)
- [2) Easier to change](#2-easier-to-change)
- [3) Assertive programming](#3-assertive-programming)
- [4) Robustness](#4-robustness)
- [5) Extendibility](#5-extendibility)
- [Evaluating the current nested `src/` per tool](#evaluating-the-current-nested-src-per-tool)
- [uv vs pixi for tool aliases and organization](#uv-vs-pixi-for-tool-aliases-and-organization)
- [Recommendations by horizon](#recommendations-by-horizon)
- [Low-risk, near-term](#low-risk-near-term)
- [Medium-term](#medium-term)
- [Long-term (optional)](#long-term-optional)
- [Proposed standard tool layout (if keeping nested `src/`)](#proposed-standard-tool-layout-if-keeping-nested-src)
- [Risks to track](#risks-to-track)
- [Open questions](#open-questions)
- [Suggested next steps](#suggested-next-steps)

### Executive summary

The monorepo is already workable and pragmatic: tools are colocated under a single package namespace, CLI entry points are defined in one place, and uv is the primary dependency manager with a lockfile. The structure is opinionated (tool directories with nested `src/` for code) and provides good UX for tool-level IO and configs. The largest risks are inconsistent sub-structure across tools, global dependency coupling, and weak boundaries between tools (making change harder over time). These are fixable with convention, small reorganizations, and clearer package boundaries.

Top opportunities (highest impact, lowest disruption):
- Document a standard tool layout and enforce it across tools.
- Make boundaries explicit: separate shared libraries from tool apps, and restrict cross-tool imports.
- Use extras/dependency-groups or per-tool environments to reduce global dependency coupling.
- Expand task aliases in uv/pixi for consistent entry points.

### Current structure (observed)

#### Top-level
- Root docs: `docs/`
- Package code: `src/dnadesign/`
- Build/test configuration: `pyproject.toml`
- Environment files: `uv.lock`, `pixi.toml`, `pixi.lock`
- CI: `.github/workflows/ci.yaml` (not opened in this audit)

#### Tool directories under `src/dnadesign/`
Observed tool roots:
- `aligner`, `baserender`, `billboard`, `cluster`, `cruncher`, `densegen`, `infer`, `latdna`, `libshuffle`, `nmf`, `opal`, `permuter`, `tfkdanalysis`, `usr`
- `archived/` and `prototypes/` are present (excluded from tests via `pytest` config)

#### Nested `src/` tool layout
Several tools include their own internal `src/`:
- `baserender`, `cluster`, `cruncher`, `densegen`, `opal`, `permuter`, `usr` (and several under `archived/`)

This creates a layout pattern like:

```
src/dnadesign/<tool>/
  src/
  tests/ (varies)
  configs/ (varies)
  notebooks/ (varies)
  outputs/ (varies)
```

#### CLI entry points (from `pyproject.toml`)
Scripts currently exposed:
- `usr`, `opal`, `dense`, `infer`, `cluster`, `permuter`, `mb`, `baserender`, `cruncher`

These scripts point to modules inside tool-specific `src/` folders (e.g., `dnadesign.cruncher.cli.app:app`).

#### Dependency and environment management
- uv is the primary Python dependency manager with `uv.lock`.
- `pixi.toml` exists with a minimal task list and a `meme` dependency.
- `pyproject.toml` uses a single dependency list shared by all tools.
- Dependency groups exist (`test`, `lint`, `dev`, `notebooks`).

### Pragmatic programming principles: assessment and recommendations

#### 1) Decoupled design

Observations:
- Tools share a common namespace (`dnadesign`), which is convenient but can hide coupling.
- The repository appears to use shared dependencies at the root; tool-specific deps are not isolated.
- Mixed internal layouts (some tool roots, some nested `src/`) can lead to accidental imports across tools.

Risks:
- Cross-tool imports can become implicit dependencies without clear ownership.
- Global dependencies increase coupling and make individual tools harder to evolve.

Recommendations:
- Define a formal boundary between “shared libraries” and “tool apps.”
  - Example: `dnadesign/core/`, `dnadesign/algorithms/`, `dnadesign/io/` as shared libraries; tool packages import only from these shared packages.
- Add a rule: tool-to-tool imports are not allowed except via shared packages.
- If shared code is needed, move it to a common module and document it.

#### 2) Easier to change

Observations:
- Tool roots are grouped, which helps discoverability.
- Nested `src/` creates additional mental mapping and can add path complexity.

Risks:
- Inconsistent structures increase onboarding time and inhibit refactoring.
- Tools with different internal layouts slow down changes across multiple tools.

Recommendations:
- Standardize a tool structure and apply it across all tools:
  - `src/` for code
  - `cli.py` or `cli/` at a predictable location
  - `configs/`, `notebooks/`, `tests/`, `outputs/` with consistent naming
- Add a single “tool layout” reference doc in `docs/` and link it from each tool.
- Create a small template (copyable) for new tools.

#### 3) Assertive programming

Observations:
- The monorepo doesn't currently enforce guardrails at the layout level beyond `pytest` exclusions.

Risks:
- Unclear boundaries allow fragile couplings or ad-hoc structure.

Recommendations:
- Add explicit “public API” modules per tool (for example, `dnadesign.<tool>.api`) to make dependencies explicit.
- Consider enforcing import boundaries with a lint rule or simple test (import graph sanity checks).
- Prefer typed configuration objects (pydantic models already in dependencies) for tool configuration, with validation in CLI entry points.

#### 4) Robustness

Observations:
- Global dependency list increases the chance that tools pick up unused but installed dependencies.

Risks:
- Breakages in one dependency update can affect unrelated tools.

Recommendations:
- Split tool-specific dependencies into extras (e.g., `cruncher`, `opal`, `densegen`) or optional dependency groups.
- For GPU-specific tools, keep GPU dependencies in explicit extras (as already done for `infer-evo2`).
- Build minimal environments per tool to reduce “accidental coupling.”

#### 5) Extendibility

Observations:
- A single package with multiple subpackages is easy to extend initially.

Risks:
- Growth in number of tools can crowd the `dnadesign` namespace.

Recommendations:
- Establish a “tool registry” or list in docs, with pointers to each tool’s entry point and layout.
- If new tools continue to appear, consider a sub-namespace: `dnadesign.tools.<tool>`.

### Evaluating the current nested `src/` per tool

Your current approach (tool root with a nested `src/` for code and adjacent IO/config directories) is a valid and pragmatic pattern. It helps avoid a flat, cluttered top-level and keeps non-code assets next to the tool. The tradeoffs:

Benefits:
- Keeps tool-specific IO/configs close to the tool.
- Reduces noise at top-level `src/dnadesign/`.
- Encourages tool-level ownership and separation.

Costs:
- The inner `src/` is non-standard in a monorepo where the root already uses `src/` layout.
- Import paths are less obvious to new contributors.
- Some Python tooling expects a single `src/` root, which can complicate discovery.

If you keep this pattern, consider standardizing on it and documenting it clearly. If you want a more “typical” modern practice, the two common alternatives are:

1) Single-package layout (flattened):
```
src/dnadesign/
  cruncher/
    __init__.py
    cli.py
    app/
    io/
    tests/
    configs/
```

2) Multi-package workspace (each tool is its own package):
```
packages/
  cruncher/
    pyproject.toml
    src/cruncher/
    tests/
  opal/
    pyproject.toml
    src/opal/
```

Either is common. The single-package layout is simpler to maintain. The multi-package workspace scales better if tools have divergent dependencies or release cycles.

### uv vs pixi for tool aliases and organization

Current state:
- `pyproject.toml` defines all CLIs under `[project.scripts]` and uv runs them via `uv run <script>`.
- `pixi.toml` defines a single task (`cruncher`) and the `meme` dependency.

Recommendations:
- If uv is the primary Python environment manager (as it appears), keep `[project.scripts]` as the canonical tool list.
- Use `pixi` for non-Python/system dependencies (e.g., MEME suite), but expand tasks to mirror the uv scripts for convenience:
  - `pixi run cruncher`, `pixi run opal`, `pixi run dense`, etc.
- Optionally define a “tool runner” task in pixi that accepts an argument to reduce duplication.
- If you want per-tool environments, pixi’s “features” or uv extras can model tool-specific dependencies.

Summary: uv already provides the standard aliasing mechanism for Python entry points. pixi is best used for system-level deps and task shortcuts (especially on clusters).

### Recommendations by horizon

#### Low-risk, near-term
- Add a “Tool Layout Standard” doc with a canonical tree and naming conventions.
- Add a “Tool Registry” doc listing tools, entry points, and where their code lives.
- Establish one shared library namespace (example: `dnadesign/core`) and start moving shared code there.

#### Medium-term
- Introduce tool-specific dependency extras (e.g., `cruncher`, `opal`, `densegen`).
- Add lint checks to discourage cross-tool imports (unless via shared libs).
- Standardize tests layout per tool.

#### Long-term (optional)
- Consider a uv workspace or multi-package layout if tools diverge heavily in dependencies or deployment lifecycle.
- If you adopt multi-package layout, use a shared “dnadesign-core” package for common libraries.

### Proposed standard tool layout (if keeping nested `src/`)

```
src/dnadesign/<tool>/
  AGENTS.md
  README.md
  src/
    __init__.py
    cli.py
    app/
    io/
    core/
    config/
    tests/
  configs/
  notebooks/
  outputs/   # generated only
```

This preserves your current preference while making the internal structure consistent across tools.

### Risks to track

- Tool sprawl without a shared “core” library can lead to duplicated utilities.
- The root dependency list can balloon; extras help control this.
- Archived/prototype code under `src/` can confuse packaging if it becomes importable.

### Open questions

- Which tools are actively maintained vs legacy? This affects how strict the layout standard should be.
- Do any tools require conflicting dependency versions? If yes, consider multi-package workspace sooner.

### Suggested next steps

- Decide on a single “tool layout standard” and document it.
- Decide whether to keep the nested `src/` pattern and enforce it consistently.
- Decide whether to split dependencies into tool extras.
