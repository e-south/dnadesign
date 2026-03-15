## Construct Development Journal

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-14

This journal tracks `dnadesign.construct` design and implementation decisions, scope boundaries, and validation notes.

## 2026-03-14 - Phase 0 Kickoff (Create-Plan + MVP Spec)

### Semantic objective

`construct` is the sibling tool that realizes new DNA sequences from:

- anchor sequences from USR
- template context such as plasmids or other backbone sequences
- explicit placement and windowing rules

The realized sequences are first-class USR records. The realization provenance is stored as `construct__*` lineage metadata.

### Plan intent summary

Ship a real first implementation of `construct` as a sibling tool with a layout consistent with other dnadesign packages and a usable MVP workflow, not a placeholder scaffold.

### Explicit scope

- In scope:
  - new `construct` package with lightweight top-level surface
  - config-driven run path from USR anchors to derived USR records
  - template loading from literal sequence or file
  - multi-part placement on a shared template, with first-class support for one or many placed parts
  - windowed extraction around a focal part, plus full-construct output mode
  - construct lineage overlays in USR
  - CLI validate and workspace scaffolding
  - targeted tests for package layout, config validation, workspace behavior, and runtime realization
- Out of scope:
  - genbank parsing, rich plasmid annotation import, or graphical map manipulation
  - generalized edit history or resume/retry orchestration
  - infer-specific shortcuts or cross-tool internal imports

### Boundary and contract decisions

- `construct` owns realization logic only.
- `usr` remains the canonical home for resulting sequence records.
- `infer` remains downstream and consumes construct outputs as ordinary USR sequences.
- `construct` must use public USR APIs plus the documented `registry.yaml` artifact contract, not internal `usr.src.*` imports.
- Missing template files, missing input fields, overlapping placements, invalid window bounds, and duplicate realized IDs must fail fast.

### MVP config contract

```yaml
job:
  id: promoter_context_1kb
  input:
    source: usr
    dataset: densegen_anchor_set
    field: sequence
  template:
    id: plasmid_demo
    path: inputs/template.fa
    circular: true
    source: inputs/template.fa
  parts:
    - name: anchor
      role: anchor
      sequence:
        source: input_field
        field: sequence
      placement:
        kind: replace
        start: 100
        end: 160
        orientation: forward
        expected_template_sequence: REPLACE_WITH_TEMPLATE_INTERVAL
  realize:
    mode: window
    focal_part: anchor
    focal_point: center
    anchor_offset_bp: 0
    window_bp: 1000
  output:
    dataset: densegen_anchor_set_construct_1kb
```

### Ordered action checklist

1. Create the new sibling-tool layout:
   - top README
   - docs index surfaces
   - dev journal
   - `src/`, `tests/`, and `workspaces/`
2. Add package wiring and CLI entrypoints:
   - `construct run`
   - `construct validate config`
   - `construct workspace where|init`
3. Implement the realization runtime:
   - load config
   - read anchor rows from USR
   - load template sequence
   - apply ordered part placements
   - extract either a focal window or the full construct
4. Persist outputs to USR:
   - create output dataset if needed
   - import realized sequences as base records
   - attach `construct__*` lineage overlay columns
5. Add tests and validate:
   - package layout contracts
   - config validation behavior
   - workspace scaffold behavior
   - runtime realization for linear and circular cases

### Validation and risk handling

- Required checks:
  - targeted `pytest` for new construct tests
  - `ruff check` on touched construct files
- Risk controls:
  - strict error behavior only; no hidden fallback to alternate paths or sequence fields
  - overlap and duplicate realized-sequence collisions are hard errors
  - registry bootstrap is explicit and deterministic for the output USR root

### Open questions for later phases

- When richer plasmid-map sources arrive, decide whether to add a dedicated parser layer or keep maps external to the first construct contract.
- Decide whether future multi-part constructs should support cross-boundary replacement spans on circular templates or keep first-class wrap only at the extraction stage.

## 2026-03-14 - Phase 1 Tracer Bullet (Promoter Swap + Control Anchors)

### Objective

Harden `construct` around the first real biological use case:

- control/wild-type promoter anchors that live in USR beside DenseGen-derived anchors
- a circular plasmid/scaffold template that contains an incumbent promoter interval
- coordinate-driven replacement of that incumbent interval
- fixed-length realized windows for downstream `infer`

### Contract refinements

- Placement semantics are explicit:
  - `kind: insert` requires `start == end`
  - `kind: replace` requires `end > start`
- Replacement specs may declare `expected_template_sequence`.
  - Runtime verifies the template interval before constructing outputs.
  - This keeps the contract coordinate-based while still protecting against wrong-incumbent swaps.
- `construct validate config --config <path> --runtime` now resolves the template, input dataset, and projected output lengths without writing USR data.

### Control-anchor dataset stance

- Control promoters and wild-type promoters are ordinary USR anchor records, not DenseGen exceptions.
- The workspace scaffold now includes `inputs/anchor_manifest.template.yaml` as a worksheet for:
  - `spyP_MG1655`
  - `sulAp`
  - `soxS`
  - `J23105`
  - optional `pDual-10` template reference metadata

### Fidelity rule for pasted plasmid sequence

- Long plasmid DNA pasted into chat is not treated as canonical tracked sequence content.
- The workspace scaffold keeps placeholders and notes instead of committing `pDual-10` sequence text from chat.
- Real runs should use a canonical FASTA file and, when possible, a checksum recorded in the workspace manifest.

### Tracer-bullet example assumptions captured from discussion

- The first 1 kb promoter-context example is a scaffold-before-insertion template.
- The incumbent promoter interval is replaced, not substring-discovered at runtime.
- The discussion-derived incumbent `J23105` interval on the 1 kb snippet is zero-based half-open `[405, 440)`.
- Downstream fixed-length context windows should treat `pDual-10` as circular.

## 2026-03-14 - Phase 2 Tracer Bullet (USR-Native pDual-10 Demo Flow)

### Objective

Promote the first real promoter-swap flow from a file-backed placeholder into a USR-native workflow:

- control anchors are seeded into a local USR dataset
- `pDual-10` is seeded as its own USR template record
- `construct` resolves the template from USR, not a FASTA side file
- packaged workspaces document both 1 kb-window and full-plasmid realization flows

### Decisions

- `pDual-10` is treated as a first-class USR record, separate from the anchor dataset.
- The tracer-bullet bootstrap keeps anchor and template datasets separate.
- The provided full `pDual-10` record contains two exact `J23105` matches:
  - `slot_a`: `[2300, 2335)`
  - `slot_b`: `[3621, 3656)`
- The earlier scaffold-only interval `[405, 440)` does not apply to the full `pDual-10` record and must not be reused blindly in demo configs.

### CLI surface additions

- `construct seed promoter-swap-demo [--root <usr-root>] --manifest <path>`
  - bootstraps curated anchor/template demo datasets
  - defaults to the canonical repo USR datasets root when `--root` is omitted
  - attaches `construct_seed__*` catalog overlays for human-readable labels
  - attaches standardized `usr_label__primary` / `usr_label__aliases` label fields and materializes those into `records.parquet`
  - writes a manifest with deterministic record ids and slot coordinates
- `construct workspace init --profile promoter-swap-demo`
  - copies a packaged demo workspace with slot_a/slot_b and window/full configs

### Validation contract

- Runtime preflight now reports template provenance in enough detail to distinguish file/literal vs USR-backed templates.
- Runtime preflight reports resolved `input_root` and `output_root` so canonical-vs-workspace placement is visible before writes.
- Demo flow validation remains:
  - `construct validate config --config <path> --runtime`
  - `construct run --config <path> --dry-run`
  - `usr validate <dataset> --strict`

### SCC sync stance

- Pull-only remote bootstrap belongs to the existing `usr` CLI contract, not a new private construct-side sync implementation.
- For the tracer-bullet runbooks, `construct` references the `usr diff` / `usr pull` loop and avoids SCC mutation or deletion.
- Current SCC inventory also includes `datasets/mock_dataset`, but it is only an empty directory there and should not be treated as a pullable USR dataset until it contains `records.parquet`.
- The USR remote-lock harness now tolerates benign shell noise before the lock marker, which removes the SCC-specific `AGENT_MANAGE_RUNTIME_SKILLS=1` handshake failure from the normal `usr pull` path.
- The USR rsync harness intentionally avoids replaying remote owner/group/permission metadata on local destinations, so SCC pulls remain portable across constrained local filesystems while preserving content and sidecar fidelity.

### Input ontology refinement

- Curated biological input datasets should use the least-coupled semantic ids possible at the USR layer, such as `mg1655_promoters` and `plasmids`, because `construct` assigns anchor/template role in config rather than by dataset path.
- Human-readable sequence names belong in a standardized `usr_label__*` overlay contract owned by `usr`, while construct-specific bootstrap provenance remains in `construct_seed__*`.
- Realized tracer-bullet outputs should also use flat semantic USR dataset ids such as `pdual10_slot_a_window_1kb_demo`, rather than a tool-owned dataset namespace, so downstream tools see biological products instead of construct-internal routing labels.

## 2026-03-14 - Audit Hardening (Workspace Registry + Collision Policy)

### Trigger

Pressure testing against real demo flows and a bounded review swarm surfaced three contract gaps:

- preflight did not catch output-id collisions that would later fail at run time
- construct metadata attach blocked append-to-existing-output flows even when the new rows were distinct
- workspaces lacked a first-class registry surface for multi-project provenance and config inventory

### Decisions

- Runtime preflight now fingerprints full construct provenance more accurately:
  - selected `input.ids`
  - resolved input/output roots
  - output dataset and collision policy
- `template.kind=path` now rejects multi-record FASTA input instead of concatenating records silently.
- Equal-coordinate part execution now preserves config order in both realization and recorded lineage order.
- Construct output datasets now support explicit collision policy:
  - `output.on_conflict=error` remains the fail-fast default
  - `output.on_conflict=ignore` supports idempotent reruns or selective append flows
- Writing output to the same dataset/root as input is blocked unless `output.allow_same_as_input=true`.
- Every workspace now carries `construct.workspace.yaml` as a project registry so multi-template or multi-slot studies stay auditable as multiple explicit config entries.

### Workspace stance refinement

- Packaged construct workspaces now default to workspace-local `outputs/usr_datasets`, consistent with repo-wide workspace-scoping guidance.
- Shared repo or external USR roots remain allowed, but only through explicit config `root:` fields or explicit `construct seed --root ...` usage.

### Validation additions

- Runtime tests now cover:
  - multi-record FASTA rejection
  - spec fingerprint differentiation for selected input ids
  - equal-coordinate ordering determinism
  - same-dataset guardrails
  - append-to-existing-output behavior
  - preflight collision detection and `on_conflict=ignore`
- CLI/workspace tests now cover:
  - workspace registry creation
  - `workspace show`
  - blank-vs-demo next-step messaging

## 2026-03-14 - Contract Hardening (Workspace Doctor + Manifest Import)

### Trigger

The next maintainer audit pass still exposed three pragmatic gaps:

- `template.kind=path` could leak raw filesystem exceptions instead of a shaped construct error
- `construct.workspace.yaml` described projects but did not enforce registry/config alignment
- construct-owned onboarding still stopped at the packaged demo instead of supporting generic user-provided inputs

### Decisions

- Path-backed templates now fail as construct validation errors when the resolved path is not a readable file.
- Runtime preflight now prints realization settings plus the declared placement contract so coordinate debugging stays in the CLI instead of forcing a YAML diff loop.
- Registry hardening now validates existing `construct`, `construct_seed`, `usr_label`, and `usr_state` column types instead of only appending missing fields.
- `construct workspace doctor` is the new contract check for registry/config drift.
- `construct workspace validate-project` and `construct workspace run-project` now resolve configs by project id, so construct workspaces can be operated as project registries instead of loose config directories.
- `construct seed import-manifest` now materializes arbitrary anchor/template datasets into USR using a manifest with flat semantic dataset ids and explicit record labels.

### Ontology stance

- USR dataset ids remain biological collections such as `mg1655_promoters`, `plasmids`, or user-defined flat semantic ids.
- `anchor` and `template` stay construct-role terms in config and seed manifests, not mandatory USR path taxons.
- Human-readable labels stay in `usr_label__*`.
- Construct bootstrap provenance now stays in `construct_seed__*`, including `manifest_id` and `source_ref`, so imported datasets can be audited without encoding construct routing into dataset ids.

### Validation additions

- CLI tests now cover:
  - direct `construct run` dry-run and write paths
  - shaped template path I/O failures
  - `workspace doctor`
  - `workspace validate-project`
  - `workspace run-project`
  - generic `seed import-manifest`
- Runtime tests now cover:
  - registry type-drift rejection before output writes

## 2026-03-14 - Placement Order And Docs Tightening

### Trigger

The next adversarial audit pass found that same-start mixed placements could be accepted while `validate --runtime`
described a different order than the one later persisted in construct lineage.

### Decisions

- Same-start placements with different template intervals now fail fast as ambiguous instead of relying on implicit ordering.
- Runtime preflight now reports placements in the same execution order used for realized lineage.
- Added a short getting-started doc and tightened workspace/runbook docs around root precedence and safe-to-edit files.

### Validation additions

- Runtime tests now cover:
  - negative circular `anchor_offset_bp`
  - equal-coordinate preflight/lineage order consistency
  - same-start mixed-interval rejection
- CLI validation tests now cover the shaped same-start ambiguity error.
