## Sequence View Ontology and Infer Completion Hardening Spec

**Status:** active implementation reference
**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-28
**Primary study:** `stress_ethanol_cipro_growth`

### Intent

Harden the USR, Construct, Infer, Study, and Ops contract for sequence-view
semantics, reference promoter products, reverse-complement contexts, and feature
completion. The goal is to keep the current promoter study shippable while
moving the underlying ontology toward reusable sequence-product semantics that
can support future non-promoter studies.

This spec consolidates the recent audits into one implementation reference. It
does not authorize a broad rewrite. It defines the smallest pragmatic contract
changes needed to make the system decoupled, easier to change, fail-fast,
explicit, and future-proof.

### Scope

In scope:

- USR sequence-view ontology, sidecar semantics, and QA checks.
- Construct reference-core and template-context contracts.
- Infer sequence-view planning, feature reuse, feature aliases, and completion
  behavior.
- Study and Ops reporting for product-kind, orientation, pooling, and coverage.
- Documentation and harness checks that keep the record plane honest.

Out of scope:

- Rerunning all existing Infer features by default.
- Replacing current promoter study configs in one change.
- Making Infer synthesize missing sequence products.
- Renaming every promoter-shaped class or enum immediately.
- Committing generated USR Parquet datasets without explicit sync/publish policy.

### Implementation Status

This spec now covers both intended contracts and the implemented hardening
slices through the generic product-kind cutover. Treat the status below as the
handoff boundary for the next engineering pass.

Implemented in the current working tree:

- USR sequence-view identity and sidecar IO.
- USR mutable view-semantics addendum sidecar at `_views/view_semantics.parquet`.
- USR sequence-view QA helper for expected counts, orientations, poolings,
  product kinds, exact product lengths, and emitted anchor bounds.
- Construct reference-core and forward/reverse-complement context behavior in
  the active study workspace branch.
- Infer sequence-view input loading, pooling validation, feature alias/vector
  persistence, and feature-vector de-duplication.
- Infer sequence-view completion planner exposed through
  `uv run infer validate sequence-view-completion --config <config.yaml>
  --format json`.
- Infer completion planner emits first-class `missing_products` evidence for
  zero-row sequence-view selectors instead of forcing operators to infer product
  gaps from command failure.
- Infer completion validation supports explicit thresholds for missing feature
  vectors, stale vectors, and missing sequence products; sequence-view batch
  runbook plans now gate `missing_products=0` and `stale_vectors=0` before
  submit while still allowing missing feature vectors as the batch workload.
- Study prose describing native references, `analysis_window`, full 1 kb
  contexts, reverse-complement contexts, and `anchor_mean`.
- Hard-cut generic product-kind vocabulary:
  `source_record`, `selected_region`, `construct_insert`, `analysis_window`, and
  `realized_context`.
- Ops/preflight `sequence_view_contract` checks that consume USR sequence-view
  QA for product-kind, orientation, pooling, length, and anchor-bound evidence.
- Ops/preflight `infer_sequence_view_completion` checks that run the Infer
  completion planner and surface reusable/stale/missing feature-vector counts.
- `promoter-study-status` now exposes cheap sequence-view product-contract
  summaries, generated-artifact freshness, and Infer sequence-view
  feature-completion summaries in structured JSON fields.
- Active study `_views/sequence_views.parquet` sidecars have been migrated from
  legacy product-kind names to the generic vocabulary with recomputed `view_id`
  values:
  `promoter_insert -> construct_insert`,
  `analysis_core60 -> analysis_window`,
  `biological_insert -> selected_region`, and
  `context1kb_* -> realized_context` plus explicit orientation.
- Active study `_views/view_semantics.parquet` addenda have been materialized
  for `472029` sequence views with `source_family`, `selection_basis`,
  `view_collections`, and `role_tags`.

Still open:

- Full active-study feature completion remains open after the safe legacy
  migration slice: anchor and forward-context legacy-compatible vectors have
  been sidecar-backfilled locally and their duplicated legacy embedding payload
  columns have been retired from the row-overlay parts. New or previously
  uncovered rows still require Infer execution.
- The planner can prove digest-matching legacy overlays reusable, but it does
  not yet prove sequence/model/pooling identity when legacy digest metadata is
  absent or drifted.
- Dataset-local `_views/*` and `_derived/infer/*` sidecars are current in this
  checkout, but they remain generated artifacts. Another checkout still needs
  USR sync/publish handling before assuming the migrated product names,
  recomputed `view_id` values, semantic addenda, and feature alias/vector
  sidecars are present.
- Notify profile/runbook execution still needs operator environment setup
  (`NOTIFY_WEBHOOK` or `NOTIFY_WEBHOOK_FILE`, TLS certificate path, and
  file-backed secrets) before next-run preflight is submit-ready.
- Full repo validation remains pending; current evidence is focused tests plus
  docs checks.

Latest audit delta:

- The Infer completion planner and Ops/preflight integration are now implemented
  enough to expose product-missing, stale, missing, and reusable states before
  model execution.
- Generated product sidecar freshness is now green locally:
  `promoter-study-status --json` reports `sequence-view product contracts 4/4 ok`
  and `generated_artifact_freshness.state=ok`.
- Generated view-semantics addenda are now populated locally for active study
  datasets: `157279` merged-anchor rows, `314558` merged-context rows, `48`
  reference source rows, `48` reference analysis-window rows, and `96` reference
  realized-context rows.
- Feature completion now reaches real sequence-view selectors instead of failing
  during sidecar load. After local legacy alias/vector backfill, the planner
  reports `missing_products=0`, `314328` reusable main 7B sequence-view feature
  vectors, `157509` missing main 7B vectors, and `144` missing reference 7B
  vectors. The remaining main work is `115` anchor vectors, `115`
  forward-context vectors, and `157279` reverse-complement context vectors.
- Duplicated legacy embedding payload columns have been retired after canonical
  sidecar protection was verified. `usr_prom_eth_cip_anchor` no longer carries
  the duplicate legacy `anchor_only_7b_features` intermediate-embedding payload
  column, and `construct_prom_eth_cip_context` no longer carries the duplicate
  legacy `template_1kb_7b_features` anchor-mean intermediate-embedding payload
  column. The row-overlay parts remain only for non-retired legacy evidence and
  metadata.
- Completion/status code now uses key-only parquet inventory for
  `_derived/infer/feature_vectors.parquet` and does not load embedding payload
  columns during normal planner or status checks. Payload reads are reserved for
  explicit migration `--write` or `--verify-payloads` operations.
- Sequence-view Infer runbook plans now include the completion planner as a
  product/staleness gate before `infer run --dry-run`. The gate uses
  `--max-missing-products 0 --max-stale-vectors 0` and intentionally does not
  set `--max-missing-vectors 0`.
- The cheap `promoter-study-status` snapshot now summarizes product-contract
  and feature-completion coverage, while `promoter-study-preflight` remains the
  authoritative next-run readiness surface.

### Core Decision

`product_kind` describes sequence-product lineage, not cohort membership and not
every downstream analysis role.

Consequences:

- A native or designed exact-60 sequence is not automatically `analysis_window`.
- `analysis_window` means Construct derived a 60 bp analysis-only comparability
  view by an explicit focal rule.
- The merged anchor handoff can expose every row as `construct_insert` while
  preserving analysis-only lineage for rows that came from reference core60.
- `core60_mean`, `anchor_mean`, and `seq_mean` are Infer pooling operations, not
  sequence-product kinds.
- Equivalent feature vectors should be deduplicated through Infer feature aliases,
  not by lying in USR product identity.

### Existing Evidence

| Surface | Observed fact | Evidence |
| --- | --- | --- |
| USR semantic identity | `view_id` is computed from a stable semantic key that includes `sequence_id`, `source_dataset_id`, `product_kind`, parent ids, derivation spec, source interval, anchor bounds, orientation, template ids, and `analysis_only`; labels are mutable. | `src/dnadesign/usr/src/sequence_views/models.py` |
| USR sidecar validation | Missing sequence ids, invalid bounds, parent-id misses, alias conflicts, and semantic collisions fail before write. | `src/dnadesign/usr/src/sequence_views/store.py` |
| USR view semantics | Mutable `source_family`, `selection_basis`, `view_collections`, and `role_tags` are stored in a separate addendum keyed by `view_id`, not in the stable semantic hash. | `src/dnadesign/usr/src/sequence_views/semantics.py` |
| USR view-semantics materializer | Active promoter-study datasets can be populated with mutable provenance, selection-basis, collection, and role-tag addenda without changing sequence-view identity. | `src/dnadesign/usr/scripts/materialize_promoter_study_view_semantics.py` |
| USR sequence-view QA | Dataset-local QA can assert counts by product kind/orientation/context/pooling, product lengths, and anchor-bound validity. | `src/dnadesign/usr/src/sequence_views/qa.py` |
| Current product vocabulary | Current values are `source_record`, `selected_region`, `construct_insert`, `analysis_window`, and `realized_context`. | `src/dnadesign/usr/src/sequence_views/models.py` |
| Construct core60 | `normalize_anchor.product_kind` is currently constrained to `analysis_window`; generated derived rows carry target length, focal rule, source interval, retention metadata, and `analysis_only=true`. | `src/dnadesign/construct/src/config.py`, `src/dnadesign/construct/src/runtime.py` |
| Construct contexts | Forward and reverse-complement context variants are explicit products; reverse-complement bounds are transformed into emitted-orientation coordinates. | `src/dnadesign/construct/src/runtime.py`, `src/dnadesign/construct/src/orientation.py` |
| Infer sequence-view path | Sequence-view inputs can select by product kind, view name, alias, and orientation; `anchor_mean` requires bounds and `core60_mean` requires exact 60 bp. | `src/dnadesign/infer/src/features/contracts.py`, `src/dnadesign/infer/src/features/sequence_views.py` |
| Infer completion planner | Sequence-view bundles can be classified before model execution as reusable, stale, missing, or missing sequence products; zero-row selectors emit `missing_products` and `missing_product_selectors`. | `src/dnadesign/infer/src/features/completion_planner.py`, `src/dnadesign/infer/src/cli/commands/validate.py` |
| Active Infer planner configs | Main and reference 7B sequence-view completion configs exist for dry-run feature planning. | `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.sequence_views.main.evo2_7b.yaml`, `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.sequence_views.reference.evo2_7b.yaml` |
| Sequence-view runbook gates | Sequence-view Infer runbook plans add completion validation with product/stale thresholds before dry-run and submit planning. | `src/dnadesign/ops/orchestrator/plan_tools.py`, `src/dnadesign/ops/tests/test_runbook_orchestrator.py` |
| Ops completion preflight | `infer_sequence_view_completion` runs the planner command, parses JSON, aggregates reusable/stale/missing/product-missing counts, and applies configured thresholds. | `src/dnadesign/ops/preflight/contract_checks.py`, `docs/studies/stress_ethanol_cipro_growth/ops.study.yaml` |
| Status aggregation | The cheap promoter status snapshot exposes `sequence_view_contract_state` and `infer_feature_completion_state` for product contracts, generated sidecar freshness, and nonblocking feature-completion awareness. | `src/dnadesign/studies/families/promoter/adapter.py`, `src/dnadesign/studies/families/promoter/snapshot.py` |
| Study state | The checked-in study record says the branch is in `infer_batch_preparation`, with 157,279 anchor rows and 314,558 context rows. | `docs/studies/stress_ethanol_cipro_growth/status.md` |
| Coverage gap | Existing Infer overlays are row-based and cover 157,164 ids on the anchor and context handoffs. Missing context coverage includes all reverse-complement contexts. | `docs/studies/stress_ethanol_cipro_growth/status.md` |

### Quality Bar

The implementation should satisfy these engineering constraints:

- **Orthogonal semantics:** product identity, source provenance, selection method,
  study collection membership, and pooling must be independent axes.
- **Fail fast:** zero-row selectors, ambiguous focal annotations, invalid bounds,
  stale semantic collisions, and missing required products must fail before model
  execution or partial writes.
- **Reversible delivery:** add contracts, checks, and sidecars before forcing
  schema-wide renames or deleting legacy paths.
- **Explicit ownership:** USR owns durable sequence views; Construct owns derived
  sequence products; Infer owns feature execution and aliases; Study/Ops owns
  status and readiness.
- **No silent fallback:** fallback from annotation-centered selection to midpoint
  selection, or from missing reverse-complement contexts to forward contexts, is
  never implicit.
- **Generated artifact discipline:** USR Parquet outputs remain generated artifacts
  unless sync/publish policy says otherwise.

### Semantic Axes

The current system has one overloaded axis, `product_kind`, plus several
supporting fields. The hardened contract keeps `product_kind` narrow and adds
machine-readable semantics beside it.

| Axis | Owner | Purpose | Examples | Identity impact |
| --- | --- | --- | --- | --- |
| `product_kind` | USR/Construct | What sequence product is this? | `construct_insert`, `analysis_window`, `realized_context` | In `view_id` semantic hash |
| `orientation` | USR/Construct | What emitted orientation is the sequence in? | `forward`, `reverse_complement` | In `view_id` semantic hash |
| `analysis_only` | USR/Construct | Is this a comparability view rather than a biological source product? | `true` for derived core60 | In `view_id` semantic hash |
| `recommended_pooling` | USR/Construct hint, Infer enforced | How should Infer pool this view? | `seq_mean`, `core60_mean`, `anchor_mean` | Not product identity |
| `source_family` | USR/study manifest | Where did this view come from? | `densegen_generated`, `genbank_reference`, `sfxi_archive`, `construct_derived` | Not product identity |
| `selection_basis` | Construct/USR derivation | How was this region selected or derived? | `native_source_length`, `sigma_site_pair_midpoint`, `template_centered_window`, `whole_output_reverse_complement` | Not product identity unless encoded by derivation spec |
| `view_collections` | Study/Ops | Why are these views in a cohort or run? | `merged_anchor_handoff`, `reference_core60_comparison`, `realized_context_reverse_complement_all` | Not product identity |

Implementation note: do not immediately force these new axes into
`SequenceViewRecord` if that causes broad sidecar migration churn. The smallest
safe step is a companion view-semantics sidecar keyed by `view_id`.

Preferred companion sidecar:

```text
_views/view_semantics.parquet
```

Suggested columns:

```yaml
view_id: string
sequence_id: string
source_family: string|null
selection_basis: string|null
view_collections: list<string>|null
role_tags: list<string>|null
study_id: string|null
created_at: string
created_by: string|null
```

Rules:

- `view_id` must exist in `_views/sequence_views.parquet`.
- `source_family`, `selection_basis`, and `view_collections` are not part of
  `view_id`.
- A change in `view_collections` never changes sequence identity.
- A change in `selection_basis` must be checked against `derived__*` metadata when
  a derived overlay exists.
- Machine selectors may use these fields only after a fail-fast join confirms all
  selected views have a semantics row.

Producer/consumer rules:

- USR owns the addendum store and validates that every addendum row references an
  existing sequence view with the same `sequence_id`.
- Construct may propose `selection_basis` values through derived metadata, but
  Construct should not write study cohort membership directly unless the study
  workflow delegates that responsibility.
- Study/Ops owns `view_collections` because collections answer "why is this view
  in this study cohort?", not "what sequence product is this?".
- Infer may read `view_collections` only as an explicit selector layer after the
  addendum is present and validated; it must still enforce product kind,
  orientation, and pooling bounds.
- `view_semantics.parquet` is mutable metadata. Replacing a row may change
  study membership, but it must not change `view_id`, `sequence_id`, or the
  underlying sequence-view semantic hash.

### Current Product Kinds

Use generic product kinds as the active contract. Domain labels such as
promoter, core60, and pDual-10 live in `context_kind`, derivation metadata,
view-semantics addenda, dataset ids, or study collections, not in
`product_kind`.

| Product kind | Meaning | Allowed producer |
| --- | --- | --- |
| `source_record` | Full source record where the source sequence is the product. | USR import |
| `selected_region` | Source-backed selected interval, often projected from GenBank or curated annotations. | USR import/projection |
| `construct_insert` | Construct-ready insert/anchor row in a merged handoff. | USR merge/materializer |
| `analysis_window` | Construct-derived analysis-only comparability window. | Construct normalize-anchor |
| `realized_context` | Template-realized emitted context; orientation and length are separate fields. | Construct context realization |

Forbidden uses:

- Do not emit `analysis_window` for native exact-60 DenseGen, SFXI, or reference
  rows only because length equals 60.
- Do not make Construct emit `construct_insert` unless ownership changes. The
  merged handoff product is currently a USR materializer responsibility.
- Do not encode `anchor_mean` or `core60_mean` as product kinds.

Compatibility note: old sidecars or overlays that still contain
`native_record`, `biological_insert`, `promoter_insert`, `analysis_core60`,
`context1kb_forward`, or `context1kb_reverse_complement` will fail enabled
`sequence_view_contract` checks until regenerated or migrated. Reference
datasets marked `required: false` remain non-blocking until the study record
promotes them.

### Lifecycle Contract

The desired data lifecycle is:

1. USR stores source rows, GenBank annotations, derived overlays, and initial
   sequence views.
2. Construct derives `analysis_window` only for selected reference views and keeps
   source/native rows intact.
3. USR merges DenseGen, SFXI, native references, and derived core60 rows into
   `usr_prom_eth_cip_anchor`.
4. USR materializes one `construct_insert` view for each merged anchor row, with
   `analysis_only=true` only where the row came from derived core60 lineage.
5. Construct realizes full 1 kb pDual contexts for all merged anchor rows and
   emits paired `realized_context` views distinguished by `orientation=forward`
   and `orientation=reverse_complement`.
6. Infer consumes explicit sequence views, computes only missing or stale feature
   vectors, and writes aliases when existing vectors are semantically equivalent.
7. Study/Ops reports product completion and feature completion separately.

### Reference Promoter Contract

Native/reference rows:

- Keep source lengths.
- Keep GenBank provenance, annotations, strength metadata, and derivation
  intervals in USR overlays/sidecars.
- Are biological records or inserts, not corrected 60 bp products.

Analysis-core rows:

- Live in `construct_prom_eth_cip_reference_core60`.
- Must be exactly 60 bp.
- Must be `analysis_only=true`.
- Must be `product_kind=analysis_window` in the source core60 dataset.
- Must carry focal rule, focal confidence, parent id, source interval, and feature
  retention metadata.
- Must fail if one required `sigma70_minus35` or one required `sigma70_minus10`
  annotation cannot be selected, unless an explicit configured fallback is
  present and marked low confidence.

Merged anchor handoff:

- Lives in `usr_prom_eth_cip_anchor`.
- Exposes every row as `product_kind=construct_insert`,
  `context_kind=anchor_only`, `orientation=forward`, and
  `recommended_pooling=seq_mean`.
- Preserves `analysis_only=true` for rows sourced from core60.
- Does not duplicate native exact-60 rows as core60.

### Construct Context Contract

Forward context:

- Construct receives a `construct_insert` row from the merged anchor handoff.
- Construct inserts or replaces it in the pDual-10 template according to the
  workspace config.
- Construct emits the full 1 kb context sequence.
- The anchor bounds identify the inserted promoter span inside the emitted 1 kb
  sequence.

Reverse-complement context:

- Construct reverse-complements the full emitted forward 1 kb context.
- Construct transforms anchor bounds into emitted-orientation coordinates:

```text
reverse_anchor_start_0 = L - forward_anchor_end_0
reverse_anchor_end_0 = L - forward_anchor_start_0
```

- Infer must consume those emitted-orientation bounds directly and must not apply
  the transform again.

`anchor_mean` meaning:

- Infer runs the model on the full emitted context sequence.
- Infer mean-pools model features over `[anchor_start_0, anchor_end_0)`.
- `anchor_mean` does not mean the input sequence is truncated before inference.

### Infer Completion Contract

Infer must distinguish four states:

| State | Meaning | Owner |
| --- | --- | --- |
| Product missing | Required sequence view or context product is absent. | USR/Construct |
| Feature missing | Product exists, but feature vector does not. | Infer |
| Feature reusable | Existing vector matches sequence, model, layer, pooling, bounds, and orientation. | Infer |
| Feature stale | Existing vector exists, but semantic identity or digest does not match. | Infer |

Current planner state model:

| Planner field | Meaning |
| --- | --- |
| `persisted_vector_reusable` | Required feature-vector key already exists in `_derived/infer/feature_vectors.parquet`. |
| `legacy_digest_reusable` | Legacy row overlay has the requested output and a matching `metadata__feature_request_digest`. |
| `legacy_unclassified_vectors` | Legacy row overlay has a value, but lacks enough matching digest evidence to prove semantic identity. |
| `stale_vectors` | Existing values are present but cannot be used safely without alias migration or recomputation. |
| `missing_vectors` | No usable feature vector was found for the requested view/representation/pooling. |
| `missing_products` | Required sequence products are absent. The planner emits a plan object with `missing_products`, `missing_product_selectors`, and construct-completion command hints instead of requiring operators to infer product gaps from command failure. |

Infer must not create missing sequence products. Missing `analysis_window` rows or
missing reverse-complement contexts are Construct/USR completion failures, not
Infer fallback opportunities.

Reusable feature identity:

```text
model_family
+ model_revision
+ layer or representation selector
+ exact emitted sequence
+ pooling operation
+ pooling bounds
+ emitted orientation
```

If this identity matches, old row-based overlays may be reusable even if they lack
new sequence-view metadata fields. In that case, the migration should write
aliases or semantic metadata, not recompute the model by default.

Naive rerun risk:

- Current Infer resume logic checks `metadata__feature_request_digest`.
- If the digest payload changed to include sequence-view fields, a rerun may
  recompute vectors that are scientifically reusable.
- Add a classification planner before large reruns.

Required planner output:

```yaml
dataset: string
bundle_id: string
model_family: string
required_views: int
required_vectors: int
existing_vectors: int
reusable_vectors: int
stale_vectors: int
missing_vectors: int
missing_products: int
missing_product_selectors: list
persisted_vector_reusable: int
legacy_digest_reusable: int
legacy_unclassified_vectors: int
existing_aliases: int
by_product_kind: map
by_orientation: map
by_pooling_operation: map
commands:
  construct_completion: list<string>
  infer_backfill: list<string>
  alias_backfill: list<string>
```

Operational command:

```bash
uv run infer validate sequence-view-completion \
  --config <config.yaml> \
  --job <optional-job-id> \
  --format json
```

Planner limitations:

- Use `--max-missing-products 0 --max-stale-vectors 0` as the submit gate for
  sequence-view batch runbooks. Do not use `--max-missing-vectors 0` for a lane
  that is expected to generate missing features.
- Digest-matching legacy overlays can be classified as reusable.
- Drifted or missing legacy digests are not yet sequence-audited; they are
  treated as stale/unclassified to avoid accidental overclaiming.
- The planner validates sequence-view selectors and pooling spans by loading
  sequence-view records, but it does not run model adapters or allocate GPU
  resources.
- The planner reports Infer feature completion plus product-missing selectors. If
  a selector resolves zero rows, the plan emits `missing_products`; completion
  still routes back to USR or Construct rather than Infer fabricating products.

### Study and Ops Contract

Study/Ops must report product completion and feature completion separately.

Current row-count status is insufficient because one anchor can produce multiple
sequence products:

- anchor-only view,
- forward 1 kb context,
- reverse-complement 1 kb context,
- optional analysis core60 view,
- multiple feature vectors by pooling/model/layer.

Add status sections or JSON fields for:

- sequence-view sidecar presence by dataset,
- counts by `product_kind`,
- counts by `orientation`,
- counts by `recommended_pooling`,
- anchor-bound validity counts,
- feature coverage by product kind, orientation, pooling, model, and layer,
- reusable/stale/missing feature vector counts,
- generated dataset sync posture.

Minimum product-completion checks for the active study:

| Dataset | Required check | Blocking behavior |
| --- | --- | --- |
| `usr_prom_eth_cip_anchor` | records count equals `construct_insert` forward view count | Blocks construct/infer handoff if required. |
| `construct_prom_eth_cip_context` | forward and reverse-complement context counts each equal anchor count | Blocks reverse-complement context Infer lanes if missing. |
| `construct_prom_eth_cip_reference_core60` | every row is length 60, `analysis_window`, `analysis_only=true` | Blocks reference-core feature lanes. |
| `construct_prom_eth_cip_reference_contexts` | 48 forward and 48 reverse-complement context views | Blocks reference context feature lanes. |

Minimum feature-completion checks for the active study:

| Surface | Required check | Interpretation |
| --- | --- | --- |
| Anchor features | Sequence-view planner reports missing/stale counts for `construct_insert` `seq_mean` views | Existing DenseGen vectors may be reusable; new SFXI/reference/core rows need completion. |
| Forward context features | Planner reports missing/stale counts for `realized_context` `anchor_mean` views with `orientation=forward` | Covered legacy forward vectors may be reusable if digest identity matches. |
| Reverse-complement context features | Planner reports missing/stale counts for `realized_context` `anchor_mean` views with `orientation=reverse_complement` | Existing legacy forward overlays must not be treated as reverse-complement coverage. |
| Reference core/context features | Planner reports reference branch separately from the main DenseGen handoff | Missing planned reference features stay non-blocking unless the study record marks them required. |

Study prose must use full terms:

- Write `reverse-complement`, not vague `RC`, in semantic names and prose.
- Write `anchor_mean` as a pooling operation over emitted anchor coordinates, not
  as a sequence truncation.
- Write `analysis_window` as an analysis-only derived view, not as the native
  promoter.

Current reporting split:

- `usr.data-plane.promoter-study-preflight` is the authoritative next-run
  readiness surface for sequence-view product contracts and Infer completion
  planner output.
- `usr.data-plane.promoter-study-status` remains the cheap record-plane snapshot;
  it now exposes `sequence_view_contract_state` and
  `infer_feature_completion_state` as structured JSON summaries. Those fields are
  situational-awareness summaries, not a replacement for preflight when an
  operator needs command-level blockers.

### Required Harness Endpoints

#### `knowledge-integrity`

Inputs:

- `docs/studies/stress_ethanol_cipro_growth/status.md`
- `docs/studies/stress_ethanol_cipro_growth/datasets.yaml`
- `docs/studies/stress_ethanol_cipro_growth/pipeline.yaml`
- Construct, USR, and Infer reference docs.

Assertions:

- Docs describe the same datasets, counts, product kinds, and planned/present
  posture as `ops progress show`.
- Docs do not claim missing feature outputs are complete.
- Docs distinguish native references, core60 views, forward contexts, and
  reverse-complement contexts.

Evidence:

- `uv run ops progress show usr.data-plane.promoter-study-status --json`
- `uv run python -m dnadesign.devtools.docs_checks --repo-root .`
- focused docs/source diff review.

#### `architecture-invariants`

Inputs:

- USR sequence-view models and store.
- Construct config/runtime.
- Infer sequence-view loaders and execution.
- Study/Ops status normalizers.

Assertions:

- No tool silently fabricates another tool's product.
- Sequence-view selectors fail on zero rows unless explicitly optional.
- Bounds are validated before model execution.
- Reverse-complement bounds follow `L-b, L-a`.
- Native exact-60 rows are not mislabeled as `analysis_window`.
- `product_kind`, `recommended_pooling`, and `view_collections` remain separate
  axes.

Evidence:

- USR sequence-view tests.
- Construct core60/context tests.
- Infer sequence-view and alias tests.
- Study/Ops status tests.
- `uv run python -m dnadesign.devtools.architecture_boundaries --repo-root .`
  when cross-tool imports change.

#### `drift-gc`

Inputs:

- Generated USR datasets.
- Infer overlay parts.
- Sequence-view sidecars.
- Study status snapshots.

Assertions:

- Generated datasets are either synced/published or clearly local.
- Stale overlay registry metadata is distinguishable from schema failure.
- Old row-based feature overlays are classified before deletion or recomputation.

Evidence:

- USR sync diff/audit files.
- Infer feature completion planner output.
- Study status coverage report.

### Implementation Plan

#### Slice 1: Freeze this spec as the reference

Status: implemented.

Deliverables:

- Add this dev spec under `docs/dev/plans/`.
- Link it from follow-up PR descriptions and implementation plans.

Validation:

- `uv run python -m dnadesign.devtools.docs_checks --repo-root .`

#### Slice 2: Add sequence-view semantic addendum support

Status: done as a reusable USR sidecar store and active-study materializer.

Done: local active-study datasets now have `_views/view_semantics.parquet`
addenda for every sequence view. The materializer is idempotent and fails on
unsupported datasets or non-idempotent semantic drift.

Open: generated study addenda still need explicit USR sync/publish handling for
other checkouts or cluster roots.

Deliverables:

- Add a view-semantics sidecar model/store keyed by `view_id`.
- Support `source_family`, `selection_basis`, `view_collections`, and
  `role_tags`.
- Add fail-fast validation that all referenced `view_id` values exist.
- Add a bounded promoter-study materializer that derives mutable addenda from
  sequence views and record sources.
- Keep existing `_views/sequence_views.parquet` schema stable unless a deliberate
  migration is chosen.

Tests:

- Missing `view_id` fails.
- Sequence-id mismatch between addendum row and sequence view fails.
- Re-running the materializer is idempotent.
- Mutating `view_collections` does not change `view_id`.
- If a collection registry is introduced later, unknown collection ids fail fast.
- Unsupported promoter-study datasets fail fast.

#### Slice 3: Add USR dataset QA checks

Status: done as a USR library helper and as an Ops/preflight
`sequence_view_contract` check.

Done: generated active-study sequence-view sidecars have been migrated to the
generic product-kind vocabulary and recomputed `view_id` values. Direct QA,
`promoter-study-status`, and next-scope preflight report the product contracts
as current in this checkout.

Open: generated sidecars still need explicit USR sync/publish handling for other
checkouts or cluster roots. `view_semantics.parquet` addenda remain a separate
open slice.

Deliverables:

- Add a USR check that validates a dataset's sequence-view sidecar against
  records and expected view counts.
- Add study-specific manifests for expected counts:
  - `usr_prom_eth_cip_anchor`: one `construct_insert` forward view per row.
  - `construct_prom_eth_cip_context`: one forward and one reverse-complement
    context per merged anchor row.
  - `construct_prom_eth_cip_reference_core60`: all rows `analysis_window`, length
    60, `analysis_only=true`.

Tests:

- Native exact-60 rows remain `construct_insert` or `selected_region`.
- Core60 source rows are the only `analysis_window` products.
- Context view counts fail if one orientation is missing.

#### Slice 4: Harden Construct contracts

Status: done for the core Construct code path and focused tests.

Open: generated dataset regeneration/publish and full-repo validation remain
separate artifact and integration tasks.

Deliverables:

- Keep `normalize_anchor` constrained to `analysis_window` for now.
- Keep `construct_insert` ownership in USR materialization.
- Add tests that fail when -35/-10 selectors are missing or ambiguous.
- Add tests that all core60 outputs retain required sigma-site features.
- Add Biopython-backed or independently verified tests for full reverse-complement
  context equality and emitted-orientation anchor bounds.

Tests:

- `reverse_complement(forward_context) == reverse_context`.
- `reverse_anchor_start_0 = L - forward_anchor_end_0`.
- `reverse_anchor_end_0 = L - forward_anchor_start_0`.
- `anchor_mean` bounds are valid in both orientations.

#### Slice 5: Add Infer completion planner

Status: done for sequence-view bundles, first-class missing product reporting,
explicit completion thresholds, persisted feature-vector sidecars, and
digest-matching legacy overlays.

Open: sequence-auditing drifted legacy overlays remains open.

Deliverables:

- Add a dry-run planner that materializes sequence-view selectors and classifies
  required features as reusable, stale, missing, or blocked by missing product.
- Make the planner validate pooling bounds without loading GPU models.
- Emit machine-readable JSON for Ops and batch runbooks.
- Add optional thresholds so batch runbooks can fail on missing sequence
  products or stale vectors without failing on feature vectors they are meant to
  generate.

Tests:

- Old row-based overlay with matching identity is `reusable`.
- Missing reverse-complement context feature is `missing`, not `stale`.
- Missing sequence view products are reported through `missing_products` and
  `missing_product_selectors` before model execution.
- A zero-row required selector still returns a machine-readable plan instead of
  terminating the planner before Ops can aggregate the result.
- Digest mismatch is reported as stale/unclassified unless a future
  sequence/model/pooling audit can prove equivalence.
- Persisted feature-vector sidecars are counted as reusable without loading the
  model.
- Thresholded CLI validation fails when `missing_products` or `stale_vectors`
  exceed the configured submit gate.

#### Slice 6: Add Infer feature alias migration

Status: implemented as a conservative bridge; active-study anchor and forward
context backfill is written locally. Remaining work is true new inference plus
generated-artifact sync/publish.

Runtime feature execution writes aliases and feature vectors for
new sequence-view runs. The standalone migration command now audits and, when
explicitly run with `--write`, copies verified legacy row-overlay vectors into
the sequence-view `feature_aliases.parquet` and `feature_vectors.parquet`
sidecars.

Command:

```bash
uv run infer migrate legacy-overlay-aliases \
  --config <sequence-view-config.yaml> \
  --job <sequence-view-job-id> \
  --legacy-job-id <old-row-overlay-job-id> \
  --format json
```

Use `--max-views <n>` for smoke tests and omit it for full metadata-only
planning. Dry-run mode intentionally skips large embedding payload columns and
reports those rows in `payload_unverified_vectors`; add `--verify-payloads` only
when the expensive payload scan is required before write. Use `--write` only
after the reusable, missing, unclassified, and orientation-blocked counts match
expectations.

After alias/vector sidecars are protected, retire duplicated legacy embedding
payload columns with:

```bash
uv run infer migrate retire-legacy-payloads \
  --config <sequence-view-config.yaml> \
  --job <sequence-view-job-id> \
  --legacy-job-id <old-row-overlay-job-id>
```

The retirement command is dry-run by default. It scans legacy metadata and
canonical feature-vector keys without reading embedding payload columns. With
`--write`, it refuses to mutate if any legacy-present vector lacks canonical
sidecar protection or if the legacy identity is unclassified. It rewrites
Parquet parts by streaming retained columns, so normal cleanup does not
materialize multi-GB embedding tables in memory.

Approved stale lanes that are not protected by a modern semantic sidecar use a
different command. This is intentionally explicit: it is for payload families
that have been declared out of scope or non-actionable, not for reusable legacy
vectors.

```bash
uv run infer migrate prune-stale-overlay-columns \
  --usr-root src/dnadesign/usr/datasets \
  --dataset <dataset-id> \
  --column-prefix infer__evo2_20b__ \
  --reason "collapsed 20B row-overlay lane retired before sequence-view completion"
```

`prune-stale-overlay-columns` is dry-run by default, scans only Parquet schemas
and column-chunk metadata, and rewrites retained columns in small batches when
`--write` is supplied. It refuses to remove the overlay join column `id` and
logs an `infer_stale_overlay_column_prune` event after a successful write.

Deliverables:

- Done: add alias backfill from verified old row overlays into new feature alias
  maps.
- Done: backfill `157164` anchor `construct_insert` aliases/vectors from the
  compatible legacy `anchor_only` row-overlay lane.
- Done: backfill `157164` forward `realized_context` anchor-mean
  aliases/vectors from the compatible legacy forward-context row-overlay lane.
- Done: write `forward_pass_key` and `feature_vector_key` where existing vectors
  can be proven equivalent.
- Done: preserve `metadata__feature_request_digest` as resume metadata; do not
  reuse it as the feature-vector identity.
- Done: make full-study dry-runs metadata-only by default so planning does not
  load large embedding payload columns.
- Done: make completion/status count persisted feature-vector sidecars through
  key-only parquet inventory so normal checks do not load multi-GB embedding
  payload columns.
- Done: retire duplicated legacy row-overlay embedding payload columns for
  `anchor_construct_insert_seq_mean_7b` and `context_forward_anchor_mean_7b`
  after verifying `157164` protected vectors in each canonical sidecar store.
  The cleanup rewrote `57` anchor overlay parts and `1012` context overlay parts,
  reclaiming about `2.1 GB` total while preserving modern
  `_derived/infer/feature_vectors.parquet` payloads.
- Done: remove collapsed/debug-required `infer__evo2_20b__*` row-overlay payload
  and metadata columns from the active study handoffs after explicit approval.
  The cleanup rewrote one mixed part and deleted 20B-only parts in each dataset:
  `usr_prom_eth_cip_anchor` reclaimed about `2.07 GB`; `construct_prom_eth_cip_context`
  reclaimed about `4.08 GB`. Post-cleanup dry-runs report zero remaining
  `infer__evo2_20b__*` columns in both active handoffs.
- Done: add an offline USR event-log gardening lifecycle. `uv run usr maintenance
  event-log-garden <dataset>` dry-runs by default, archives the full log on
  write, retains a bounded live tail, appends an `event_log_garden` audit event,
  and requires `--acknowledge-notify-cursor-reset` before rewriting `.events.log`.
  Active handoff logs were inspected in dry-run mode only; they were not
  gardened because live cursor/sync coordination must be explicit.
- Open: run Infer for the `115` missing anchor rows, `115` missing forward
  context rows, all `157279` reverse-complement context rows, and the `144`
  reference-view feature vectors if the reference branch is in scope.
- Open: publish or sync the resulting `_derived/infer/*` sidecars to canonical
  shared/cluster roots.
- Guardrail: `--write` reads full vector payloads by design and should be
  treated as a deliberate generated-data batch, not a casual interactive status
  or preflight command.

Tests:

- Existing runtime tests cover exact 60 bp `seq_mean` and `core60_mean` views
  aliasing to one vector when the
  pooling span is identical.
- Existing runtime tests cover same sequence with different pooling span
  producing a different feature vector key.
- Existing runtime tests cover forward and reverse-complement emitted
  orientations remaining different forward pass keys.
- New migration tests cover verified forward legacy-overlay reuse and refusal to
  reuse forward overlays for reverse-complement views.
- New retirement tests cover refusing cleanup before canonical vectors exist and
  pruning only protected duplicate payload columns while preserving legacy
  metadata columns.

#### Slice 7: Upgrade Study/Ops status and preflight

Status: implemented for product and feature status aggregation. The active
study product sidecar migration is complete locally; remaining hardening is
generated-data sync/publish discipline and richer legacy Infer identity proof.

Done: study prose is updated, Ops/preflight has first-class USR sequence-view QA
through `sequence_view_contract`, and Ops/preflight consumes Infer
feature-completion planner output through `infer_sequence_view_completion`.
`promoter-study-status` now exposes summarized product completion,
feature-completion, and generated-artifact freshness fields without duplicating
the full preflight payload.
Sequence-view Infer runbook plans now render the completion planner before
`infer run --dry-run` with product/stale thresholds, so runbook execution cannot
submit a lane whose required sequence products are absent.

Deliverables:

- Add a sequence-view contract check to promoter preflight. Implemented.
- Update promoter-study status to report coverage by product kind, orientation,
  pooling, and model/layer. Implemented as summary fields; legacy
  sequence/model/pooling proof remains an alias-migration slice.
- Add explicit "product completion" versus "feature completion" sections.
  Implemented in preflight and status summary fields.
- Keep the reference-view branch non-blocking unless the study record marks it
  required. Implemented for reference sequence-view checks by using
  `required: false`.
- Gate sequence-view batch runbooks on missing sequence products and stale
  vectors while allowing missing feature vectors as the batch workload.
  Implemented in runbook planning.

Tests:

- Status remains `ok` when planned reference Infer features are missing but marked
  non-blocking.
- Status fails or warns when required reverse-complement context products are
  missing.
- Status distinguishes reusable old vectors from missing vectors.

#### Slice 8: Generalize without churn

Status: partially implemented.

Done: product kinds are hard-cut to the generic vocabulary in code, configs,
tests, docs, and local active-study sequence-view sidecars.

Open: promoter-specific class names and some context-kind values remain as
compatibility surfaces for existing configs.

Deliverables:

- Introduce generic docs language for sequence products, selected regions,
  analysis windows, and realized contexts. Implemented for product-kind terms.
- Keep promoter-specific values as compatibility terms. Superseded for
  product-kind values; generated active-study sidecars have been migrated
  locally, but sync/publish handling remains required for other roots.
- Do not rename `PromoterFeatureBundleConfig` until a generic
  `SequenceFeatureBundleConfig` adapter exists and legacy configs are covered by
  tests.

Tests:

- Existing promoter configs keep validating.
- New sequence-view configs can be expressed without adding another promoter-only
  product kind.

### Completion Matrix for the Active Study

Current desired products:

| Dataset | Required product surface | Current expectation |
| --- | --- | --- |
| `usr_promoter_references` | source-backed reference inserts | native lengths retained |
| `construct_prom_eth_cip_reference_core60` | `analysis_window` | 48 rows, length 60, analysis-only |
| `usr_prom_eth_cip_anchor` | merged `construct_insert` anchor handoff | 157,279 rows/views |
| `construct_prom_eth_cip_context` | paired forward and reverse-complement 1 kb contexts | 314,558 rows/views |
| `construct_prom_eth_cip_reference_contexts` | paired forward and reverse-complement contexts for core60 references | 96 rows/views |

Current feature completion interpretation:

- Existing DenseGen anchor and covered forward context features are legacy-schema
  features. After the local migration bridge writes aliases/vector sidecars, the
  compatible subset is now sequence-view reusable rather than merely presumed
  reusable.
- Do not reset them globally.
- Classify them with `infer validate sequence-view-completion` before
  recomputation.
- Product-contract health must be `ok` before interpreting feature-completion
  counts. If sequence-view sidecars fail to load or carry stale product-kind
  names, the feature planner has not reached a meaningful vector-classification
  stage.
- In the current local checkout, product selectors resolve and
  `missing_products=0`. The main 7B planner reports `314328` reusable vectors
  from local alias/vector sidecars and `157509` missing vectors.
- The missing main vectors are targeted work: `115` anchor rows, `115` forward
  context rows, and all `157279` reverse-complement context rows.
- The reference 7B planner reports `144` missing vectors for the reference
  branch (`48` analysis-window/core60 rows plus `96` paired reference
  contexts).
- Reverse-complement context features are a new feature surface. Existing forward
  context overlays are not coverage for reverse-complement contexts, even when
  they share parent anchors.
- Exact 60 bp native or designed rows may be feature-equivalent to some
  `analysis_window` requests, but that equivalence must be represented through
  Infer feature aliases rather than through duplicate USR product rows.

Current implemented checks:

| Check | Command or surface |
| --- | --- |
| USR sequence-view unit and QA behavior | `uv run pytest -q src/dnadesign/usr/tests/datasets/views/test_sequence_views.py` |
| USR promoter-study view-semantics materializer | `uv run pytest -q src/dnadesign/usr/tests/scripts/test_materialize_promoter_study_view_semantics.py` |
| Infer sequence-view completion planner | `uv run pytest -q src/dnadesign/infer/tests/runtime/test_sequence_view_completion_planner.py` |
| Infer CLI planner surface | `uv run pytest -q src/dnadesign/infer/tests/cli/test_validate_command.py` |
| Existing feature-bundle sequence-view behavior | `uv run pytest -q src/dnadesign/infer/tests/runtime/test_feature_bundle_execution.py` |
| Ops/preflight sequence-view contract check | `uv run pytest -q src/dnadesign/ops/tests/test_progress_cli.py::test_promoter_study_preflight_reports_sequence_view_contract_health` |
| Ops/status sequence-view and Infer completion summaries | `uv run pytest -q src/dnadesign/ops/tests/test_progress_cli.py::test_promoter_study_status_reports_sequence_view_and_infer_completion_summary` |
| Sequence-view runbook product/stale submit gate | `uv run pytest -q src/dnadesign/ops/tests/test_runbook_orchestrator.py::test_sequence_view_infer_runbook_preflight_gates_missing_products_not_missing_vectors` |
| Study/Ops record-plane smoke check | `uv run ops progress show usr.data-plane.promoter-study-status --json` |

### Cutover Acceptance

The sequence-view ontology cutover is only complete when product semantics,
feature completion, generated artifacts, and operator runbooks agree. Current
state:

- [x] Product contracts are green in `promoter-study-status` and preflight for
  the active main/reference sequence-view datasets.
- [x] Sequence-view completion reports `missing_products=0` for active main and
  reference 7B configs.
- [x] Anchor and forward legacy-compatible vectors have been sidecar-backfilled
  from verified row-overlay lanes.
- [x] Validation/status use key-only vector inventory and do not load embedding
  payload columns.
- [x] Sequence-view batch runbook plans gate missing sequence products and stale
  vectors before submit, without blocking on missing feature vectors.
- [ ] Reverse-complement, remaining `115 + 115` main rows, and `144` reference
  vectors are generated or explicitly deferred.
- [ ] Notify sequence-view batch profiles and runbook plans are validated with
  real file-backed secret/TLS settings in the target execution environment.
- [ ] Generated `_views/*` and `_derived/infer/*` sidecars are synced or
  published to shared/cluster roots.
- [ ] Full repo validation passes for the integrated branch.

### Failure Contracts

Fail before writing or model execution when:

- A required sequence-view selector resolves zero rows.
- A required product kind or orientation count is lower than expected.
- `anchor_mean` lacks explicit bounds.
- Pooling bounds are outside emitted sequence length.
- `core60_mean` receives a non-60 bp sequence.
- Reverse-complement contexts lack a forward parent or fail sequence equality.
- A sidecar view references a missing sequence id.
- A semantic addendum references a missing `view_id`.
- A semantic addendum references the right `view_id` but the wrong
  `sequence_id`.
- A stale row-based overlay cannot be proven equivalent to the requested
  sequence-view feature identity.
- A planner selector resolves zero rows for a required sequence-view input.
- An Ops/status surface reports feature coverage only by dataset row count after
  reverse-complement contexts are in scope.

Allowed degraded states:

- Planned reference-view Infer features may remain missing without blocking the
  main study if the study record marks the branch non-blocking.
- Old row-based Infer overlays may remain present while alias migration is
  incomplete, but status must label them as legacy-schema reusable or stale.
- `view_semantics.parquet` may be absent from generated datasets while the first
  product-kind and pooling contracts are enforced by `_views/sequence_views.parquet`;
  machine selection by `view_collections` must remain disabled until the addendum
  exists and validates.

Disallowed degraded states:

- Treating forward contexts as reverse-complement contexts.
- Treating source/native exact-60 rows as `analysis_window`.
- Treating `anchor_mean` as sequence truncation.
- Recomputing large Infer batches only because metadata fields changed, without a
  reuse/staleness audit.
- Selecting study cohorts from human aliases when a typed `view_collections` or
  manifest is required.

### Verification Commands

Focused checks after documentation-only changes:

```bash
uv run python -m dnadesign.devtools.docs_checks --repo-root .
```

Focused checks after USR sequence-view changes:

```bash
uv run pytest -q src/dnadesign/usr/tests/datasets/views
uv run pytest -q src/dnadesign/usr/tests/scripts/test_materialize_promoter_anchor_sequence_views.py
uv run pytest -q src/dnadesign/usr/tests/test_module_layout.py
```

Focused checks after USR sequence-view semantics/QA changes:

```bash
uv run pytest -q src/dnadesign/usr/tests/datasets/views/test_sequence_views.py
uv run pytest -q src/dnadesign/usr/tests/test_module_layout.py
```

Focused checks after Construct changes:

```bash
uv run pytest -q src/dnadesign/construct/tests/runtime/test_run_construct.py
uv run pytest -q src/dnadesign/construct/tests/test_study_workspace_contract.py
```

Focused checks after Infer changes:

```bash
uv run pytest -q src/dnadesign/infer/tests/runtime/test_feature_bundle_execution.py
uv run pytest -q src/dnadesign/infer/tests/runtime/test_sequence_view_completion_planner.py
uv run pytest -q src/dnadesign/infer/tests/cli/test_validate_command.py
```

Focused checks after Study/Ops changes:

```bash
uv run ops progress show usr.data-plane.promoter-study-status --json
uv run pytest -q src/dnadesign/studies/tests src/dnadesign/ops/tests
```

Repo-level checks before merging:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run python -m dnadesign.devtools.docs_checks --repo-root .
```

### Acceptance Checklist

- [x] Product identity, pooling, provenance, selection basis, and study collection
  membership are documented as separate axes.
- [x] Native exact-60 rows are not duplicated or relabeled as `analysis_window`.
- [x] Generated study datasets include `view_semantics.parquet` where
  machine-readable `source_family`, `selection_basis`, or `view_collections`
  selectors are required.
- [x] Sequence-view QA checks validate product-kind, orientation, pooling, exact
  product length, and emitted bounds before handoff.
- [x] Reference core60 derivation is fail-fast around required sigma-site
  annotations.
- [x] Reverse-complement context generation is tested by sequence equality and
  emitted-orientation anchor bounds.
- [x] Infer dry-run can classify reusable, stale, and missing feature states
  without loading models.
- [x] Infer dry-run emits missing sequence products as a first-class planner
  state instead of only failing zero-row selectors.
- [x] Infer planner output is consumed by Ops/preflight before any large Evo2
  backfill is submitted.
- [x] Study/Ops status reports coverage by product kind, orientation, pooling, and
  model/layer.
- [x] Existing row-based Infer outputs can be classified before any large rerun
  when digest evidence is present.
- [x] Active-study anchor and forward-context legacy-compatible vectors are
  backfilled into sequence-view feature alias/vector sidecars.
- [x] Completion/status checks use key-only feature-vector sidecar inventory and
  avoid loading embedding payload columns during normal planning.
- [x] Sequence-view runbook plans fail before submit when sequence products are
  missing or vectors are stale, while allowing missing vectors as intended
  batch work.
- [ ] Drifted legacy row overlays can be sequence/model/pooling-audited and
  alias-migrated without recomputation.
- [x] Generated USR data sync/publish posture is explicit.
- [ ] Reverse-complement context vectors, the remaining `115` anchor vectors,
  the remaining `115` forward-context vectors, and `144` reference-branch
  vectors are generated or explicitly deferred in the study record.
- [ ] Generated `_views/*` and `_derived/infer/*` sidecars are synced or
  published to canonical shared/cluster roots.
- [ ] Notify sequence-view batch runbooks have real file-backed secret/TLS
  configuration and pass next-run preflight in the target environment.
- [ ] Full repo checks pass for the integrated branch.

### Skeptic Checks

Objection: "Why not make every 60 bp row `analysis_window`?"

Answer: because `analysis_window` encodes derivation and analysis-only semantics.
DenseGen and SFXI exact-60 rows are designed/native insert products. Feature
equivalence belongs in Infer aliases, not in USR product identity.

Objection: "Why not let Infer generate missing reverse complements?"

Answer: reverse-complement contexts are materialized sequence products with
Construct-owned bounds and provenance. Infer consumes emitted products and
features; it must not mutate sequence-product state.

Objection: "Why add another sidecar instead of adding more product kinds?"

Answer: product-kind growth couples study membership, source provenance, and
selection logic into one enum. A companion view-semantics sidecar lets the study
ask richer questions without changing stable view identity or forcing broad
schema migration.

Objection: "Are legacy Infer features invalid?"

Answer: not by default. They are legacy-schema outputs. They can be reused if
model, sequence, pooling span, bounds, and orientation match the new requested
view identity. The planner must prove that before aliasing or rerunning.

Objection: "Why not let `view_collections` change the `view_id`?"

Answer: study collection membership is mutable operational metadata. If it were
part of `view_id`, the same sequence product would acquire new identities every
time a study cohort changes. That would break feature aliasing and force
unnecessary recomputation.

Objection: "Why classify drifted legacy overlays as stale or unclassified instead
of reusable?"

Answer: digest drift is not proof of invalid science, but absent a
sequence/model/pooling audit it is also not proof of compatibility. The planner
therefore uses a conservative state until an alias-migration command can prove
equivalence explicitly.

### Next Increment

The generated sequence-view product sidecar migration is complete locally. The
former blocking artifact slice mapped stale names to the generic vocabulary,
rebuilt each `SequenceViewRecord`, and recomputed `view_id` values instead of
raw-editing Parquet strings. Current local product-contract acceptance is:

- `usr_prom_eth_cip_anchor`: `157279` `construct_insert` views.
- `construct_prom_eth_cip_context`: `314558` `realized_context` views, split
  `157279` forward and `157279` reverse-complement.
- `construct_prom_eth_cip_reference_core60`: `48` `analysis_window` views,
  length 60, `core60_mean`.
- `construct_prom_eth_cip_reference_contexts`: `96` `realized_context` views,
  split `48` forward and `48` reverse-complement.
- `usr_promoter_references`: `48` `selected_region` source-backed reference
  views.
- `uv run ops progress show usr.data-plane.promoter-study-status --json`
  reports `state=ok`, `sequence-view product contracts 4/4 ok`, and
  `generated_artifact_freshness.state=ok`.

Remaining generated-data work:

- Publish or sync the migrated generated sidecars, view-semantics addenda, and
  feature alias/vector sidecars to any canonical shared or cluster roots. The
  local Parquet files are still generated artifacts outside normal
  tracked-source review.

The highest-leverage generated-data slice has moved from legacy migration to
true feature completion. Local alias/vector backfill is complete for the
legacy-compatible anchor and forward-context surfaces:

- `usr_prom_eth_cip_anchor`: `157164` reusable aliases/vectors written,
  `115` vectors still missing.
- `construct_prom_eth_cip_context` forward anchor-mean: `157164` reusable
  aliases/vectors written, `115` vectors still missing.
- `construct_prom_eth_cip_context` reverse-complement anchor-mean: `0`
  reusable legacy vectors, `157279` vectors still missing.
- Reference sequence-view branch: `144` vectors still missing.

The storage cleanup portion of the legacy migration is also complete locally:

- `usr_prom_eth_cip_anchor`: duplicate legacy `anchor_only_7b_features`
  intermediate-embedding payload column retired from `57` row-overlay parts;
  post-cleanup dry-run reports `parts_with_payload=0`.
- `construct_prom_eth_cip_context`: duplicate legacy
  `template_1kb_7b_features` anchor-mean intermediate-embedding payload column
  retired from `1012` row-overlay parts; post-cleanup dry-run reports
  `parts_with_payload=0`.
- Canonical modern feature-vector sidecars remain present for both protected
  surfaces, each with `157164` reusable vectors.
- Stale `infer__evo2_20b__*` row-overlay columns have also been removed from the
  active anchor and context handoff datasets. This is a deletion of collapsed,
  debug-required legacy payloads, not a migration of reusable vectors.

Exit criteria:

- Done: a dry-run mode reports reusable, missing, unclassified, and
  orientation-blocked legacy rows
  without writing sidecars.
- Done: a write mode creates feature aliases only for rows whose exact row id,
  model/layer, pooling operation, pooling bounds, and emitted orientation match
  the requested sequence-view feature identity.
- Done: completion/status inventory reads only feature keys from large vector
  sidecars.
- Done: a guarded payload-retirement mode removes duplicate legacy embedding
  payload columns only after canonical sidecar protection is proven.
- Drifted or absent legacy digests are never marked reusable without this exact
  identity audit.
- Open: reverse-complement, remaining anchor/forward, and reference feature
  execution should be run as explicit Infer batch operations, not as accidental
  status/preflight side effects.

The third highest-leverage slice, cheap status aggregation of the
feature-completion planner output, is implemented. `usr.data-plane.promoter-study-status
--json` now exposes summarized product completion, feature completion, and
generated-artifact freshness fields. The text status page links the detailed
preflight route for next-run blockers, and planned reference features remain
non-blocking unless the study record marks them required.
