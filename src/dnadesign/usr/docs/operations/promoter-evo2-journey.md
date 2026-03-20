## Promoter Evo2 Workflow Journey

**Type:** route
**Plane:** data-plane
**Owner-boundary:** usr
**Entry artifact:** promoter anchors or wildtype/manual promoter records that still need a concrete cross-tool route
**Exit artifact:** chosen owner-local runbook for source assembly, optional construct contextualization, Evo2 feature extraction, notification, or downstream analysis

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-20

Use this page when you need the whole promoter/Evo2 route in one place before choosing a concrete runbook. This page is a route map only. The linked owner-local runbooks remain authoritative for commands, schemas, and failure handling.

### If you are entering from Ops

- `uv run ops catalog show usr.data-plane.promoter-feature-matrix`: inspect the authoritative cross-tool feature-matrix route plus related tool docs.
- `uv run ops progress explain usr.data-plane.promoter-feature-matrix`: see the required read-only status inputs before you touch artifacts.
- `uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix`: emit a starting manifest across the related route set when you want a read-only campaign view.

### Route at a glance

1. Start with promoter anchors.
   DenseGen-generated anchors: [DenseGen documentation](../../../densegen/docs/README.md)
   Manual or wildtype promoter records that still need a USR dataset boundary: [USR CLI quickstart](../getting-started/cli-quickstart.md)
2. Merge upstream sources when one shared dataset is required.
   Use [Multi-source source-of-truth assembly](multi-source-source-of-truth-assembly.md) when DenseGen outputs, manual promoters, or wildtype controls still live in separate USR datasets.
3. Keep anchor-only and template-backed contexts explicit.
   Anchor-only routes can stay in the merged USR dataset.
   Template-backed routes should go through [Construct -> USR -> Infer source-of-truth runbook](construct-infer-source-of-truth-runbook.md) plus the Construct-owned [template/context contract](../../../construct/docs/reference/template-contexts.md).
4. Run infer-owned Evo2 feature extraction.
   Use [Evo2 promoter feature runbook](../../../infer/docs/operations/evo2-promoter-features.md) for validate, dry-run, run, and OPAL-export flow.
   Use [Evo2 provider reference](../../../infer/docs/reference/evo2-provider.md) for `evo2_7b`, `evo2_20b`, block-26 default, pooling, and stored output names.
5. Validate the optional Notify side branch only when watcher behavior matters.
   Use [Notify operations route](../../../../../docs/notify/README.md) or the operator manual [Notify: consuming Universal Sequence Record events](../../../../../docs/notify/usr-events.md).
6. Hand off into the authoritative cross-tool feature matrix route, then branch downstream.
   Use [Promoter characterization feature matrix](promoter-characterization-feature-matrix.md) once the study should become one infer-annotated USR dataset.
   After that handoff, choose either [cluster exploratory clustering workflow](../../../cluster/docs/workflows/exploratory-clustering.md) or [USR dataset with infer-derived X -> OPAL active learning](../../../opal/docs/workflows/usr-infer-x-active-learning.md).

### Boundary reminders

- USR owns the durable dataset identity and cross-tool handoff.
- Construct owns anchor/template geometry and `construct__*` coordinate metadata.
- Infer owns Evo2 lane choice, feature-bundle pooling, caching, and `infer__*` write-back.
- Notify consumes `.events.log`; it does not own feature assembly.
- OPAL starts only after one explicit infer-derived `X` column or exported matrix already exists.

### Choose the next deep procedure

- If you already have one merged anchor-only dataset and need infer-derived columns next, use [Promoter characterization feature matrix](promoter-characterization-feature-matrix.md).
- If you need template-backed contexts such as `template_1kb` before feature extraction, use [Construct -> USR -> Infer source-of-truth runbook](construct-infer-source-of-truth-runbook.md).
- If DenseGen, manual, and wildtype sources still need explicit merge/carry setup, use [Multi-source source-of-truth assembly](multi-source-source-of-truth-assembly.md).
- If the data-plane handoff is already clear and you only need the infer-owned Evo2 contract, use [Evo2 promoter feature runbook](../../../infer/docs/operations/evo2-promoter-features.md).

### Related docs

- Root docs router: [../../../../../docs/README.md](../../../../../docs/README.md)
- USR docs index: [../README.md](../README.md)
- USR operations index: [README.md](README.md)
- DenseGen docs: [../../../densegen/docs/README.md](../../../densegen/docs/README.md)
- Construct docs: [../../../construct/docs/README.md](../../../construct/docs/README.md)
- Infer docs: [../../../infer/docs/README.md](../../../infer/docs/README.md)
- Notify operations route: [../../../../../docs/notify/README.md](../../../../../docs/notify/README.md)
