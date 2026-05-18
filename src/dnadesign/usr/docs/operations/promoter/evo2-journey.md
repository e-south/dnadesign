## Promoter Study Evo2 Workflow Journey

**Type:** route
**Plane:** data-plane
**Owner-boundary:** usr
**Entry artifact:** promoter anchors or wildtype/manual promoter records that still need a concrete cross-tool route
**Exit artifact:** the next concrete runbook for source assembly, contextualization, Evo2 feature extraction, notification, or downstream analysis

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-20

Use this page when you need the full promoter-study Evo2 path in one place before choosing a concrete runbook.
If you are checking the live study record, start with [Stress ethanol/cipro status contract](../../../../../../docs/studies/stress_ethanol_cipro_growth/contracts/status.md).
If you need blockers or next-run readiness, switch to `uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --json`.

If you are entering from Ops, use `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json` for the current snapshot record and `uv run ops catalog show studies.stress-ethanol-cipro-growth.status --json` to inspect its related LatentDNA route entry.
The study record is the right place for current phase, dataset posture, and downstream handoff state. This route page is for movement between surfaces.

### Workflow summary

1. Start with promoter anchors.
   DenseGen-generated anchors live in [DenseGen documentation](../../../../densegen/docs/README.md).
   Manual or wildtype promoter records that still need a USR dataset boundary live in [USR CLI quickstart](../../getting-started/cli-quickstart.md).
2. Merge upstream sources when one shared dataset is required.
   Use [Multi-source shared dataset assembly](../assembly/multi-source-shared-dataset.md) when DenseGen outputs, manual promoters, or wildtype controls still live in separate USR datasets.
3. Keep anchor-only and template-backed contexts explicit.
   Anchor-only routes can stay in the merged USR dataset.
   Template-backed routes should go through [Construct -> USR -> Infer shared dataset runbook](../assembly/construct-infer-shared-dataset-runbook.md) plus the Construct-owned [template/context contract](../../../../construct/docs/reference/template-contexts.md).
4. Run infer-owned Evo2 feature extraction.
   Use [Evo2 sequence-feature runbook](../../../../infer/docs/operations/evo2-sequence-features.md) for validate, dry-run, run, and study-bound anchor/full-context feature extraction.
   Use [Evo2 provider reference](../../../../infer/docs/reference/evo2-provider.md) for `evo2_7b`, `evo2_20b`, the model-aware intermediate default, pooling, and stored output names.
5. Validate the optional Notify side branch only when watcher behavior matters.
   Use [Notify operations route](../../../../../../docs/notify/README.md) or the operator manual [Notify: consuming Universal Sequence Record events](../../../../../../docs/notify/usr-events.md).
6. Hand off into the next study-owned downstream branch.
   Use [Stress ethanol/cipro status contract](../../../../../../docs/studies/stress_ethanol_cipro_growth/contracts/status.md) plus the checked-in study `routes/README.md` when you need the live study-specific DenseGen, Construct, Infer, LatentDNA, Cluster, or OPAL handoff rather than the generic route.
   Use [Stress ethanol/cipro representation comparison](../../../../latentdna/docs/workflows/stress-ethanol-cipro-representation-comparison.md) when the stress study's anchor and construct-context datasets already carry the vector columns you want to compare in latent space.
   Continue to [cluster exploratory clustering workflow](../../../../cluster/docs/workflows/exploratory-clustering.md) when you want exploratory grouping next.
   Continue to [USR dataset with infer-derived X -> OPAL active learning](../../../../opal/docs/workflows/usr-infer-x-active-learning.md) only after a separate study-owned downstream decision names a concrete feature bundle worth carrying into OPAL.

### Boundary reminders

- USR owns the durable dataset identity and cross-tool handoff.
- Construct owns anchor/template geometry and `construct__*` coordinate metadata.
- Infer owns Evo2 lane choice, feature-bundle pooling, caching, and `infer__*` write-back.
- Notify consumes `.events.log`; it does not own feature assembly.
- OPAL starts only after a separate study-owned downstream decision names one explicit feature bundle worth carrying into active learning.

### Choose the next deep procedure

- If you need one maintained study snapshot for current-status checks, use [Stress ethanol/cipro status contract](../../../../../../docs/studies/stress_ethanol_cipro_growth/contracts/status.md).
- If you need the study-bound LatentDNA comparison workspace and workflow next, use [Stress ethanol/cipro representation comparison](../../../../latentdna/docs/workflows/stress-ethanol-cipro-representation-comparison.md).
- If you need template-backed contexts such as `template_1kb` before feature extraction, use [Construct -> USR -> Infer shared dataset runbook](../assembly/construct-infer-shared-dataset-runbook.md).
- If DenseGen, manual, and wildtype sources still need explicit merge/carry setup, use [Multi-source shared dataset assembly](../assembly/multi-source-shared-dataset.md).
- If the data-plane handoff is already clear and you only need the infer-owned Evo2 contract, use [Evo2 sequence-feature runbook](../../../../infer/docs/operations/evo2-sequence-features.md).

### Related docs

- Docs index: [../../../../../../docs/README.md](../../../../../../docs/README.md)
- USR docs index: [../README.md](../README.md)
- USR operations index: [README.md](../README.md)
- DenseGen docs: [../../../../densegen/docs/README.md](../../../../densegen/docs/README.md)
- Construct docs: [../../../../construct/docs/README.md](../../../../construct/docs/README.md)
- Infer docs: [../../../../infer/docs/README.md](../../../../infer/docs/README.md)
- LatentDNA docs: [../../../../latentdna/docs/README.md](../../../../latentdna/docs/README.md)
- Notify operations route: [../../../../../../docs/notify/README.md](../../../../../../docs/notify/README.md)
