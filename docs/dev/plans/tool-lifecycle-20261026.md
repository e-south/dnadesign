## Tool lifecycle and independent installation

**Owner:** dnadesign-maintainers
**Status:** initial migration complete; DenseGen consumer handoffs remain
**Last verified:** 2026-10-26

Keep supported tools independently usable, move new library construction to
[DenseArrays](https://github.com/e-south/dense-arrays), and preserve the exact
versions needed for historical calculations. Retirement removes a tool from the
current distribution only after its consumers and retrieval path are accounted for.

### Tool disposition

The inventory covers tracked Python imports, configuration references and the
public companion's declared software roles. A missing import is not proof of
no use: command-line calls, notebook-generated code and external users also
matter. Library dependencies below name the main runtime families, not a
second dependency specification; `pyproject.toml` owns install requirements.

| Tool | CLI | Runtime and tool dependencies | Consumers and retained role | Disposition |
| --- | --- | --- | --- | --- |
| OPAL | `opal` | Numerical/table libraries, scikit-learn; optional USR and OPS integrations | Experimental and construction-label campaigns, public scoring API, companion replay | Active; isolate its installation and local-table path |
| USR | `usr` | Arrow/tables; artifacts, OPS, selected Cruncher integration | Construct, Infer, LatentDNA, Cluster and explicit dataset workflows | Active; optional for OPAL |
| Construct | `construct` | Sequence/table libraries; contracts, Folding, USR | Reporter-context reconstruction and sequence assembly clients | Active |
| Infer | `infer` | Table/runtime libraries; USR; explicit GPU model extra | Sequence models and feature consumers, including Permuter | Active; keep model installations explicit |
| LatentDNA | `latentdna` | Numerical/geometry/plotting libraries; USR | Companion feature transformations and reference comparisons | Active |
| DenseGen | `dense` | DenseArrays and solver, tables/plots; Cruncher, USR, BaseRender | Existing construction clients and historically pinned retained-part sampler | Being replaced by DenseArrays; remove only after consumer migration |
| Cruncher | `cruncher` | Motif/sequence/plotting libraries; artifacts, contracts, BaseRender | Motif reconstruction, DenseGen sampling and USR integrations | Active; retire only superseded individual capabilities |
| BaseRender | `baserender` | Rendering libraries; artifacts and contracts | Cruncher, DenseGen and external sequence-figure clients | Active neutral renderer |
| Folding | `folding` | Folding/structure libraries; artifacts and contracts | Construct and external structural assessment | Active |
| Junction | `junction` | Sequence/structure libraries; artifacts and contracts | External oligo planning clients | Active |
| Aligner | module API | Sequence-alignment libraries | Cluster, library-analysis tools and external alignment clients | Active shared capability; not an archive candidate |
| Cluster | `cluster` | Numerical/clustering/plotting libraries; Aligner, USR, OPS | Supported exploratory clustering workflow | Retain pending direct-user inventory |
| Permuter | `permuter` | Sequence/tables; Infer | External sequence-perturbation clients | Active |
| Thread | module API | Structure/model clients and visualization | External structure-prediction workflows | Active |
| Notify | `notify` | Transport/CLI libraries; producer event integrations | OPS and producer notification workflows | Active; detach retired producer integrations |
| OPS | `ops` | CLI/config libraries; explicit tool providers | Workflow/status callers, including USR and OPAL integrations | Active; providers follow tool lifecycle |
| LibShuffle | module API | Numerical/clustering libraries; Aligner, Billboard, NMF | No external runtime consumer found in the inspected checkouts | Archived and removed; retrieval is recorded |
| Billboard | module API | Sequence/table/plotting libraries; Aligner | LibShuffle | Archived and removed with LibShuffle |
| NMF | module API | Factorization/table/plotting libraries | LibShuffle | Archived and removed with LibShuffle |
| TFKDAnalysis | module API | Scientific tables/plots; stale utility import | No external runtime consumer found in the inspected checkouts | Archived and removed; retrieval is recorded |
| Contracts and artifacts | shared APIs | Small schema/publication dependencies | Multiple retained tools and external callers | Retain; avoid replacing tool-specific integrations with a generic catch-all |
| Devtools | maintainer commands | Validation/test libraries | Repository maintenance and CI | Retain; test user behavior rather than incidental prose |

### Migration boundaries

DenseArrays already owns prepared pools, constrained generation, saved arrays,
inspection, export and run recovery. Compare remaining DenseGen consumers by
operation before porting code. Study identities, cohorts, assay interpretation
and publication composition remain with their clients. Historical sampler
environments keep their immutable source and dependency pins.

The remaining DenseGen consumers need distinct handoffs before code removal:

| Consumer | Current dependency | Removal condition |
| --- | --- | --- |
| Retained-part reconstruction | Pinned historical DenseGen sampler and separate solver environment | Keep the exact historical source/dependency pins; do not redirect replay to current DenseArrays |
| Stress-study motif scans | PWM loading, motif digests, FASTA writing and FIMO parsing imported from DenseGen internals | Qualify a public motif-tool interface with identical records before migrating callers |
| Study analysis status | DenseGen analysis-surface inspection | Replace the status route only when the study consumes DenseArrays records |
| OPS and Notify | DenseGen contracts, event source and overlay adapters | Retire each producer integration when its last workflow moves |
| New array generation | DenseArrays 0.3.0 saved-run workflow | A standalone one-design run and saved-record verification passed locally |

DenseArrays owns current generation, inspection and export. It is not a drop-in
replacement for every motif utility or historical sampler. DenseGen's solver
demo failure remains recorded; its tutorial is not the entry point for new work.

Preserve retirement checkpoints in a verified Git bundle outside this checkout.
Save any necessary non-Git assets separately with checksums and their owner.
Restore and inspect that archive before removing a tool's tracked source,
entry points, exclusive dependencies, package data, CI registrations and routes.
The current package must not contain a second executable archived tree.
Removing source from the current revision does not rewrite Git history.

### Delivery sequence and checks

1. Complete consumer classification and record any removal blockers.
2. Add an OPAL-specific installation. A fresh environment must execute two
   local-table learning rounds without importing USR or unrelated tool runtimes;
   the existing USR route must still validate when explicitly selected.
3. Direct new construction work to DenseArrays and qualify each remaining
   DenseGen handoff before retirement. Preserve historical replay pins.
4. Verify the external archive, then remove eligible historical tools and all
   registration/dependency surfaces together. Test active consumers and inspect
   the built wheel for retired code and assets.
5. Put installation choices and active tools before contributor material.
   Use same-version relative links, and keep tool examples in tool READMEs.
6. Repair retained-tool diagnostics, OPAL handoff/terminology, and companion
   input access. Test missing-data notebook behavior and source-compatible
   bundles. Public release/access changes remain separate from local fixes.
7. Scope CI by changed behavior and replace brittle prose assertions with
   route, command and artifact checks. Refine active banners after lifecycle
   changes, preserving their dimensions and rounded corners.

### Progress

- Consumer/import inventory recorded; direct consumers prevent blanket retirement.
- OPAL-only wheel installed in a fresh environment and completed two local-table
  rounds with USR imports prohibited. Sequence-bearing exports and both saved
  selections verified. Optional USR configuration tests also passed.
- DenseGen handoff requirements are listed above; historical pins remain intact.
- LibShuffle, Billboard, NMF and TFKDAnalysis retired after verifying all 584
  source/local-result files against an external Git bundle and asset archive.
  Their dependencies are shared with retained tools, so retirement alone does
  not justify removing those packages from the full runtime.
- USR short-error delivery, OPAL launcher/export terminology, same-version
  navigation and documentation-only CI routing repaired.
- Five replacement companion bundles were published privately and downloaded
  through the normal fetch command with every file checksum verified. S6 now
  has a download; exploratory UMAP payloads are optional source-external data.
- Installed-wheel checks passed for the NumPy-only API, the two-round OPAL
  workflow and the full runtime. The broad core test run found documentation
  assertions and a missing module header; corrected surfaces passed focused
  reruns. External integration and GPU execution were not rerun.
- Repository links remain relative for branch review. Package builds resolve
  those links to release URLs in the PyPI description without duplicating prose.

Use the [repository change gate](../../repo-change-gate.md) for local validation.
No scientific input, frozen model, historical pin or author-managed manuscript
is changed by this migration.
