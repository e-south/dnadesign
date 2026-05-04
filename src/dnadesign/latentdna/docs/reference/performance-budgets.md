# Performance Budgets

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-02

The package currently treats promoter-study scale as a normal operating target, but checked-in verification remains contract-first and fixture-scale.

Current smoke coverage:

- `tests/contracts/test_study_workspace_contracts.py`
- `tests/contracts/test_promoter_reference_margin_template.py`
- `tests/cli/test_workspace_command.py`

Registered smoke slices:

- workspace snapshot publication
- canonical browser geometry inventory
- study-bound workspace initialization from promoter-study bindings
- deep workspace validation for the checked-in promoter template

Each smoke record emits:

- correctness summary
- contract-schema validation
- artifact-presence and docs-reference checks

Interpretation:

- these fixture-scale checks are regression guards for the refactored study seam and artifact contracts
- they are not substitutes for live promoter-study pressure runs on the real USR planes
- live study workspaces that rely on large USR directory overlays should opt into `digest_ledger.json` source provenance before regeneration when snapshot freshness latency becomes dominant
- the April 17, 2026 live `stress_ethanol_cipro_growth` snapshot pressure run measured warmed `workspace_snapshot()` at 1.12s mean across three runs after shared overlay-inventory freshness caching, down from 2.36s cumulative on the profiled pre-pass hot path
- the same study now measures warmed `workspace_snapshot()` at 0.22s mean across three runs after decoupling browser geometry assembly from full notebook context-audit work, with a fresh-process single run at 1.05s
- after splitting plot-semantics sidecar validation out of generic workspace loading and caching the canonical `output_root` on `WorkspaceContext`, the same study now measures warmed `workspace_snapshot()` at 0.18s mean across three runs and warmed `deliverable_status(representation_health_summary)` at 0.13s mean across three runs
- fresh-process `workspace_snapshot()` launches after that loader/control-plane split measured 1.75s on the first cold launch and about 1.00s on subsequent independent launches
- the profiled status path is now dominated by workspace-config YAML parsing plus deliverable freshness recursion rather than plot-semantics sidecar validation
- generated notebook runtime startup no longer opens view `matrix.npy` files for
  row-count or dimensionality text; those values come from the published
  `candidate_inventory` ledger and geometry-control rows
- direct notebook control-plane assembly now reads current view shapes from view
  manifest `stats.rows` / `stats.dims`, with `np.load(..., mmap_mode="r")` only
  as a legacy fallback for artifacts without manifest shape stats
- the May 2, 2026 live controls-build pressure run measured
  `stress_ethanol_cipro_growth` at 6.7704s / 5.4983s / 5.4861s with 45
  `np.load` calls before the shape-cache and manifest-stats pass, and 5.3724s /
  4.1354s / 4.0924s with 0 `np.load` calls after also removing duplicate Infer
  sidecar alias-table scans during schema inspection
- the same pass measured `regulondb_native_promoter_panel` controls builds at
  0.1016s / 0.0763s / 0.0821s with 16 `np.load` calls before, and 0.0815s /
  0.0571s / 0.0574s with 0 `np.load` calls after
- catalog-backed notebook controls assembly, the normal `notebook generate`
  path after catalog publication, reuses the catalog candidate inventory and
  measured 0.0277s / 0.0199s / 0.0200s for the live stress workspace with 0
  `np.load` calls
- the May 2, 2026 live thread-cap dogfood run on
  `stress_ethanol_cipro_growth` found that materializing
  `intermediate_embedding_7b_anchor_60bp` took 200.44s / 5.23 GB max RSS with
  2 BLAS/OpenMP threads and 190.87s / 5.92 GB max RSS with 4 threads; the small
  speedup does not justify a workspace-wide default of 4 on 16 GB machines
- the same pass found PCA reduction on the scorecard anchor sample at 5.97s /
  3.70 GB max RSS with 2 threads, 5.51s / 3.72 GB with 4 threads, and 5.58s /
  3.70 GB with the process default; 4 threads is acceptable for targeted
  reducer pressure runs, but 2 threads remains the safer default for full recipe
  operation on 16 GB machines
- stale-only recipe dogfood under a 4-thread cap completed
  `pre_assay_representation_triage_recipe` in 276.90s / 8.76 GB max RSS and
  `appendix_umap_gallery_recipe` in 1255.60s / 7.47 GB max RSS; seeded UMAP
  still ran single-worker because `random_state` forces `n_jobs=1`
- real-study pressure evidence should continue to be recorded in the development journal

See also:

- [../dev/journal.md](../dev/journal.md)
- [../workflows/promoter-study-representation-comparison.md](../workflows/promoter-study-representation-comparison.md)
