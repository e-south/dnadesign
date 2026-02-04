# DenseGen Dev Journal

## 2026-02-03
- Started refactor to reduce DenseGen monoliths (orchestrator/CLI/config).
- Decisions: enforce strict no-fallback behavior across CLI/reporting; split config into submodules; extract pipeline phases; consolidate Stage-A modules; keep public imports stable via re-exports; add background progress + logos (prior commit).
- Extracted pipeline validation/usage helpers and Stage-A pool preparation into dedicated modules.
- Added shared sequence GC utility and corrected Stage-A import layering to eliminate circular imports.
- Extracted library artifact loading/writing into a dedicated pipeline helper with assertive parquet handling.
- Extracted resume-state loading into a dedicated pipeline helper.
- Extracted run-state initialization/writing into a dedicated pipeline helper.
- Split reporting into data and rendering modules; keep public facade stable.
- Added shared record value coercion helpers and removed duplicated list parsing.
- Fixed a run_metrics circular import by deferring plan_pools import to call site.
- Added shared event log parsing helpers and removed duplicated event loaders.

## 2026-02-04
- Fixed Stage-B stall handling for zero-solution generators: exit the library loop so stall detection triggers, and ensure max_consecutive_failures is enforced even with one_subsample_only.
- Tests: `uv run pytest -q src/dnadesign/densegen/tests/test_round_robin_chunk_cap.py`.
- Refactored `_process_plan_for_source` to delegate to `_run_stage_b_sampling`, separating plan setup, pool loading, and Stage-B execution.
- Tests: `uv run pytest -q src/dnadesign/densegen/tests/test_round_robin_chunk_cap.py src/dnadesign/densegen/tests/test_source_cache.py`.
- Profiling: Stage-A PWM build using `demo_meme_three_tfs` with `--max-seconds 2` (per-motif) wrote pools under `outputs/pools_profile`.
  Hotspots: `stage_a_mining.mine_pwm_candidates`, `pwm_fimo.run_fimo` + `subprocess.run`, `_generate_batch`.
  Stage-B profiling blocked: `neutral_bg` pool missing; need background pool build to proceed.
- Background pool negative selection now uses FIMO p-value threshold 1e-4 and scans both strands for hits.
- Demo config: increased `generation.sequence_length` to 64 so ethanol_ciprofloxacin plan meets minimum required bp (fixed + required motifs).
- Plan (densegen-refactor): holistic refactor + strict config default.
- Reverted demo `generation.sequence_length` to 60 to keep dense arrays fixed-length; Stage-B feasibility checks now warn (no abort) for short libraries or oversized motifs.
- Tests: `uv run pytest -q src/dnadesign/densegen/tests/test_stage_b_library_builder.py -k "allows_short_library_bp or allows_infeasible_min_required_len or allows_long_motif_length"`.
- Docs: run_metrics tier aggregations exclude background pools without tier labels.
- Refactored `TFSampler.generate_binding_site_library` with per-TF index caching and helper split.
- Moved Stage-A algorithmic modules to `core/stage_a`; adapters now re-export Stage-A modules.
- Stage-B sampling now reads typed config fields directly (no getattr defaults).
- Added DenseGen docs index at `docs/README.md`.
- Removed remaining Stage-B sampling fallbacks in `stage_b_library_builder`.
- Docs index now links to dev README + architecture.
- Assessment: background-pool negative selection uses FIMO p=1e-4 with allow_zero_hit_only; consider manifest metrics for reject counts + max_score_norm summaries.
- Assessment: background-pool FIMO exclusion runs once per motif per batch; consider single MEME run or cached motif files for throughput.
- Assessment: decide whether Stage-B feasibility errors for missing regulators should be treated as resample events instead of fatal.
- Stage-A progress settings now initialize during `dense run` Stage-A rebuilds; background pool progress streams in non-TTY mode.
- Added colorized DenseArray visuals in screen progress output with explicit `logging.visuals.tf_colors` mapping; no fallback for missing TF colors.
- Fixed DenseArray visual rendering when promoter constraints append motifs: extend visual labels with `__densegen__extra__`, update docs/demo config, and add tests.
- Fixed composition output to emit per-placement rows (solution_id/tf/tfbs/offset/length) so placement_map plots can run.
- Screen progress now includes an in-panel legend table plus global progress; shared dashboards prevent per-plan screen redraw spam.
- Console logging now suppresses fontTools INFO noise (kept in log file); demo palette updated to a more distinct set.
- Fixed-element labels now derive from `promoter_constraints[].name` (joined with `+`) and are used for DenseArray visuals/legend + optimizer regulator mapping.
- Screen legend now wraps into multiple rows to avoid wide panels and improve Live refresh behavior.
- Tests: added fixed-elements label helper test + custom extra-label visual test.
- Stage-B usage tracking now updates live during runs (tf_coverage/tfbs_coverage/tfbs_entropy reflect accepted solutions).
- Debugged Stage-B crash when visuals enabled: `logging.visuals.tf_colors` must map display TF labels (e.g., `lexA`,
  `background`); updated demo config + docs and aligned AGENTS auto-resume note.
- Ensured Matplotlib cache setup validates writability and demotes fontTools console noise; added tests for cache setup.
- [ ] Profile Stage-A and Stage-B (identify hotspots before algorithm changes).
- [x] Refactor `_process_plan_for_source` into helpers (plan setup, library build, solver loop, outputs).
- [x] Refactor `TFSampler.generate_binding_site_library` into validation, required-placement, sampling loop, metadata assembly, and add per-TF index to avoid repeated slicing.
- [x] Enforce strict config resolution by default (explicit `-c` or `DENSEGEN_CONFIG_PATH` only) and update tests/docs.
- [ ] Evaluate `USRWriter._load_existing_ids` vs id-index approach and decide on implementation.
- [x] Relocate Stage-A internals into `core/stage_a` and keep adapters thin.
- [x] Add docs index at `src/dnadesign/densegen/docs/README.md`.
- [x] Run DenseGen test subset and capture results.
- [ ] Profiling target workspace: TBD (need config path).
