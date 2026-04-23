## Workspace Runbook Steps

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


**Last updated by:** cruncher-maintainers on 2026-04-05

### Contents
- [Overview](#overview)
- [Generated step table](#generated-step-table)

### Overview
This table is generated from workspace machine runbooks and is the standard cross-workspace step inventory.

### Generated step table
<!-- docs:runbook-steps:start -->
| Workspace | Step ID | Description | Command |
| --- | --- | --- | --- |
| `demo_monotypic_baer` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `demo_monotypic_baer` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `demo_monotypic_baer` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf baeR --update -c configs/config.yaml` |
| `demo_monotypic_baer` | `fetch_sites_baer_chip_exo` |  | `cruncher fetch sites --source baer_chip_exo --tf baeR --update -c configs/config.yaml` |
| `demo_monotypic_baer` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id demo_baer_multiplicity_meme_oops -c configs/config.yaml` |
| `demo_monotypic_baer` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `demo_monotypic_baer` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `demo_monotypic_baer` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `demo_monotypic_baer` | `analyze_summary` | Analyze the occurrence-aware run and render the standard static plot suite, including the multi-offset elite showcase. | `cruncher analyze --summary -c configs/config.yaml` |
| `demo_monotypic_baer` | `show_sample_outputs` | Show the occurrence-aware run manifest and artifact inventory for the workspace root run. | `cruncher runs show outputs -c configs/config.yaml` |
| `demo_monotypic_baer` | `render_logos` |  | `cruncher catalog logos --source demo_baer_multiplicity_meme_oops --set 1 -c configs/config.yaml` |
| `demo_monotypic_baer` | `yiu_validate` | Validate the sample-backed YIU payload handoff against the selected occurrence-aware elite. | `cruncher yiu validate --spec configs/yiu/baer_monotypic_hit.yiu.yaml` |
| `demo_monotypic_baer` | `yiu_render` | Publish the YIU payload bundle under outputs/plots without a redundant workspace-level mirror. | `cruncher yiu render --spec configs/yiu/baer_monotypic_hit.yiu.yaml --force-overwrite --emit-renders` |
| `demo_monotypic_baer` | `yiu_show` | Inspect the published YIU payload bundle and integrity checks. | `cruncher yiu show --bundle outputs/plots/yiu__baer_monotypic_hit` |
| `demo_monotypic_cpxr` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `demo_monotypic_cpxr` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `demo_monotypic_cpxr` | `fetch_sites_demo_local_meme` |  | `cruncher fetch sites --source demo_local_meme --tf cpxR --update -c configs/config.yaml` |
| `demo_monotypic_cpxr` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf cpxR --update -c configs/config.yaml` |
| `demo_monotypic_cpxr` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id demo_cpxr_multiplicity_meme_oops -c configs/config.yaml` |
| `demo_monotypic_cpxr` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `demo_monotypic_cpxr` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `demo_monotypic_cpxr` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `demo_monotypic_cpxr` | `analyze_summary` | Analyze the occurrence-aware run and render the standard static plot suite, including the multi-offset elite showcase. | `cruncher analyze --summary -c configs/config.yaml` |
| `demo_monotypic_cpxr` | `show_sample_outputs` | Show the occurrence-aware run manifest and artifact inventory for the workspace root run. | `cruncher runs show outputs -c configs/config.yaml` |
| `demo_monotypic_cpxr` | `render_logos` |  | `cruncher catalog logos --source demo_cpxr_multiplicity_meme_oops --set 1 -c configs/config.yaml` |
| `demo_monotypic_cpxr` | `yiu_validate` | Validate the sample-backed YIU payload handoff against the selected occurrence-aware elite. | `cruncher yiu validate --spec configs/yiu/cpxr_monotypic_hit.yiu.yaml` |
| `demo_monotypic_cpxr` | `yiu_render` | Publish the YIU payload bundle under outputs/plots without a redundant workspace-level mirror. | `cruncher yiu render --spec configs/yiu/cpxr_monotypic_hit.yiu.yaml --force-overwrite --emit-renders` |
| `demo_monotypic_cpxr` | `yiu_show` | Inspect the published YIU payload bundle and integrity checks. | `cruncher yiu show --bundle outputs/plots/yiu__cpxr_monotypic_hit` |
| `demo_monotypic_lexa` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `demo_monotypic_lexa` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `demo_monotypic_lexa` | `fetch_sites_demo_local_meme` |  | `cruncher fetch sites --source demo_local_meme --tf lexA --update -c configs/config.yaml` |
| `demo_monotypic_lexa` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf lexA --update -c configs/config.yaml` |
| `demo_monotypic_lexa` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id demo_lexa_multiplicity_meme_oops -c configs/config.yaml` |
| `demo_monotypic_lexa` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `demo_monotypic_lexa` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `demo_monotypic_lexa` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `demo_monotypic_lexa` | `analyze_summary` | Analyze the occurrence-aware run and render the standard static plot suite, including the multi-offset elite showcase. | `cruncher analyze --summary -c configs/config.yaml` |
| `demo_monotypic_lexa` | `show_sample_outputs` | Show the occurrence-aware run manifest and artifact inventory for the workspace root run. | `cruncher runs show outputs -c configs/config.yaml` |
| `demo_monotypic_lexa` | `render_logos` |  | `cruncher catalog logos --source demo_lexa_multiplicity_meme_oops --set 1 -c configs/config.yaml` |
| `demo_monotypic_lexa` | `yiu_validate` | Validate the sample-backed YIU payload handoff against the selected occurrence-aware elite. | `cruncher yiu validate --spec configs/yiu/lexa_monotypic_hit.yiu.yaml` |
| `demo_monotypic_lexa` | `yiu_render` | Publish the YIU payload bundle under outputs/plots without a redundant workspace-level mirror. | `cruncher yiu render --spec configs/yiu/lexa_monotypic_hit.yiu.yaml --force-overwrite --emit-renders` |
| `demo_monotypic_lexa` | `yiu_show` | Inspect the published YIU payload bundle and integrity checks. | `cruncher yiu show --bundle outputs/plots/yiu__lexa_monotypic_hit` |
| `demo_monotypic_soxr` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `demo_monotypic_soxr` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `demo_monotypic_soxr` | `fetch_sites_demo_local_meme` |  | `cruncher fetch sites --source demo_local_meme --tf soxR --update -c configs/config.yaml` |
| `demo_monotypic_soxr` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf soxR --update -c configs/config.yaml` |
| `demo_monotypic_soxr` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id demo_soxr_multiplicity_meme_oops -c configs/config.yaml` |
| `demo_monotypic_soxr` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `demo_monotypic_soxr` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `demo_monotypic_soxr` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `demo_monotypic_soxr` | `analyze_summary` | Analyze the occurrence-aware run and render the standard static plot suite, including the multi-offset elite showcase. | `cruncher analyze --summary -c configs/config.yaml` |
| `demo_monotypic_soxr` | `show_sample_outputs` | Show the occurrence-aware run manifest and artifact inventory for the workspace root run. | `cruncher runs show outputs -c configs/config.yaml` |
| `demo_monotypic_soxr` | `render_logos` |  | `cruncher catalog logos --source demo_soxr_multiplicity_meme_oops --set 1 -c configs/config.yaml` |
| `demo_monotypic_soxr` | `yiu_validate` | Validate the sample-backed YIU payload handoff against the selected occurrence-aware elite. | `cruncher yiu validate --spec configs/yiu/soxr_monotypic_hit.yiu.yaml` |
| `demo_monotypic_soxr` | `yiu_render` | Publish the YIU payload bundle under outputs/plots without a redundant workspace-level mirror. | `cruncher yiu render --spec configs/yiu/soxr_monotypic_hit.yiu.yaml --force-overwrite --emit-renders` |
| `demo_monotypic_soxr` | `yiu_show` | Inspect the published YIU payload bundle and integrity checks. | `cruncher yiu show --bundle outputs/plots/yiu__soxr_monotypic_hit` |
| `demo_monotypic_soxs` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `demo_monotypic_soxs` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `demo_monotypic_soxs` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf soxS --update -c configs/config.yaml` |
| `demo_monotypic_soxs` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id demo_soxs_multiplicity_meme_oops -c configs/config.yaml` |
| `demo_monotypic_soxs` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `demo_monotypic_soxs` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `demo_monotypic_soxs` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `demo_monotypic_soxs` | `analyze_summary` | Analyze the occurrence-aware run and render the standard static plot suite, including the multi-offset elite showcase. | `cruncher analyze --summary -c configs/config.yaml` |
| `demo_monotypic_soxs` | `show_sample_outputs` | Show the occurrence-aware run manifest and artifact inventory for the workspace root run. | `cruncher runs show outputs -c configs/config.yaml` |
| `demo_monotypic_soxs` | `render_logos` |  | `cruncher catalog logos --source demo_soxs_multiplicity_meme_oops --set 1 -c configs/config.yaml` |
| `demo_monotypic_soxs` | `yiu_validate` | Validate the sample-backed YIU payload handoff against the selected occurrence-aware elite. | `cruncher yiu validate --spec configs/yiu/soxs_monotypic_hit.yiu.yaml` |
| `demo_monotypic_soxs` | `yiu_render` | Publish the YIU payload bundle under outputs/plots without a redundant workspace-level mirror. | `cruncher yiu render --spec configs/yiu/soxs_monotypic_hit.yiu.yaml --force-overwrite --emit-renders` |
| `demo_monotypic_soxs` | `yiu_show` | Inspect the published YIU payload bundle and integrity checks. | `cruncher yiu show --bundle outputs/plots/yiu__soxs_monotypic_hit` |
| `demo_monotypic_tetr` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `demo_monotypic_tetr` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `demo_monotypic_tetr` | `fetch_motifs_westmann` |  | `cruncher fetch motifs --source westmann_tetr_mitomi --tf tetR --update -c configs/config.yaml` |
| `demo_monotypic_tetr` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `demo_monotypic_tetr` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `demo_monotypic_tetr` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `demo_monotypic_tetr` | `analyze_summary` | Analyze the occurrence-aware run and render the standard static plot suite, including the multi-offset elite showcase. | `cruncher analyze --summary -c configs/config.yaml` |
| `demo_monotypic_tetr` | `show_sample_outputs` | Show the occurrence-aware run manifest and artifact inventory for the workspace root run. | `cruncher runs show outputs -c configs/config.yaml` |
| `demo_monotypic_tetr` | `render_logos` |  | `cruncher catalog logos --source westmann_tetr_mitomi --set 1 -c configs/config.yaml` |
| `demo_monotypic_tetr` | `export_meme` | Export the normalized TetR PWM as a minimal MEME artifact under outputs/artifacts/meme for downstream motif-tool interoperability. | `cruncher catalog export-meme --set 1 --source westmann_tetr_mitomi -c configs/config.yaml` |
| `demo_monotypic_tetr` | `yiu_validate` | Validate the sample-backed YIU payload handoff against the selected occurrence-aware elite. | `cruncher yiu validate --spec configs/yiu/tetr_monotypic_hit.yiu.yaml` |
| `demo_monotypic_tetr` | `yiu_render` | Publish the YIU payload bundle under outputs/plots without a redundant workspace-level mirror. | `cruncher yiu render --spec configs/yiu/tetr_monotypic_hit.yiu.yaml --force-overwrite --emit-renders` |
| `demo_monotypic_tetr` | `yiu_show` | Inspect the published YIU payload bundle and integrity checks. | `cruncher yiu show --bundle outputs/plots/yiu__tetr_monotypic_hit` |
| `demo_multitf` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `demo_multitf` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `demo_multitf` | `fetch_sites_demo_local_meme` |  | `cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c configs/config.yaml` |
| `demo_multitf` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf lexA --tf cpxR --tf baeR --update -c configs/config.yaml` |
| `demo_multitf` | `fetch_sites_baer_chip_exo` |  | `cruncher fetch sites --source baer_chip_exo --tf baeR --update -c configs/config.yaml` |
| `demo_multitf` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id demo_merged_meme_oops_multitf -c configs/config.yaml` |
| `demo_multitf` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `demo_multitf` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `demo_multitf` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `demo_multitf` | `analyze_summary` | Analyze run outputs; set analysis.trajectory_video.enabled=true in configs/config.yaml to emit plots/chain_trajectory_video.mp4. | `cruncher analyze --summary -c configs/config.yaml` |
| `demo_multitf` | `export_sequences_latest` |  | `cruncher export sequences --latest -c configs/config.yaml` |
| `demo_multitf` | `render_logos` |  | `cruncher catalog logos --source demo_merged_meme_oops_multitf --set 1 -c configs/config.yaml` |
| `demo_multitf` | `study_run_length_vs_score` | Sweep sequence_length with a step-2 grid plus base-config anchor and emit length-vs-score aggregates. | `cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite` |
| `demo_multitf` | `study_run_diversity_vs_score` | Sweep diversity from 0.00 to 1.00 at fixed workspace sequence_length and emit diversity-vs-score aggregates. | `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite` |
| `demo_pairwise` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `demo_pairwise` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `demo_pairwise` | `fetch_sites_demo_local_meme` |  | `cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c configs/config.yaml` |
| `demo_pairwise` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf lexA --tf cpxR --update -c configs/config.yaml` |
| `demo_pairwise` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id demo_merged_meme_oops -c configs/config.yaml` |
| `demo_pairwise` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `demo_pairwise` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `demo_pairwise` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `demo_pairwise` | `analyze_summary` | Analyze run outputs and emit plots/chain_trajectory_video.mp4 from the enabled trajectory_video config. | `cruncher analyze --summary -c configs/config.yaml` |
| `demo_pairwise` | `export_sequences_latest` |  | `cruncher export sequences --latest -c configs/config.yaml` |
| `demo_pairwise` | `render_logos` |  | `cruncher catalog logos --source demo_merged_meme_oops --set 1 -c configs/config.yaml` |
| `demo_pairwise` | `study_run_length_vs_score` | Sweep sequence_length with a step-2 grid plus base-config anchor and emit length-vs-score aggregates. | `cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite` |
| `demo_pairwise` | `study_run_diversity_vs_score` | Sweep diversity from 0.00 to 1.00 at fixed workspace sequence_length and emit diversity-vs-score aggregates. | `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite` |
| `de033` | `snapback_released_target_search` | Search the operational dual-enzyme 0/3/3 lane against the full built-in local nickase catalog and the release-enzyme preset. | `cruncher snapback released-target-search --workspace-root . --nick-preset neb_nicking_v1 --nick-additional-preset thermo_nicking_v1 --release-preset type_iis_release_v1 --nick-boundary 0 --paired-bp 3 --cap-nt 3 --json` |
| `de033` | `snapback_released_solve` | Materialize whole-catalog allowed released-product near or exact hits and emit one rendered triptych per hit. | `cruncher snapback released-solve --workspace-root . --nick-preset neb_nicking_v1 --nick-additional-preset thermo_nicking_v1 --release-preset type_iis_release_v1 --nick-boundary 0 --paired-bp 3 --cap-nt 3 --run-dir outputs/released_solve --materialize-top-k 8 --render-format pdf --emit-renders --force-overwrite --json` |
| `demo_yiu_payload` | `yiu_validate` | Validate the checked-in user-sequence YIU demo spec. | `cruncher yiu validate --spec configs/yiu/example_payload.yiu.yaml` |
| `demo_yiu_payload` | `yiu_render` | Publish the deterministic user-sequence YIU v4 payload bundle and render the standard views. | `cruncher yiu render --spec configs/yiu/example_payload.yiu.yaml --force-overwrite --emit-renders` |
| `demo_yiu_payload` | `yiu_show` | Inspect the published user-sequence payload bundle and integrity checks. | `cruncher yiu show --bundle outputs/example_payload` |
| `multitf_baer_lexa_soxr` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `multitf_baer_lexa_soxr` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `multitf_baer_lexa_soxr` | `fetch_sites_demo_local_meme` |  | `cruncher fetch sites --source demo_local_meme --tf lexA --tf soxR --update -c configs/config.yaml` |
| `multitf_baer_lexa_soxr` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf baeR --tf lexA --tf soxR --update -c configs/config.yaml` |
| `multitf_baer_lexa_soxr` | `fetch_sites_baer_chip_exo` |  | `cruncher fetch sites --source baer_chip_exo --tf baeR --update -c configs/config.yaml` |
| `multitf_baer_lexa_soxr` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id multitf_baer_lexa_soxr_merged_meme_oops -c configs/config.yaml` |
| `multitf_baer_lexa_soxr` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `multitf_baer_lexa_soxr` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `multitf_baer_lexa_soxr` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `multitf_baer_lexa_soxr` | `analyze_summary` | Analyze run outputs; set analysis.trajectory_video.enabled=true in configs/config.yaml to emit plots/chain_trajectory_video.mp4. | `cruncher analyze --summary -c configs/config.yaml` |
| `multitf_baer_lexa_soxr` | `export_sequences_latest` |  | `cruncher export sequences --latest -c configs/config.yaml` |
| `multitf_baer_lexa_soxr` | `render_logos` |  | `cruncher catalog logos --source multitf_baer_lexa_soxr_merged_meme_oops --set 1 -c configs/config.yaml` |
| `multitf_baer_lexa_soxr` | `study_run_length_vs_score` | Sweep sequence_length with a step-2 grid plus base-config anchor and emit length-vs-score aggregates. | `cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite` |
| `multitf_baer_lexa_soxr` | `study_run_diversity_vs_score` | Sweep diversity from 0.00 to 1.00 at fixed workspace sequence_length and emit diversity-vs-score aggregates. | `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite` |
| `multitf_baer_lexa_soxr_soxs` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `multitf_baer_lexa_soxr_soxs` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `multitf_baer_lexa_soxr_soxs` | `fetch_sites_demo_local_meme` |  | `cruncher fetch sites --source demo_local_meme --tf lexA --tf soxR --update -c configs/config.yaml` |
| `multitf_baer_lexa_soxr_soxs` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf baeR --tf lexA --tf soxR --tf soxS --update -c configs/config.yaml` |
| `multitf_baer_lexa_soxr_soxs` | `fetch_sites_baer_chip_exo` |  | `cruncher fetch sites --source baer_chip_exo --tf baeR --update -c configs/config.yaml` |
| `multitf_baer_lexa_soxr_soxs` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id multitf_baer_lexa_soxr_soxs_merged_meme_oops -c configs/config.yaml` |
| `multitf_baer_lexa_soxr_soxs` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `multitf_baer_lexa_soxr_soxs` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `multitf_baer_lexa_soxr_soxs` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `multitf_baer_lexa_soxr_soxs` | `analyze_summary` | Analyze run outputs; set analysis.trajectory_video.enabled=true in configs/config.yaml to emit plots/chain_trajectory_video.mp4. | `cruncher analyze --summary -c configs/config.yaml` |
| `multitf_baer_lexa_soxr_soxs` | `export_sequences_latest` |  | `cruncher export sequences --latest -c configs/config.yaml` |
| `multitf_baer_lexa_soxr_soxs` | `render_logos` |  | `cruncher catalog logos --source multitf_baer_lexa_soxr_soxs_merged_meme_oops --set 1 -c configs/config.yaml` |
| `multitf_baer_lexa_soxr_soxs` | `study_run_length_vs_score` | Sweep sequence_length with a step-2 grid plus base-config anchor and emit length-vs-score aggregates. | `cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite` |
| `multitf_baer_lexa_soxr_soxs` | `study_run_diversity_vs_score` | Sweep diversity from 0.00 to 1.00 at fixed workspace sequence_length and emit diversity-vs-score aggregates. | `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite` |
| `multitf_cpxr_baer_lexa` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `multitf_cpxr_baer_lexa` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `multitf_cpxr_baer_lexa` | `fetch_sites_demo_local_meme` |  | `cruncher fetch sites --source demo_local_meme --tf cpxR --tf lexA --update -c configs/config.yaml` |
| `multitf_cpxr_baer_lexa` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf cpxR --tf baeR --tf lexA --update -c configs/config.yaml` |
| `multitf_cpxr_baer_lexa` | `fetch_sites_baer_chip_exo` |  | `cruncher fetch sites --source baer_chip_exo --tf baeR --update -c configs/config.yaml` |
| `multitf_cpxr_baer_lexa` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id multitf_cpxr_baer_lexa_merged_meme_oops -c configs/config.yaml` |
| `multitf_cpxr_baer_lexa` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `multitf_cpxr_baer_lexa` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `multitf_cpxr_baer_lexa` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `multitf_cpxr_baer_lexa` | `analyze_summary` | Analyze run outputs; set analysis.trajectory_video.enabled=true in configs/config.yaml to emit plots/chain_trajectory_video.mp4. | `cruncher analyze --summary -c configs/config.yaml` |
| `multitf_cpxr_baer_lexa` | `export_sequences_latest` |  | `cruncher export sequences --latest -c configs/config.yaml` |
| `multitf_cpxr_baer_lexa` | `render_logos` |  | `cruncher catalog logos --source multitf_cpxr_baer_lexa_merged_meme_oops --set 1 -c configs/config.yaml` |
| `multitf_cpxr_baer_lexa` | `study_run_length_vs_score` | Sweep sequence_length with a step-2 grid plus base-config anchor and emit length-vs-score aggregates. | `cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite` |
| `multitf_cpxr_baer_lexa` | `study_run_diversity_vs_score` | Sweep diversity from 0.00 to 1.00 at fixed workspace sequence_length and emit diversity-vs-score aggregates. | `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite` |
| `pairwise_baer_lexa` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `pairwise_baer_lexa` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `pairwise_baer_lexa` | `fetch_sites_demo_local_meme` |  | `cruncher fetch sites --source demo_local_meme --tf lexA --update -c configs/config.yaml` |
| `pairwise_baer_lexa` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf baeR --tf lexA --update -c configs/config.yaml` |
| `pairwise_baer_lexa` | `fetch_sites_baer_chip_exo` |  | `cruncher fetch sites --source baer_chip_exo --tf baeR --update -c configs/config.yaml` |
| `pairwise_baer_lexa` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id pairwise_baer_lexa_merged_meme_oops -c configs/config.yaml` |
| `pairwise_baer_lexa` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `pairwise_baer_lexa` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `pairwise_baer_lexa` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `pairwise_baer_lexa` | `analyze_summary` | Analyze run outputs; set analysis.trajectory_video.enabled=true in configs/config.yaml to emit plots/chain_trajectory_video.mp4. | `cruncher analyze --summary -c configs/config.yaml` |
| `pairwise_baer_lexa` | `export_sequences_latest` |  | `cruncher export sequences --latest -c configs/config.yaml` |
| `pairwise_baer_lexa` | `render_logos` |  | `cruncher catalog logos --source pairwise_baer_lexa_merged_meme_oops --set 1 -c configs/config.yaml` |
| `pairwise_baer_lexa` | `study_run_length_vs_score` | Sweep sequence_length with a step-2 grid plus base-config anchor and emit length-vs-score aggregates. | `cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite` |
| `pairwise_baer_lexa` | `study_run_diversity_vs_score` | Sweep diversity from 0.00 to 1.00 at fixed workspace sequence_length and emit diversity-vs-score aggregates. | `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite` |
| `pairwise_baer_soxr` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `pairwise_baer_soxr` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `pairwise_baer_soxr` | `fetch_sites_demo_local_meme` |  | `cruncher fetch sites --source demo_local_meme --tf soxR --update -c configs/config.yaml` |
| `pairwise_baer_soxr` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf baeR --tf soxR --update -c configs/config.yaml` |
| `pairwise_baer_soxr` | `fetch_sites_baer_chip_exo` |  | `cruncher fetch sites --source baer_chip_exo --tf baeR --update -c configs/config.yaml` |
| `pairwise_baer_soxr` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id pairwise_baer_soxr_merged_meme_oops -c configs/config.yaml` |
| `pairwise_baer_soxr` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `pairwise_baer_soxr` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `pairwise_baer_soxr` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `pairwise_baer_soxr` | `analyze_summary` | Analyze run outputs; set analysis.trajectory_video.enabled=true in configs/config.yaml to emit plots/chain_trajectory_video.mp4. | `cruncher analyze --summary -c configs/config.yaml` |
| `pairwise_baer_soxr` | `export_sequences_latest` |  | `cruncher export sequences --latest -c configs/config.yaml` |
| `pairwise_baer_soxr` | `render_logos` |  | `cruncher catalog logos --source pairwise_baer_soxr_merged_meme_oops --set 1 -c configs/config.yaml` |
| `pairwise_baer_soxr` | `study_run_length_vs_score` | Sweep sequence_length with a step-2 grid plus base-config anchor and emit length-vs-score aggregates. | `cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite` |
| `pairwise_baer_soxr` | `study_run_diversity_vs_score` | Sweep diversity from 0.00 to 1.00 at fixed workspace sequence_length and emit diversity-vs-score aggregates. | `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite` |
| `pairwise_cpxr_baer` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `pairwise_cpxr_baer` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `pairwise_cpxr_baer` | `fetch_sites_demo_local_meme` |  | `cruncher fetch sites --source demo_local_meme --tf cpxR --update -c configs/config.yaml` |
| `pairwise_cpxr_baer` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf cpxR --tf baeR --update -c configs/config.yaml` |
| `pairwise_cpxr_baer` | `fetch_sites_baer_chip_exo` |  | `cruncher fetch sites --source baer_chip_exo --tf baeR --update -c configs/config.yaml` |
| `pairwise_cpxr_baer` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id pairwise_cpxr_baer_merged_meme_oops -c configs/config.yaml` |
| `pairwise_cpxr_baer` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `pairwise_cpxr_baer` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `pairwise_cpxr_baer` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `pairwise_cpxr_baer` | `analyze_summary` | Analyze run outputs; set analysis.trajectory_video.enabled=true in configs/config.yaml to emit plots/chain_trajectory_video.mp4. | `cruncher analyze --summary -c configs/config.yaml` |
| `pairwise_cpxr_baer` | `export_sequences_latest` |  | `cruncher export sequences --latest -c configs/config.yaml` |
| `pairwise_cpxr_baer` | `render_logos` |  | `cruncher catalog logos --source pairwise_cpxr_baer_merged_meme_oops --set 1 -c configs/config.yaml` |
| `pairwise_cpxr_baer` | `study_run_length_vs_score` | Sweep sequence_length with a step-2 grid plus base-config anchor and emit length-vs-score aggregates. | `cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite` |
| `pairwise_cpxr_baer` | `study_run_diversity_vs_score` | Sweep diversity from 0.00 to 1.00 at fixed workspace sequence_length and emit diversity-vs-score aggregates. | `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite` |
| `pairwise_cpxr_lexa` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `pairwise_cpxr_lexa` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `pairwise_cpxr_lexa` | `fetch_sites_demo_local_meme` |  | `cruncher fetch sites --source demo_local_meme --tf cpxR --tf lexA --update -c configs/config.yaml` |
| `pairwise_cpxr_lexa` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf cpxR --tf lexA --update -c configs/config.yaml` |
| `pairwise_cpxr_lexa` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id pairwise_cpxr_lexa_merged_meme_oops -c configs/config.yaml` |
| `pairwise_cpxr_lexa` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `pairwise_cpxr_lexa` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `pairwise_cpxr_lexa` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `pairwise_cpxr_lexa` | `analyze_summary` | Analyze run outputs; set analysis.trajectory_video.enabled=true in configs/config.yaml to emit plots/chain_trajectory_video.mp4. | `cruncher analyze --summary -c configs/config.yaml` |
| `pairwise_cpxr_lexa` | `export_sequences_latest` |  | `cruncher export sequences --latest -c configs/config.yaml` |
| `pairwise_cpxr_lexa` | `render_logos` |  | `cruncher catalog logos --source pairwise_cpxr_lexa_merged_meme_oops --set 1 -c configs/config.yaml` |
| `pairwise_cpxr_lexa` | `study_run_length_vs_score` | Sweep sequence_length with a step-2 grid plus base-config anchor and emit length-vs-score aggregates. | `cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite` |
| `pairwise_cpxr_lexa` | `study_run_diversity_vs_score` | Sweep diversity from 0.00 to 1.00 at fixed workspace sequence_length and emit diversity-vs-score aggregates. | `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite` |
| `pairwise_cpxr_soxr` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `pairwise_cpxr_soxr` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `pairwise_cpxr_soxr` | `fetch_sites_demo_local_meme` |  | `cruncher fetch sites --source demo_local_meme --tf cpxR --tf soxR --update -c configs/config.yaml` |
| `pairwise_cpxr_soxr` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf cpxR --tf soxR --update -c configs/config.yaml` |
| `pairwise_cpxr_soxr` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id pairwise_cpxr_soxr_merged_meme_oops -c configs/config.yaml` |
| `pairwise_cpxr_soxr` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `pairwise_cpxr_soxr` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `pairwise_cpxr_soxr` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `pairwise_cpxr_soxr` | `analyze_summary` | Analyze run outputs; set analysis.trajectory_video.enabled=true in configs/config.yaml to emit plots/chain_trajectory_video.mp4. | `cruncher analyze --summary -c configs/config.yaml` |
| `pairwise_cpxr_soxr` | `export_sequences_latest` |  | `cruncher export sequences --latest -c configs/config.yaml` |
| `pairwise_cpxr_soxr` | `render_logos` |  | `cruncher catalog logos --source pairwise_cpxr_soxr_merged_meme_oops --set 1 -c configs/config.yaml` |
| `pairwise_cpxr_soxr` | `study_run_length_vs_score` | Sweep sequence_length with a step-2 grid plus base-config anchor and emit length-vs-score aggregates. | `cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite` |
| `pairwise_cpxr_soxr` | `study_run_diversity_vs_score` | Sweep diversity from 0.00 to 1.00 at fixed workspace sequence_length and emit diversity-vs-score aggregates. | `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite` |
| `pairwise_laci_arac` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `pairwise_laci_arac` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `pairwise_laci_arac` | `fetch_sites_demo_local_meme` |  | `cruncher fetch sites --source demo_local_meme --tf lacI --update -c configs/config.yaml` |
| `pairwise_laci_arac` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf lacI --tf araC --update -c configs/config.yaml` |
| `pairwise_laci_arac` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id pairwise_laci_arac_merged_meme_oops -c configs/config.yaml` |
| `pairwise_laci_arac` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `pairwise_laci_arac` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `pairwise_laci_arac` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `pairwise_laci_arac` | `analyze_summary` | Analyze run outputs; set analysis.trajectory_video.enabled=true in configs/config.yaml to emit plots/chain_trajectory_video.mp4. | `cruncher analyze --summary -c configs/config.yaml` |
| `pairwise_laci_arac` | `export_sequences_latest` |  | `cruncher export sequences --latest -c configs/config.yaml` |
| `pairwise_laci_arac` | `render_logos` |  | `cruncher catalog logos --source pairwise_laci_arac_merged_meme_oops --set 1 -c configs/config.yaml` |
| `pairwise_laci_arac` | `export_densegen` |  | `cruncher catalog export-densegen --set 1 --densegen-workspace study_constitutive_sigma_panel -c configs/config.yaml` |
| `pairwise_laci_arac` | `study_run_length_vs_score` | Sweep sequence_length with a step-2 grid plus base-config anchor and emit length-vs-score aggregates. | `cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite` |
| `pairwise_laci_arac` | `study_run_diversity_vs_score` | Sweep diversity from 0.00 to 1.00 at fixed workspace sequence_length and emit diversity-vs-score aggregates. | `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite` |
| `pairwise_soxr_soxs` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `pairwise_soxr_soxs` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `pairwise_soxr_soxs` | `fetch_sites_demo_local_meme` |  | `cruncher fetch sites --source demo_local_meme --tf soxR --update -c configs/config.yaml` |
| `pairwise_soxr_soxs` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf soxR --tf soxS --update -c configs/config.yaml` |
| `pairwise_soxr_soxs` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id pairwise_soxr_soxs_merged_meme_oops -c configs/config.yaml` |
| `pairwise_soxr_soxs` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `pairwise_soxr_soxs` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `pairwise_soxr_soxs` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `pairwise_soxr_soxs` | `analyze_summary` | Analyze run outputs; set analysis.trajectory_video.enabled=true in configs/config.yaml to emit plots/chain_trajectory_video.mp4. | `cruncher analyze --summary -c configs/config.yaml` |
| `pairwise_soxr_soxs` | `export_sequences_latest` |  | `cruncher export sequences --latest -c configs/config.yaml` |
| `pairwise_soxr_soxs` | `render_logos` |  | `cruncher catalog logos --source pairwise_soxr_soxs_merged_meme_oops --set 1 -c configs/config.yaml` |
| `pairwise_soxr_soxs` | `study_run_length_vs_score` | Sweep sequence_length with a step-2 grid plus base-config anchor and emit length-vs-score aggregates. | `cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite` |
| `pairwise_soxr_soxs` | `study_run_diversity_vs_score` | Sweep diversity from 0.00 to 1.00 at fixed workspace sequence_length and emit diversity-vs-score aggregates. | `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite` |
| `portfolios` | `portfolio_run_master_all_workspaces` |  | `cruncher portfolio run --spec configs/master_all_workspaces.portfolio.yaml --prepare-ready skip --force-overwrite` |
| `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs` | `fetch_sites_demo_local_meme` |  | `cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --tf rcdA --tf lrp --tf acrR --tf soxR --update -c configs/config.yaml` |
| `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf lexA --tf cpxR --tf baeR --tf rcdA --tf lrp --tf fur --tf fnr --tf acrR --tf soxR --tf soxS --update -c configs/config.yaml` |
| `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs` | `fetch_sites_baer_chip_exo` |  | `cruncher fetch sites --source baer_chip_exo --tf baeR --update -c configs/config.yaml` |
| `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id project_merged_meme_oops_all_tfs -c configs/config.yaml` |
| `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs` | `analyze_summary` | Analyze run outputs; set analysis.trajectory_video.enabled=true in configs/config.yaml to emit plots/chain_trajectory_video.mp4. | `cruncher analyze --summary -c configs/config.yaml` |
| `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs` | `export_sequences_latest` |  | `cruncher export sequences --latest -c configs/config.yaml` |
| `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs` | `render_logos` |  | `cruncher catalog logos --source project_merged_meme_oops_all_tfs --set 1 -c configs/config.yaml` |
| `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs` | `study_run_length_vs_score` | Sweep sequence_length with a step-2 grid plus base-config anchor and emit length-vs-score aggregates. | `cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite` |
| `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs` | `study_run_diversity_vs_score` | Sweep diversity from 0.00 to 1.00 at fixed workspace sequence_length and emit diversity-vs-score aggregates. | `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite` |
<!-- docs:runbook-steps:end -->
