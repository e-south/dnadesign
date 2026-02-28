## Workspace Runbook Steps

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


**Last updated by:** cruncher-maintainers on 2026-02-23

### Contents
- [Overview](#overview)
- [Generated step table](#generated-step-table)

### Overview
This table is generated from workspace machine runbooks and is the standard cross-workspace step inventory.

### Generated step table
<!-- docs:runbook-steps:start -->
| Workspace | Step ID | Description | Command |
| --- | --- | --- | --- |
| `demo_multitf` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `demo_multitf` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `demo_multitf` | `fetch_sites_demo_local_meme` |  | `cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c configs/config.yaml` |
| `demo_multitf` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf lexA --tf cpxR --tf baeR --update -c configs/config.yaml` |
| `demo_multitf` | `fetch_sites_baer_chip_exo` |  | `cruncher fetch sites --source baer_chip_exo --tf baeR --update -c configs/config.yaml` |
| `demo_multitf` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id demo_merged_meme_oops_multitf -c configs/config.yaml` |
| `demo_multitf` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `demo_multitf` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `demo_multitf` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `demo_multitf` | `analyze_summary` |  | `cruncher analyze --summary -c configs/config.yaml` |
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
| `demo_pairwise` | `analyze_summary` |  | `cruncher analyze --summary -c configs/config.yaml` |
| `demo_pairwise` | `export_sequences_latest` |  | `cruncher export sequences --latest -c configs/config.yaml` |
| `demo_pairwise` | `render_logos` |  | `cruncher catalog logos --source demo_merged_meme_oops --set 1 -c configs/config.yaml` |
| `demo_pairwise` | `study_run_length_vs_score` | Sweep sequence_length with a step-2 grid plus base-config anchor and emit length-vs-score aggregates. | `cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite` |
| `demo_pairwise` | `study_run_diversity_vs_score` | Sweep diversity from 0.00 to 1.00 at fixed workspace sequence_length and emit diversity-vs-score aggregates. | `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite` |
| `multitf_baer_lexa_soxr` | `reset_workspace` |  | `cruncher workspaces reset --root . --confirm` |
| `multitf_baer_lexa_soxr` | `config_summary` |  | `cruncher config summary -c configs/config.yaml` |
| `multitf_baer_lexa_soxr` | `fetch_sites_demo_local_meme` |  | `cruncher fetch sites --source demo_local_meme --tf lexA --tf soxR --update -c configs/config.yaml` |
| `multitf_baer_lexa_soxr` | `fetch_sites_regulondb` |  | `cruncher fetch sites --source regulondb --tf baeR --tf lexA --tf soxR --update -c configs/config.yaml` |
| `multitf_baer_lexa_soxr` | `fetch_sites_baer_chip_exo` |  | `cruncher fetch sites --source baer_chip_exo --tf baeR --update -c configs/config.yaml` |
| `multitf_baer_lexa_soxr` | `discover_motifs` |  | `cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id multitf_baer_lexa_soxr_merged_meme_oops -c configs/config.yaml` |
| `multitf_baer_lexa_soxr` | `lock_targets` |  | `cruncher lock -c configs/config.yaml` |
| `multitf_baer_lexa_soxr` | `parse_run` |  | `cruncher parse --force-overwrite -c configs/config.yaml` |
| `multitf_baer_lexa_soxr` | `sample_run` |  | `cruncher sample --force-overwrite -c configs/config.yaml` |
| `multitf_baer_lexa_soxr` | `analyze_summary` |  | `cruncher analyze --summary -c configs/config.yaml` |
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
| `multitf_baer_lexa_soxr_soxs` | `analyze_summary` |  | `cruncher analyze --summary -c configs/config.yaml` |
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
| `multitf_cpxr_baer_lexa` | `analyze_summary` |  | `cruncher analyze --summary -c configs/config.yaml` |
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
| `pairwise_baer_lexa` | `analyze_summary` |  | `cruncher analyze --summary -c configs/config.yaml` |
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
| `pairwise_baer_soxr` | `analyze_summary` |  | `cruncher analyze --summary -c configs/config.yaml` |
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
| `pairwise_cpxr_baer` | `analyze_summary` |  | `cruncher analyze --summary -c configs/config.yaml` |
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
| `pairwise_cpxr_lexa` | `analyze_summary` |  | `cruncher analyze --summary -c configs/config.yaml` |
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
| `pairwise_cpxr_soxr` | `analyze_summary` |  | `cruncher analyze --summary -c configs/config.yaml` |
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
| `pairwise_laci_arac` | `analyze_summary` |  | `cruncher analyze --summary -c configs/config.yaml` |
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
| `pairwise_soxr_soxs` | `analyze_summary` |  | `cruncher analyze --summary -c configs/config.yaml` |
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
| `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs` | `analyze_summary` |  | `cruncher analyze --summary -c configs/config.yaml` |
| `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs` | `export_sequences_latest` |  | `cruncher export sequences --latest -c configs/config.yaml` |
| `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs` | `render_logos` |  | `cruncher catalog logos --source project_merged_meme_oops_all_tfs --set 1 -c configs/config.yaml` |
| `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs` | `study_run_length_vs_score` | Sweep sequence_length with a step-2 grid plus base-config anchor and emit length-vs-score aggregates. | `cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite` |
| `project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs` | `study_run_diversity_vs_score` | Sweep diversity from 0.00 to 1.00 at fixed workspace sequence_length and emit diversity-vs-score aggregates. | `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite` |
<!-- docs:runbook-steps:end -->
