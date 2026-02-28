## DenseGen workspace catalog

Use this catalog to pick a packaged workspace and run it through its local runbook.

| Workspace id | Purpose | Prereqs | Runtime profile | Tutorial | Runbook |
| --- | --- | --- | --- | --- | --- |
| `demo_tfbs_baseline` | Small binding-site baseline without PWM mining. | MILP solver | `sequence_length=100`, `plans=2`, `total_quota=100`, `library_size=10`, `max_accepted_per_library=10`, `no_progress_seconds_before_resample=10`, `max_consecutive_no_progress_resamples=60` | [TFBS baseline](../docs/tutorials/demo_tfbs_baseline.md) | [runbook](demo_tfbs_baseline/runbook.md) |
| `demo_sampling_baseline` | PWM artifact sampling baseline with ethanol/ciprofloxacin plans. | MILP solver + FIMO | `sequence_length=100`, `plans=2`, `total_quota=12`, `library_size=10`, `max_accepted_per_library=2`, `no_progress_seconds_before_resample=10`, `max_consecutive_no_progress_resamples=6` | [Sampling baseline](../docs/tutorials/demo_sampling_baseline.md) | [runbook](demo_sampling_baseline/runbook.md) |
| `study_constitutive_sigma_panel` | Constitutive sigma70 panel with fixed-element expansion and strict LacI/AraC exclusion. | MILP solver + FIMO | `sequence_length=60`, `plans=1`, `total_quota=48`, `library_size=16`, `max_accepted_per_library=2`, `no_progress_seconds_before_resample=10`, `max_consecutive_no_progress_resamples=25` | [Constitutive sigma panel](../docs/tutorials/study_constitutive_sigma_panel.md) | [runbook](study_constitutive_sigma_panel/runbook.md) |
| `study_stress_ethanol_cipro` | Multi-condition stress campaign with USR-ready outputs. | MILP solver + FIMO | `sequence_length=60`, `plans=3`, `total_quota=200`, `library_size=10`, `max_accepted_per_library=2`, `no_progress_seconds_before_resample=10`, `max_consecutive_no_progress_resamples=25` | [Stress ethanol/cipro](../docs/tutorials/study_stress_ethanol_cipro.md) | [runbook](study_stress_ethanol_cipro/runbook.md) |

### Runbook command pattern

Each packaged workspace supports the same workspace-local happy path:

```bash
# Enter the workspace directory so relative paths resolve correctly.
cd src/dnadesign/densegen/workspaces/<workspace_id>
# Execute the packaged workspace runbook sequence.
./runbook.sh
```

Use this when you want the didactic runbook sequence without writing long path-export one-liners.
