## DenseGen

![DenseGen banner](assets/densegen-banner.svg)

DenseGen wraps the [dense-arrays](https://github.com/e-south/dense-arrays) optimizer to execute full DNA library design workflows from workspace-local configuration and inputs. A run validates strict schema contracts, resolves input sources, builds Stage-A candidate pools when sampling is enabled, executes Stage-B library generation to quota, and writes reproducible artifacts (records, metadata, events, plots, and marimo notebooks) under the workspace `outputs/` tree. The package is designed for maintainable operations: explicit run state, strict fail-fast validation, deterministic path contracts, and direct integration points for downstream USR and Notify workflows.

### Documentation map

1. [Docs overview](docs/README.md)
2. [Workspace catalog](workspaces/catalog.md)
3. [TFBS baseline tutorial](docs/tutorials/demo_tfbs_baseline.md)
4. [Sampling baseline tutorial](docs/tutorials/demo_sampling_baseline.md)
5. [Constitutive sigma panel tutorial](docs/tutorials/study_constitutive_sigma_panel.md)
6. [Stress ethanol and ciprofloxacin tutorial](docs/tutorials/study_stress_ethanol_cipro.md)
7. [DenseGen to USR to Notify tutorial](docs/tutorials/demo_usr_notify.md)
8. [Quick checklist](docs/concepts/quick-checklist.md)
9. [CLI reference](docs/reference/cli.md)
10. [Config reference](docs/reference/config.md)
11. [Outputs reference](docs/reference/outputs.md)
12. [HPC runbook](docs/howto/hpc.md)
13. [BU SCC run guide](docs/howto/bu-scc.md)
14. [Architecture notes](docs/dev/architecture.md)
15. [Development journal](docs/dev/journal.md)
