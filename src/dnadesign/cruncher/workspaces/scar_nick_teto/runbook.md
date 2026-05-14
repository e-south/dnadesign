## scar_nick_teto workspace

This workspace owns TetO/retron scar-nick base-junction source outputs for the
`retron_hairpin_design` study.

Use one workspace for the panel. Do not create one workspace per hit. Distinct
release-enzyme panels live as separate specs under `configs/scar_nick/` and
write separate generated bundles under `outputs/scar_nick/`.

### Specs

- `configs/scar_nick/teto_upstream_processing.bbsI_hf.scar_nick.yaml`
- `configs/scar_nick/teto_upstream_processing.paqci_core_panel.scar_nick.yaml`
- `configs/runbook.yaml` records the Cruncher-only machine steps for validate,
  design, and show. BaseRender commands remain optional follow-up commands
  below.

### Step-by-Step Commands

    set -euo pipefail
    cd src/dnadesign/cruncher/workspaces/scar_nick_teto
    cruncher() { uv run cruncher "$@"; }

    # Standard machine-runbook sequence (matches configs/runbook.yaml).
    cruncher scar-nick validate --spec configs/scar_nick/teto_upstream_processing.bbsI_hf.scar_nick.yaml
    cruncher scar-nick design --spec configs/scar_nick/teto_upstream_processing.bbsI_hf.scar_nick.yaml --force-overwrite
    cruncher scar-nick show --run outputs/scar_nick/teto_upstream_processing_bbsI_hf
    cruncher scar-nick validate --spec configs/scar_nick/teto_upstream_processing.paqci_core_panel.scar_nick.yaml
    cruncher scar-nick design --spec configs/scar_nick/teto_upstream_processing.paqci_core_panel.scar_nick.yaml --force-overwrite
    cruncher scar-nick show --run outputs/scar_nick/teto_upstream_processing_paqci_core_panel

### Optional follow-up commands

Run these from the repo root after the corresponding design bundle exists.
Plain `uv run ...` from this nested workspace can hit editable-build path
issues; keep uv project execution rooted at the repository.

```bash
# Render the BbsI-HF terminal-nick visual QA PNG from the generated BaseRender job.
uv run baserender job run src/dnadesign/cruncher/workspaces/scar_nick_teto/outputs/scar_nick/teto_upstream_processing_bbsI_hf/baserender_jobs/scar_nick_terminal_nick.job.yaml
# Render the PaqCI terminal-nick visual QA PNG from the generated BaseRender job.
uv run baserender job run src/dnadesign/cruncher/workspaces/scar_nick_teto/outputs/scar_nick/teto_upstream_processing_paqci_core_panel/baserender_jobs/scar_nick_terminal_nick.job.yaml
```

### Current Strict-Catalog Posture

BbsI-HF plus PaqCI currently cover 13 of the 14 active profile buckets listed
in the study note. `WMWM` remains uncovered under the exact-terminal-nick,
downstream-degenerate, `S0=M` public-catalog policy.

Generated outputs are review artifacts. Do not hand-edit them; update the
specs or code and regenerate.
