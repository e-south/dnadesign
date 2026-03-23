## study_stress_ethanol_cipro_pdual10 Runbook

Use this runbook after `densegen/study_stress_ethanol_cipro` and
`mg1655_promoters` have been merged into the study-owned anchor set.

### 1) Bootstrap the merged anchor dataset without mutating the source datasets

```bash
uv run usr --root src/dnadesign/usr/datasets init \
  promoter/stress_ethanol_cipro_anchor_set \
  --source stress_ethanol_cipro_growth \
  --notes "Merged anchor set for Construct and Infer"

uv run usr --root src/dnadesign/usr/datasets maintenance merge \
  --dest promoter/stress_ethanol_cipro_anchor_set \
  --src mg1655_promoters \
  --union-columns \
  --if-duplicate error \
  --carry-namespace usr_label

uv run usr --root src/dnadesign/usr/datasets maintenance merge \
  --dest promoter/stress_ethanol_cipro_anchor_set \
  --src densegen/study_stress_ethanol_cipro \
  --union-columns \
  --if-duplicate error \
  --carry-namespace usr_label

uv run usr --root src/dnadesign/usr/datasets validate \
  promoter/stress_ethanol_cipro_anchor_set \
  --strict
```

### 2) Validate the Construct workspace against the real study inputs

```bash
uv run construct workspace doctor \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10

uv run construct workspace validate-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project slot_a_window \
  --runtime

uv run construct workspace validate-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project slot_b_window \
  --runtime
```

### 3) Preview the shared downstream writes

```bash
uv run construct workspace run-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project slot_a_window \
  --dry-run

uv run construct workspace run-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project slot_b_window \
  --dry-run
```

### 4) Materialize the pDual-10 context dataset

```bash
uv run construct workspace run-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project slot_a_window

uv run construct workspace run-project \
  --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10 \
  --project slot_b_window

uv run usr --root src/dnadesign/usr/datasets validate \
  promoter/stress_ethanol_cipro_construct_contexts \
  --strict
```

### 5) Continue into Infer and Notify

Hand off to:

- `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/`
- `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_only_7b_batch_with_notify.yaml`
- `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_plus_template_7b_batch_with_notify.yaml`
