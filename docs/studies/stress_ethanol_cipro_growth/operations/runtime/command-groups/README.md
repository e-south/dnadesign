---
doc_id: study-stress-ethanol-cipro-growth-command-groups
surface: study-runtime-command-group-map
study_id: stress_ethanol_cipro_growth
owner: dnadesign-maintainers
last_verified: 2026-05-18
entrypoint: self
canonical_payload: pipeline.yaml
---

## Stress Runtime Command Groups

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Use this page when an agent needs runtime command-group orientation without
opening the full `pipeline.yaml` first. The full YAML is the canonical payload
consumed by status, preflight, OPAL, LatentDNA, and docs-contract checks.

### Lane Index

| Need | Start Here | Canonical Payload |
| --- | --- | --- |
| DenseGen source generation and plot inventory | [DenseGen lane](lanes/densegen.yaml) | `pipeline.yaml:study_pipeline.densegen` |
| Infer runbooks, model families, sidecar completion, and SCC posture | [Infer lane](lanes/infer.yaml) | `pipeline.yaml:study_pipeline.infer` |
| LatentDNA representation comparison and browser review surfaces | [LatentDNA lane](lanes/latentdna.yaml) | `pipeline.yaml:study_pipeline.latentdna` |
| Cluster exploratory follow-up | [Cluster lane](lanes/cluster.yaml) | `pipeline.yaml:study_pipeline.cluster` |
| OPAL candidate table and campaign viewer handoff | [OPAL lane](lanes/opal.yaml) | `pipeline.yaml:study_pipeline.opal` |

### Navigation Rule

Open one lane file first, then jump to the matching key in `pipeline.yaml` only
when the task needs the full machine-readable payload. Keep durable current
state in `../../../record/status.md`, owner routing in `../../../routes/README.md`,
and OPS contract fragments in `../../contract/`.
