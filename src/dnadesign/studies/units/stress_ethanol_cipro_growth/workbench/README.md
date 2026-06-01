## Study Workbench

This directory stores durable, checked-in study artifacts that are neither
Python execution surfaces nor generated runtime outputs.

- `study.yaml`: study metadata used by LatentDNA deliverable-doc binding.
- `deliverables/`: LatentDNA-facing review prose organized by review role.
- `notes/`: dated interpretation, rationale, audits, and handoffs.
- `reference_sets/`: static study curation records.

Do not put executable study code here. Put OPAL decision code under
`../decision/opal/` and OPS status/preflight code under `../operations/status/`.
