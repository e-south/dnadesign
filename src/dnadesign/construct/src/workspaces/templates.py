"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/workspaces/templates.py

Workspace scaffold templates and default registry payloads for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

_WORKSPACE_PROFILE_DIR = {
    "anchor-template-demo": "demo_anchor_template_local",
    "anchor-template-shared-dataset-demo": "demo_anchor_template_shared_dataset",
}
_WORKSPACE_REGISTRY_NAME = "construct.workspace.yaml"

_INPUTS_README = """# construct workspace inputs

- `construct` now expects both anchors and templates to live in USR datasets.
- Keep workspace-level project inventory and provenance in `construct.workspace.yaml`.
- Use `uv run construct seed import-manifest --manifest inputs/import_manifest.template.yaml`
  when you want to materialize your own curated inputs and templates into this
  workspace's `outputs/usr_datasets/` root.
- When you want the packaged `anchor_parts_demo` and `template_parts_demo`
  tracer-bullet datasets, scaffold a packaged anchor/template workspace with
  `construct workspace init --profile anchor-template-demo` or
  `construct workspace init --profile anchor-template-shared-dataset-demo`.
- Omit `--root` only when you are running inside a dnadesign checkout and deliberately want to seed
  the canonical shared USR root at `src/dnadesign/usr/datasets/`, or when `DNADESIGN_USR_ROOT`
  points at a writable datasets root.
- Keep human-readable sequence names in `usr_label__primary` / `usr_label__aliases`; keep
  construct-specific seed provenance in `construct_seed__*`.
- Keep canonical template records in USR; do not fall back to ad hoc FASTA files
  for ordinary construct runs.
- Prefer flat semantic output dataset ids such as
  `anchor_template_slot_a_window_1kb_demo`, not tool-owned dataset namespaces.
"""

_IMPORT_MANIFEST_TEMPLATE = """manifest_id: example_construct_inputs
datasets:
  - id: example_anchors
    notes: Example anchor inputs for a custom construct study.
    records:
      - label: example_anchor
        intended_role: anchor
        topology: linear
        aliases: [example_anchor_alias]
        source_ref: replace-with-canonical-source
        sequence: ACGTACGT
  - id: example_templates
    notes: Example template records for a custom construct study.
    records:
      - label: example_template
        intended_role: template
        topology: circular
        aliases: [example_template_alias]
        source_ref: replace-with-canonical-source
        sequence: AAAATTTTCCCCGGGG
"""

_CONFIG_TEMPLATE = """job:
  id: {workspace_id}
  input:
    source:
      kind: usr
      dataset: REPLACE_WITH_ANCHOR_DATASET
      root: outputs/usr_datasets
    field: sequence
  template:
    id: REPLACE_WITH_TEMPLATE_LABEL
    source:
      kind: usr
      dataset: REPLACE_WITH_TEMPLATE_DATASET
      root: outputs/usr_datasets
      record_id: REPLACE_WITH_TEMPLATE_RECORD_ID
      field: sequence
    circular: true
  parts:
    - name: anchor
      role: anchor
      sequence:
        source: input_field
        field: sequence
      placement:
        kind: replace
        orientation: forward
        locator:
          kind: coordinates
          start: REPLACE_WITH_TEMPLATE_START
          end: REPLACE_WITH_TEMPLATE_END
        guards:
          replaced_sequence: REPLACE_WITH_INCUMBENT_SEQUENCE
  realize:
    mode: window
    focal_part: anchor
    window:
      semantics: fixed_total
      reference: center
      direction: symmetric
      size_bp: 1000
      offset_bp: 0
  output:
    target:
      kind: usr
      dataset: REPLACE_WITH_OUTPUT_DATASET
      root: outputs/usr_datasets
"""


def default_workspace_registry_payload(*, workspace_id: str, profile: str) -> dict[str, object]:
    project_config = "config.yaml" if profile == "blank" else "config.slot_a.window.yaml"
    project_job_id = workspace_id if profile == "blank" else "anchor_template_slot_a_window_1kb"
    project_output = "REPLACE_WITH_OUTPUT_DATASET" if profile == "blank" else "anchor_template_slot_a_window_1kb_demo"
    project_template_id = "REPLACE_WITH_TEMPLATE_LABEL" if profile == "blank" else "template_backbone_dual_slot"
    project_template_dataset = "REPLACE_WITH_TEMPLATE_DATASET" if profile == "blank" else "template_parts_demo"
    project_template_record_id = (
        "REPLACE_WITH_TEMPLATE_RECORD_ID"
        if profile == "blank"
        else "c4f17db3c2dbc17c5cb32c5eec785ea4f091e51d"  # pragma: allowlist secret
    )
    return {
        "workspace": {
            "id": workspace_id,
            "profile": profile,
            "description": (
                "Construct workspace registry for explicit project inventory, "
                "tracked config artifacts, and USR root hints."
            ),
            "roots": {
                "shared_usr_root": "src/dnadesign/usr/datasets",
                "workspace_usr_root": "outputs/usr_datasets",
            },
            "projects": [
                {
                    "id": workspace_id if profile == "blank" else "slot_a_window",
                    "artifacts": {
                        "config": {
                            "path": project_config,
                            "job_id": project_job_id,
                        }
                    },
                    "contract": {
                        "input_dataset": "REPLACE_WITH_ANCHOR_DATASET" if profile == "blank" else "anchor_parts_demo",
                        "template": {
                            "id": project_template_id,
                            "dataset": project_template_dataset,
                            "record_id": project_template_record_id,
                        },
                        "output_dataset": project_output,
                    },
                    "notes": (
                        "Replace placeholders and add more project entries as the study surface expands."
                        if profile == "blank"
                        else "Windowed anchor placement against slot_a in the packaged dual-slot template."
                    ),
                }
            ],
        }
    }


__all__ = [
    "_CONFIG_TEMPLATE",
    "_IMPORT_MANIFEST_TEMPLATE",
    "_INPUTS_README",
    "_WORKSPACE_PROFILE_DIR",
    "_WORKSPACE_REGISTRY_NAME",
    "default_workspace_registry_payload",
]
