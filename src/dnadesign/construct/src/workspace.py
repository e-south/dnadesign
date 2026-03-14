"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/workspace.py

Workspace-root helpers and scaffold generation for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path

from .errors import ConfigError

_TEMPLATE_FASTA = """>demo_template
AAAATTTTCCCCGGGGAAAATTTTCCCCGGGG
"""

_ANCHOR_MANIFEST_TEMPLATE = """dataset:
  name: anchors_controls_promoters_demo
  metadata_namespace: catalog
  notes:
    - Fill sequence fields from canonical sources before importing into USR.
    - Do not treat chat-pasted plasmid sequences as canonical without a file/checksum.

anchors:
  - label: spyP_MG1655
    class: promoter_control
    topology: linear
    organism: Escherichia coli K-12 substr. MG1655
    source_kind: curated_control
    expected_length_bp: 220
    sequence: REPLACE_WITH_CANONICAL_SEQUENCE
  - label: sulAp
    class: promoter_control
    topology: linear
    organism: Escherichia coli
    source_kind: curated_control
    expected_length_bp: 165
    sequence: REPLACE_WITH_CANONICAL_SEQUENCE
  - label: soxS
    class: promoter_control
    topology: linear
    organism: Escherichia coli
    source_kind: curated_control
    expected_length_bp: 200
    sequence: REPLACE_WITH_CANONICAL_SEQUENCE
  - label: J23105
    class: promoter_control
    topology: linear
    organism: synthetic
    source_kind: curated_control
    expected_length_bp: 35
    sequence: REPLACE_WITH_CANONICAL_SEQUENCE
  - label: pDual-10
    class: plasmid_template
    topology: circular
    organism: synthetic
    source_kind: workspace_template
    sequence_file: REPLACE_WITH_CANONICAL_FASTA_PATH
    sha256: REPLACE_WITH_CANONICAL_SHA256
"""

_INPUTS_README = """# construct workspace inputs

- `template.fa` is a throwaway demo sequence created by `construct workspace init`.
- Replace it with a canonical template FASTA before any real run.
- `anchor_manifest.template.yaml` is a worksheet for control anchors and template references.
- Do not commit long DNA copied from chat paste as canonical sequence content
  unless you have an authoritative file or checksum-backed source.
"""

_CONFIG_TEMPLATE = """job:
  id: {workspace_id}
  input:
    source: usr
    dataset: anchors_controls_promoters_demo
    root: outputs/usr_datasets
    field: sequence
  template:
    id: demo_template
    path: inputs/template.fa
    circular: true
    source: inputs/template.fa
  parts:
    - name: anchor
      role: anchor
      sequence:
        source: input_field
        field: sequence
      placement:
        kind: replace
        start: 8
        end: 12
        orientation: forward
        expected_template_sequence: CCCC
  realize:
    mode: window
    focal_part: anchor
    focal_point: center
    anchor_offset_bp: 0
    window_bp: 16
  output:
    dataset: {workspace_id}_constructed
    root: outputs/usr_datasets
"""


def default_workspace_root() -> Path:
    env_root = os.environ.get("CONSTRUCT_WORKSPACE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "workspaces").resolve()


def workspace_root_with_source(explicit_root: str | None = None) -> tuple[Path, str]:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve(), "arg"
    env_root = os.environ.get("CONSTRUCT_WORKSPACE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve(), "env"
    return default_workspace_root(), "package"


def validate_workspace_id(workspace_id: str) -> str:
    text = str(workspace_id or "").strip()
    if not text:
        raise ConfigError("workspace id cannot be empty.")
    if "/" in text or "\\" in text:
        raise ConfigError("workspace id must be a simple directory name, not a path.")
    if text in {".", ".."}:
        raise ConfigError("workspace id must be a simple directory name, not '.' or '..'.")
    return text


def init_workspace(*, workspace_id: str, root: str | None = None) -> Path:
    workspace_id = validate_workspace_id(workspace_id)
    workspace_root, _ = workspace_root_with_source(root)
    workspace_dir = workspace_root / workspace_id
    if workspace_dir.exists():
        raise ConfigError(f"workspace already exists: {workspace_dir}")

    (workspace_dir / "inputs").mkdir(parents=True, exist_ok=False)
    (workspace_dir / "outputs" / "logs" / "ops" / "audit").mkdir(parents=True, exist_ok=False)
    (workspace_dir / "outputs" / "usr_datasets").mkdir(parents=True, exist_ok=False)

    (workspace_dir / "config.yaml").write_text(
        _CONFIG_TEMPLATE.format(workspace_id=workspace_id),
        encoding="utf-8",
    )
    (workspace_dir / "inputs" / "template.fa").write_text(_TEMPLATE_FASTA, encoding="utf-8")
    (workspace_dir / "inputs" / "anchor_manifest.template.yaml").write_text(
        _ANCHOR_MANIFEST_TEMPLATE,
        encoding="utf-8",
    )
    (workspace_dir / "inputs" / "README.md").write_text(_INPUTS_README, encoding="utf-8")
    return workspace_dir
