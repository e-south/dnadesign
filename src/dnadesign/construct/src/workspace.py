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
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from .config import JobConfig, load_job_config
from .errors import ConfigError

_WORKSPACE_PROFILE_DIR = {
    "promoter-swap-demo": "demo_promoter_swap_pdual10",
}
_WORKSPACE_REGISTRY_NAME = "construct.workspace.yaml"

_INPUTS_README = """# construct workspace inputs

- `construct` now expects both anchors and templates to live in USR datasets.
- Keep workspace-level project inventory and provenance in `construct.workspace.yaml`.
- Use `uv run construct seed promoter-swap-demo --manifest inputs/seed_manifest.yaml`
  when you want the curated `mg1655_promoters` and `plasmids` datasets seeded into this
  workspace's `outputs/usr_datasets/` root.
- Omit `--root` only when you deliberately want to seed the canonical shared USR root at
  `src/dnadesign/usr/datasets/`.
- Keep human-readable sequence names in `usr_label__primary` / `usr_label__aliases`; keep
  construct-specific seed provenance in `construct_seed__*`.
- Keep canonical template records in USR; do not fall back to ad hoc FASTA files for ordinary construct runs.
- Prefer flat semantic output dataset ids such as `pdual10_slot_a_window_1kb_demo`, not tool-owned dataset namespaces.
"""

_IMPORT_MANIFEST_TEMPLATE = """manifest_id: example_construct_inputs
datasets:
  - id: example_anchors
    notes: Example anchor inputs for a custom construct study.
    records:
      - label: example_anchor
        role: anchor
        topology: linear
        aliases: [example_anchor_alias]
        source_ref: replace-with-canonical-source
        sequence: ACGTACGT
  - id: example_templates
    notes: Example template records for a custom construct study.
    records:
      - label: example_template
        role: template
        topology: circular
        aliases: [example_template_alias]
        source_ref: replace-with-canonical-source
        sequence: AAAATTTTCCCCGGGG
"""

_CONFIG_TEMPLATE = """job:
  id: {workspace_id}
  input:
    source: usr
    dataset: REPLACE_WITH_ANCHOR_DATASET
    # root: outputs/usr_datasets
    field: sequence
  template:
    id: REPLACE_WITH_TEMPLATE_LABEL
    kind: usr
    dataset: REPLACE_WITH_TEMPLATE_DATASET
    # root: outputs/usr_datasets
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
        start: REPLACE_WITH_TEMPLATE_START
        end: REPLACE_WITH_TEMPLATE_END
        orientation: forward
        expected_template_sequence: REPLACE_WITH_INCUMBENT_SEQUENCE
  realize:
    mode: window
    focal_part: anchor
    focal_point: center
    anchor_offset_bp: 0
    window_bp: 1000
  output:
    dataset: REPLACE_WITH_OUTPUT_DATASET
    # root: outputs/usr_datasets
"""


class _StrictWorkspaceModel(BaseModel):
    model_config = {"extra": "forbid"}


class WorkspaceRootsConfig(_StrictWorkspaceModel):
    shared_usr_root: str = "src/dnadesign/usr/datasets"
    workspace_usr_root: str = "outputs/usr_datasets"

    @field_validator("shared_usr_root", "workspace_usr_root")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("workspace registry roots cannot be empty.")
        return text


class WorkspaceProjectConfig(_StrictWorkspaceModel):
    id: str
    config: str
    flow: str
    input_dataset: str
    template_id: str | None = None
    template_dataset: str | None = None
    template_record_id: str | None = None
    output_dataset: str
    notes: str | None = None

    @field_validator("id", "config", "flow", "input_dataset", "output_dataset")
    @classmethod
    def _required_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("workspace registry project fields cannot be empty.")
        return text


class _WorkspaceRegistryBody(_StrictWorkspaceModel):
    id: str
    profile: str
    description: str = ""
    roots: WorkspaceRootsConfig = Field(default_factory=WorkspaceRootsConfig)
    projects: list[WorkspaceProjectConfig] = Field(default_factory=list)

    @field_validator("id", "profile")
    @classmethod
    def _workspace_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("workspace id/profile cannot be empty.")
        return text

    @model_validator(mode="after")
    def _validate_projects(self) -> "_WorkspaceRegistryBody":
        seen_ids: set[str] = set()
        seen_configs: set[str] = set()
        for project in self.projects:
            if project.id in seen_ids:
                raise ValueError(f"Duplicate workspace project id '{project.id}'.")
            if project.config in seen_configs:
                raise ValueError(f"Duplicate workspace project config '{project.config}'.")
            seen_ids.add(project.id)
            seen_configs.add(project.config)
        return self


class WorkspaceRegistry(_StrictWorkspaceModel):
    workspace: _WorkspaceRegistryBody


@dataclass(frozen=True)
class WorkspaceProjectResolution:
    workspace_dir: Path
    registry_path: Path
    registry: WorkspaceRegistry
    project: WorkspaceProjectConfig
    config_path: Path
    config: JobConfig


@dataclass(frozen=True)
class WorkspaceDoctorIssue:
    severity: str
    project_id: str
    message: str


@dataclass(frozen=True)
class WorkspaceDoctorReport:
    workspace_dir: Path
    registry_path: Path
    workspace_id: str
    profile: str
    projects_checked: int
    issues: list[WorkspaceDoctorIssue]


def _construct_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    current = _construct_root().resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise ConfigError("project root with pyproject.toml not found for construct workspace scaffolding.")


def default_workspace_root() -> Path:
    env_root = os.environ.get("CONSTRUCT_WORKSPACE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (_construct_root() / "workspaces").resolve()


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


def validate_workspace_profile(profile: str) -> str:
    text = str(profile or "").strip()
    if text in {"", "blank"}:
        return "blank"
    if text not in _WORKSPACE_PROFILE_DIR:
        choices = ", ".join(["blank", *sorted(_WORKSPACE_PROFILE_DIR)])
        raise ConfigError(f"workspace profile must be one of: {choices}")
    return text


def workspace_template_with_source(profile: str) -> tuple[Path | None, str]:
    resolved = validate_workspace_profile(profile)
    if resolved == "blank":
        return None, "builtin-blank"
    resolved_path = (_construct_root() / "workspaces" / _WORKSPACE_PROFILE_DIR[resolved]).resolve()
    if not resolved_path.exists() or not resolved_path.is_dir():
        raise ConfigError(f"packaged workspace template not found: {resolved_path}")
    return resolved_path, "package-profile"


def workspace_registry_path(workspace: str | Path) -> Path:
    base = Path(workspace).expanduser().resolve()
    return base if base.name == _WORKSPACE_REGISTRY_NAME else base / _WORKSPACE_REGISTRY_NAME


def load_workspace_registry(workspace: str | Path) -> tuple[WorkspaceRegistry, Path]:
    path = workspace_registry_path(workspace)
    if not path.exists():
        raise ConfigError(f"construct workspace registry not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ConfigError(f"construct workspace registry could not be read: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in construct workspace registry: {path}") from exc
    try:
        return WorkspaceRegistry.model_validate(payload), path
    except PydanticValidationError as exc:
        raise ConfigError(f"Invalid construct workspace registry {path}: {exc}") from exc


def _resolve_project_config_path(*, workspace_dir: Path, config_value: str) -> Path:
    candidate = Path(config_value).expanduser()
    if candidate.is_absolute():
        raise ConfigError("workspace project config paths must be workspace-relative, not absolute.")
    resolved = (workspace_dir / candidate).resolve()
    try:
        resolved.relative_to(workspace_dir.resolve())
    except ValueError as exc:
        raise ConfigError(f"workspace project config must stay inside the workspace root: {config_value}") from exc
    return resolved


def _project_contract_errors(*, project: WorkspaceProjectConfig, config: JobConfig) -> list[str]:
    errors: list[str] = []
    if config.job.input.dataset != project.input_dataset:
        errors.append(
            "registry input_dataset="
            f"'{project.input_dataset}' does not match "
            f"config input.dataset='{config.job.input.dataset}'."
        )
    if config.job.output.dataset != project.output_dataset:
        errors.append(
            "registry output_dataset="
            f"'{project.output_dataset}' does not match "
            f"config output.dataset='{config.job.output.dataset}'."
        )
    if project.template_id and config.job.template.id != project.template_id:
        errors.append(
            "registry template_id="
            f"'{project.template_id}' does not match "
            f"config template.id='{config.job.template.id}'."
        )
    if project.template_dataset:
        config_template_dataset = str(config.job.template.dataset or "")
        if config_template_dataset != project.template_dataset:
            errors.append(
                "registry template_dataset="
                f"'{project.template_dataset}' does not match config template.dataset='{config_template_dataset}'."
            )
    if project.template_record_id:
        config_template_record_id = str(config.job.template.record_id or "")
        if config_template_record_id != project.template_record_id:
            errors.append(
                "registry template_record_id="
                f"'{project.template_record_id}' does not match "
                f"config template.record_id='{config_template_record_id}'."
            )
    return errors


def resolve_workspace_project(workspace: str | Path, *, project_id: str) -> WorkspaceProjectResolution:
    registry, registry_path = load_workspace_registry(workspace)
    workspace_dir = registry_path.parent
    requested = str(project_id or "").strip()
    if not requested:
        raise ConfigError("workspace project id cannot be empty.")
    project = next((item for item in registry.workspace.projects if item.id == requested), None)
    if project is None:
        choices = ", ".join(project_item.id for project_item in registry.workspace.projects) or "<none>"
        raise ConfigError(f"workspace project '{requested}' not found. Known project ids: {choices}")
    config_path = _resolve_project_config_path(workspace_dir=workspace_dir, config_value=project.config)
    if not config_path.exists():
        raise ConfigError(f"workspace project '{requested}' config not found: {config_path}")
    config, _ = load_job_config(config_path)
    drift = _project_contract_errors(project=project, config=config)
    if drift:
        rendered = "; ".join(drift)
        raise ConfigError(f"workspace project '{requested}' is out of sync with construct.workspace.yaml: {rendered}")
    return WorkspaceProjectResolution(
        workspace_dir=workspace_dir,
        registry_path=registry_path,
        registry=registry,
        project=project,
        config_path=config_path,
        config=config,
    )


def doctor_workspace_registry(workspace: str | Path) -> WorkspaceDoctorReport:
    registry, registry_path = load_workspace_registry(workspace)
    workspace_dir = registry_path.parent
    issues: list[WorkspaceDoctorIssue] = []
    for project in registry.workspace.projects:
        config_path = _resolve_project_config_path(workspace_dir=workspace_dir, config_value=project.config)
        if not config_path.exists():
            issues.append(
                WorkspaceDoctorIssue(
                    severity="error",
                    project_id=project.id,
                    message=f"config path does not exist: {config_path}",
                )
            )
            continue
        try:
            config, _ = load_job_config(config_path)
        except ConfigError as exc:
            issues.append(
                WorkspaceDoctorIssue(
                    severity="error",
                    project_id=project.id,
                    message=str(exc),
                )
            )
            continue
        for message in _project_contract_errors(project=project, config=config):
            issues.append(
                WorkspaceDoctorIssue(
                    severity="error",
                    project_id=project.id,
                    message=message,
                )
            )
    return WorkspaceDoctorReport(
        workspace_dir=workspace_dir,
        registry_path=registry_path,
        workspace_id=registry.workspace.id,
        profile=registry.workspace.profile,
        projects_checked=len(registry.workspace.projects),
        issues=issues,
    )


def _default_workspace_registry_payload(*, workspace_id: str, profile: str) -> dict:
    project_config = "config.yaml" if profile == "blank" else "config.slot_a.window.yaml"
    project_flow = "replace-anchor-in-template"
    project_output = "REPLACE_WITH_OUTPUT_DATASET" if profile == "blank" else "pdual10_slot_a_window_1kb_demo"
    project_template_id = "REPLACE_WITH_TEMPLATE_LABEL" if profile == "blank" else "pDual-10"
    project_template_dataset = "REPLACE_WITH_TEMPLATE_DATASET" if profile == "blank" else "plasmids"
    return {
        "workspace": {
            "id": workspace_id,
            "profile": profile,
            "description": (
                "Construct workspace registry for explicit project inventory, config provenance, and USR root hints."
            ),
            "roots": {
                "shared_usr_root": "src/dnadesign/usr/datasets",
                "workspace_usr_root": "outputs/usr_datasets",
            },
            "projects": [
                {
                    "id": workspace_id if profile == "blank" else "slot_a_window",
                    "config": project_config,
                    "flow": project_flow,
                    "input_dataset": "REPLACE_WITH_ANCHOR_DATASET" if profile == "blank" else "mg1655_promoters",
                    "template_id": project_template_id,
                    "template_dataset": project_template_dataset,
                    "template_record_id": (
                        "REPLACE_WITH_TEMPLATE_RECORD_ID"
                        if profile == "blank"
                        else "c4f17db3c2dbc17c5cb32c5eec785ea4f091e51d"  # pragma: allowlist secret
                    ),
                    "output_dataset": project_output,
                    "notes": (
                        "Replace placeholders and add more project entries as flows expand."
                        if profile == "blank"
                        else "Windowed promoter swap against slot_a in pDual-10."
                    ),
                }
            ],
        }
    }


def _write_workspace_registry(*, workspace_dir: Path, workspace_id: str, profile: str) -> Path:
    path = workspace_registry_path(workspace_dir)
    payload = _default_workspace_registry_payload(workspace_id=workspace_id, profile=profile)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _rewrite_workspace_registry_identity(*, workspace_dir: Path, workspace_id: str, profile: str) -> Path:
    path = workspace_registry_path(workspace_dir)
    if path.exists():
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in packaged construct workspace registry: {path}") from exc
    else:
        payload = _default_workspace_registry_payload(workspace_id=workspace_id, profile=profile)

    workspace_payload = payload.setdefault("workspace", {})
    workspace_payload["id"] = workspace_id
    workspace_payload["profile"] = profile
    if "roots" not in workspace_payload:
        workspace_payload["roots"] = {
            "shared_usr_root": "src/dnadesign/usr/datasets",
            "workspace_usr_root": "outputs/usr_datasets",
        }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _rewrite_packaged_runbook_project_root(*, workspace_dir: Path) -> None:
    runbook_path = workspace_dir / "runbook.sh"
    if not runbook_path.exists():
        return
    text = runbook_path.read_text(encoding="utf-8")
    text = text.replace("__CONSTRUCT_PROJECT_ROOT__", project_root().as_posix())
    runbook_path.write_text(text, encoding="utf-8")


def _copy_blank_workspace(workspace_dir: Path, *, workspace_id: str) -> None:
    (workspace_dir / "inputs").mkdir(parents=True, exist_ok=False)
    (workspace_dir / "config.yaml").write_text(
        _CONFIG_TEMPLATE.format(workspace_id=workspace_id),
        encoding="utf-8",
    )
    (workspace_dir / "inputs" / "README.md").write_text(_INPUTS_README, encoding="utf-8")
    (workspace_dir / "inputs" / "import_manifest.template.yaml").write_text(
        _IMPORT_MANIFEST_TEMPLATE,
        encoding="utf-8",
    )
    _write_workspace_registry(workspace_dir=workspace_dir, workspace_id=workspace_id, profile="blank")


def init_workspace(*, workspace_id: str, root: str | None = None, profile: str = "blank") -> Path:
    workspace_id = validate_workspace_id(workspace_id)
    workspace_root, _ = workspace_root_with_source(root)
    workspace_dir = workspace_root / workspace_id
    if workspace_dir.exists():
        raise ConfigError(f"workspace already exists: {workspace_dir}")

    workspace_root.mkdir(parents=True, exist_ok=True)
    resolved_profile = validate_workspace_profile(profile)
    template_dir, _template_source = workspace_template_with_source(resolved_profile)
    if template_dir is None:
        workspace_dir.mkdir(parents=False, exist_ok=False)
        _copy_blank_workspace(workspace_dir, workspace_id=workspace_id)
    else:
        shutil.copytree(template_dir, workspace_dir)
        _rewrite_workspace_registry_identity(
            workspace_dir=workspace_dir,
            workspace_id=workspace_id,
            profile=resolved_profile,
        )
        _rewrite_packaged_runbook_project_root(workspace_dir=workspace_dir)

    (workspace_dir / "outputs" / "logs" / "ops" / "audit").mkdir(parents=True, exist_ok=True)
    return workspace_dir
