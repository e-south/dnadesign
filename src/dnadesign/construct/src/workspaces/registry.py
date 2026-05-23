"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/workspaces/registry.py

Workspace-root helpers and scaffold generation for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import ValidationError as PydanticValidationError

from ..contracts.config import JobConfig, load_job_config
from ..contracts.errors import ConfigError
from .models import (
    WorkspaceDoctorIssue,
    WorkspaceDoctorReport,
    WorkspaceProjectConfig,
    WorkspaceProjectResolution,
    WorkspaceRegistry,
)
from .templates import (
    _CONFIG_TEMPLATE,
    _IMPORT_MANIFEST_TEMPLATE,
    _INPUTS_README,
    _WORKSPACE_PROFILE_DIR,
    _WORKSPACE_REGISTRY_NAME,
    default_workspace_registry_payload,
)


def _construct_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_root_or_none() -> Path | None:
    current = _construct_root().resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def project_root() -> Path:
    resolved = project_root_or_none()
    if resolved is not None:
        return resolved
    raise ConfigError("project root with pyproject.toml not found for construct workspace scaffolding.")


def default_workspace_root() -> Path:
    env_root = os.environ.get("CONSTRUCT_WORKSPACE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path.cwd().resolve()


def workspace_root_with_source(explicit_root: str | None = None) -> tuple[Path, str]:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve(), "arg"
    env_root = os.environ.get("CONSTRUCT_WORKSPACE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve(), "env"
    return default_workspace_root(), "cwd"


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


def _workspace_inventory_entry(*, workspace_dir: Path) -> dict[str, object]:
    outputs_dir = workspace_dir / "outputs"
    output_files = 0
    latest_output_timestamp: float | None = None
    if outputs_dir.exists():
        for candidate in outputs_dir.rglob("*"):
            if not candidate.is_file():
                continue
            output_files += 1
            try:
                stat_result = candidate.stat()
            except OSError:
                continue
            if latest_output_timestamp is None or stat_result.st_mtime > latest_output_timestamp:
                latest_output_timestamp = stat_result.st_mtime
    latest_output_mtime = (
        datetime.fromtimestamp(latest_output_timestamp).astimezone().isoformat(timespec="seconds")
        if latest_output_timestamp is not None
        else None
    )
    return {
        "workspace_id": workspace_dir.name,
        "workspace_dir": str(workspace_dir.resolve()),
        "workspace_state": "attention" if output_files else "clean",
        "output_files": output_files,
        "latest_output_mtime": latest_output_mtime,
    }


def _workspace_inventory_from_root(root: Path) -> list[dict[str, object]]:
    if not root.exists() or not root.is_dir():
        return []
    inventory: list[dict[str, object]] = []
    for workspace_dir in sorted(root.iterdir()):
        if not workspace_dir.is_dir():
            continue
        if not (workspace_dir / _WORKSPACE_REGISTRY_NAME).is_file():
            continue
        inventory.append(_workspace_inventory_entry(workspace_dir=workspace_dir))
    return inventory


def list_packaged_workspace_inventory() -> list[dict[str, object]]:
    root = (_construct_root() / "workspaces").resolve()
    inventory = _workspace_inventory_from_root(root)
    for entry in inventory:
        entry["workspace_source"] = "packaged"
    return inventory


def list_workspace_inventory(root: str | None = None) -> list[dict[str, object]]:
    workspace_root, workspace_root_source = workspace_root_with_source(root)
    inventory_by_id: dict[str, dict[str, object]] = {}
    for entry in _workspace_inventory_from_root(workspace_root):
        entry["workspace_source"] = "local"
        entry["workspace_root_source"] = workspace_root_source
        inventory_by_id[str(entry["workspace_id"])] = entry
    for entry in list_packaged_workspace_inventory():
        inventory_by_id.setdefault(str(entry["workspace_id"]), entry)
    return [inventory_by_id[key] for key in sorted(inventory_by_id)]


def _simple_workspace_id(value: str | Path) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return None
    if text.endswith(_WORKSPACE_REGISTRY_NAME):
        return None
    if any(sep in text for sep in ("/", "\\")):
        return None
    return text


def _resolve_workspace_base_path(workspace: str | Path) -> Path:
    candidate = Path(workspace).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()

    workspace_id = _simple_workspace_id(workspace)
    if workspace_id is not None:
        workspace_root, _source = workspace_root_with_source()
        local_workspace = (workspace_root / workspace_id).resolve()
        if (local_workspace / _WORKSPACE_REGISTRY_NAME).is_file():
            return local_workspace
        packaged_workspace = (_construct_root() / "workspaces" / workspace_id).resolve()
        if (packaged_workspace / _WORKSPACE_REGISTRY_NAME).is_file():
            return packaged_workspace

    return candidate.resolve()


def workspace_registry_path(workspace: str | Path) -> Path:
    base = _resolve_workspace_base_path(workspace)
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


def resolve_workspace_project_config_artifact_path(*, workspace_dir: Path, config_value: str) -> Path:
    """Resolve one workspace-scoped config artifact path using construct's registry rules."""
    return _resolve_project_config_path(workspace_dir=workspace_dir, config_value=config_value)


def _project_contract_errors(*, project: WorkspaceProjectConfig, config: JobConfig) -> list[str]:
    errors: list[str] = []
    if config.job.id != project.artifacts.config.job_id:
        errors.append(
            "registry config.job_id="
            f"'{project.artifacts.config.job_id}' does not match "
            f"config job.id='{config.job.id}'."
        )
    if config.job.input.source.dataset != project.contract.input_dataset:
        errors.append(
            "registry input_dataset="
            f"'{project.contract.input_dataset}' does not match "
            f"config input.source.dataset='{config.job.input.source.dataset}'."
        )
    if config.job.output.target.dataset != project.contract.output_dataset:
        errors.append(
            "registry output_dataset="
            f"'{project.contract.output_dataset}' does not match "
            f"config output.target.dataset='{config.job.output.target.dataset}'."
        )
    template_contract = project.contract.template
    if template_contract and template_contract.id and config.job.template.id != template_contract.id:
        errors.append(
            "registry template_id="
            f"'{template_contract.id}' does not match "
            f"config template.id='{config.job.template.id}'."
        )
    if template_contract and template_contract.dataset:
        config_template_dataset = (
            str(config.job.template.source.dataset) if config.job.template.source.kind == "usr" else ""
        )
        if config_template_dataset != template_contract.dataset:
            errors.append(
                "registry template_dataset="
                f"'{template_contract.dataset}' does not match "
                f"config template.source.dataset='{config_template_dataset}'."
            )
    if template_contract and template_contract.record_id:
        config_template_record_id = (
            str(config.job.template.source.record_id) if config.job.template.source.kind == "usr" else ""
        )
        if config_template_record_id != template_contract.record_id:
            errors.append(
                "registry template_record_id="
                f"'{template_contract.record_id}' does not match "
                f"config template.source.record_id='{config_template_record_id}'."
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
    config_path = _resolve_project_config_path(
        workspace_dir=workspace_dir,
        config_value=project.artifacts.config.path,
    )
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
        config_path = _resolve_project_config_path(
            workspace_dir=workspace_dir,
            config_value=project.artifacts.config.path,
        )
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


def _write_workspace_registry(*, workspace_dir: Path, workspace_id: str, profile: str) -> Path:
    path = workspace_registry_path(workspace_dir)
    payload = default_workspace_registry_payload(workspace_id=workspace_id, profile=profile)
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
        payload = default_workspace_registry_payload(workspace_id=workspace_id, profile=profile)

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
    project_root_hint = project_root_or_none()
    text = text.replace("__CONSTRUCT_PROJECT_ROOT__", project_root_hint.as_posix() if project_root_hint else "")
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
    if workspace_root.exists() and not workspace_root.is_dir():
        raise ConfigError(f"workspace root must be a directory: {workspace_root}")

    resolved_profile = validate_workspace_profile(profile)
    template_dir, _template_source = workspace_template_with_source(resolved_profile)
    created_workspace = False
    try:
        workspace_root.mkdir(parents=True, exist_ok=True)
        if template_dir is None:
            workspace_dir.mkdir(parents=False, exist_ok=False)
            created_workspace = True
            _copy_blank_workspace(workspace_dir, workspace_id=workspace_id)
        else:
            created_workspace = True
            shutil.copytree(template_dir, workspace_dir)
            _rewrite_workspace_registry_identity(
                workspace_dir=workspace_dir,
                workspace_id=workspace_id,
                profile=resolved_profile,
            )
            _rewrite_packaged_runbook_project_root(workspace_dir=workspace_dir)
        (workspace_dir / "outputs" / "logs" / "ops" / "audit").mkdir(parents=True, exist_ok=True)
    except ConfigError:
        if created_workspace and workspace_dir.exists():
            shutil.rmtree(workspace_dir, ignore_errors=True)
        raise
    except OSError as exc:
        if created_workspace and workspace_dir.exists():
            shutil.rmtree(workspace_dir, ignore_errors=True)
        detail = exc.strerror or str(exc)
        raise ConfigError(f"construct workspace could not be created at {workspace_dir}: {detail}") from exc
    return workspace_dir
