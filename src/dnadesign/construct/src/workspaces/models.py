"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/workspaces/models.py

Typed workspace registry contracts for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from ..contracts.config import JobConfig


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


class WorkspaceProjectTemplateContractConfig(_StrictWorkspaceModel):
    id: str | None = None
    dataset: str | None = None
    record_id: str | None = None

    @field_validator("id", "dataset", "record_id")
    @classmethod
    def _optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value or "").strip()
        if not text:
            raise ValueError("workspace registry contract template fields cannot be empty when provided.")
        return text

    @model_validator(mode="after")
    def _complete_usr_identity(self) -> "WorkspaceProjectTemplateContractConfig":
        if any(value is not None for value in (self.id, self.dataset, self.record_id)):
            if not all(value is not None for value in (self.id, self.dataset, self.record_id)):
                raise ValueError("workspace registry contract template requires id, dataset, and record_id together.")
        return self


class WorkspaceProjectContractConfig(_StrictWorkspaceModel):
    input_dataset: str
    output_dataset: str
    template: WorkspaceProjectTemplateContractConfig | None = None

    @field_validator("input_dataset", "output_dataset")
    @classmethod
    def _required_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("workspace registry contract dataset fields cannot be empty.")
        return text


class WorkspaceProjectConfigArtifactConfig(_StrictWorkspaceModel):
    path: str
    job_id: str

    @field_validator("path", "job_id")
    @classmethod
    def _required_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("workspace registry config artifact fields cannot be empty.")
        return text


class WorkspaceProjectArtifactsConfig(_StrictWorkspaceModel):
    config: WorkspaceProjectConfigArtifactConfig


class WorkspaceProjectConfig(_StrictWorkspaceModel):
    id: str
    artifacts: WorkspaceProjectArtifactsConfig
    contract: WorkspaceProjectContractConfig
    notes: str | None = None

    @field_validator("id")
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
            config_path = project.artifacts.config.path
            if config_path in seen_configs:
                raise ValueError(f"Duplicate workspace project config '{config_path}'.")
            seen_ids.add(project.id)
            seen_configs.add(config_path)
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


__all__ = [
    "WorkspaceDoctorIssue",
    "WorkspaceDoctorReport",
    "WorkspaceProjectArtifactsConfig",
    "WorkspaceProjectConfig",
    "WorkspaceProjectConfigArtifactConfig",
    "WorkspaceProjectContractConfig",
    "WorkspaceProjectResolution",
    "WorkspaceProjectTemplateContractConfig",
    "WorkspaceRegistry",
    "WorkspaceRootsConfig",
]
