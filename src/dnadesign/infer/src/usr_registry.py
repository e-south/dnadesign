"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/usr_registry.py

USR namespace registration spec derivation for infer write-back jobs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import RootConfig
from .contracts import infer_usr_column_name
from .errors import ConfigError


@dataclass(frozen=True)
class UsrRegistrySpec:
    namespace: str
    root: str
    columns: tuple[str, ...]

    @property
    def columns_spec(self) -> str:
        return ",".join(self.columns)

    @property
    def register_command(self) -> str:
        return f"uv run usr --root {self.root} namespace register {self.namespace} --columns '{self.columns_spec}'"


def _registry_type_for_output_format(output_format: str) -> str:
    if output_format == "float":
        return "float64"
    if output_format == "list":
        return "list<float64>"
    raise ConfigError("USR registry spec only supports infer output formats: float, list")


def derive_usr_registry_spec(*, root: RootConfig, job_id: str | None = None) -> UsrRegistrySpec:
    selected_jobs = []
    for job in root.jobs:
        if job_id is not None and job.id != job_id:
            continue
        if job.operation != "extract":
            continue
        if job.ingest.source != "usr":
            continue
        if not bool(job.io.write_back):
            continue
        selected_jobs.append(job)

    if job_id is not None and not any(job.id == job_id for job in root.jobs):
        raise ConfigError(f"Unknown job id '{job_id}'.")
    if not selected_jobs:
        raise ConfigError("No USR write-back extract jobs found in config.")

    roots = {str(job.ingest.root or "").strip() for job in selected_jobs}
    if "" in roots:
        raise ConfigError("All selected USR write-back jobs must set ingest.root.")
    if len(roots) != 1:
        raise ConfigError("All selected USR write-back jobs must use the same ingest.root.")
    usr_root = next(iter(roots))

    columns: list[str] = []
    seen: set[str] = set()
    for job in selected_jobs:
        for output in job.outputs or []:
            column_name = infer_usr_column_name(model_id=root.model.id, job_id=job.id, out_id=output.id)
            if column_name in seen:
                raise ConfigError(f"Duplicate infer USR column derived from config: {column_name}")
            seen.add(column_name)
            columns.append(f"{column_name}:{_registry_type_for_output_format(output.format)}")

    return UsrRegistrySpec(namespace="infer", root=usr_root, columns=tuple(columns))
