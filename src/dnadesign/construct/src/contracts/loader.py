"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/contracts/loader.py

YAML loading for construct job configs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError as PydanticValidationError

from .errors import ConfigError
from .job import JobConfig


def load_job_config(path: str | Path) -> tuple[JobConfig, Path]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"Config not found: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in config: {config_path}") from exc
    try:
        return JobConfig.model_validate(data), config_path
    except PydanticValidationError as exc:
        raise ConfigError(f"Invalid config {config_path}: {exc}") from exc
