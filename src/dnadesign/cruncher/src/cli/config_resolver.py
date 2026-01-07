"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/config_resolver.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

CANDIDATE_CONFIG_FILENAMES: tuple[str, ...] = (
    "cruncher.yaml",
    "cruncher.yml",
    "config.yaml",
    "config.yml",
)


class ConfigResolutionError(ValueError):
    """Raised when a config path cannot be resolved."""


def _normalize_path(path: Path, cwd: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = cwd / expanded
    return expanded.resolve()


def _resolve_explicit_config(config: Path, *, cwd: Path) -> Path:
    path = _normalize_path(config, cwd)
    if not path.exists():
        raise ConfigResolutionError(f"Config file not found: {path}")
    if not path.is_file():
        raise ConfigResolutionError(f"Config path is not a file: {path}")
    return path


def resolve_config_path(config: Path | None, *, cwd: Path | None = None, log: bool = True) -> Path:
    """Resolve a config path from an explicit argument or from CWD defaults."""
    cwd_path = (cwd or Path.cwd()).expanduser().resolve()
    if config is not None:
        return _resolve_explicit_config(config, cwd=cwd_path)
    matches = [cwd_path / name for name in CANDIDATE_CONFIG_FILENAMES if (cwd_path / name).is_file()]
    if len(matches) == 1:
        resolved = matches[0].resolve()
        if log:
            rel = resolved.relative_to(cwd_path)
            logger.info("Using config from CWD: ./%s", rel.as_posix())
        return resolved
    if not matches:
        expected = ", ".join(CANDIDATE_CONFIG_FILENAMES)
        raise ConfigResolutionError(
            "No config argument provided and no default config file was found in the current directory.\n"
            f"Expected one of: {expected}\n"
            "Hint: pass --config PATH or create a config.yaml (or cruncher.yaml) in this directory."
        )
    rendered = "\n".join(f"- {path.relative_to(cwd_path).as_posix()}" for path in matches)
    raise ConfigResolutionError(
        f"Multiple config files found in the current directory:\n{rendered}\nHint: pass --config PATH to disambiguate."
    )


def looks_like_config_path(value: str, *, cwd: Path | None = None) -> bool:
    """Heuristic: treat as config if it looks like a yaml path or exists on disk."""
    cwd_path = (cwd or Path.cwd()).expanduser().resolve()
    path = Path(value).expanduser()
    if path.suffix.lower() in {".yaml", ".yml"}:
        return True
    normalized = _normalize_path(path, cwd_path)
    return normalized.is_file()


def parse_config_and_value(
    args: Sequence[str] | None,
    config_option: Path | None,
    *,
    value_label: str,
    command_hint: str,
    cwd: Path | None = None,
) -> tuple[Path, str]:
    """Resolve config + a required positional value from mixed args."""
    items = list(args or [])
    cwd_path = (cwd or Path.cwd()).expanduser().resolve()
    if config_option is not None:
        if len(items) != 1:
            raise ConfigResolutionError(f"Expected {value_label}. Example: {command_hint}")
        return resolve_config_path(config_option, cwd=cwd_path), items[0]
    if not items:
        raise ConfigResolutionError(f"Missing {value_label}. Example: {command_hint}")
    if len(items) == 1:
        if looks_like_config_path(items[0], cwd=cwd_path):
            raise ConfigResolutionError(
                f"Missing {value_label} after config path '{items[0]}'. "
                f"Example: {command_hint} --config path/to/config.yaml"
            )
        return resolve_config_path(None, cwd=cwd_path), items[0]
    if len(items) == 2:
        first, second = items
        first_is_cfg = looks_like_config_path(first, cwd=cwd_path)
        second_is_cfg = looks_like_config_path(second, cwd=cwd_path)
        if first_is_cfg and second_is_cfg:
            raise ConfigResolutionError(
                f"Both arguments look like config paths. Example: {command_hint} --config path/to/config.yaml"
            )
        if first_is_cfg and not second_is_cfg:
            return resolve_config_path(Path(first), cwd=cwd_path), second
        if second_is_cfg and not first_is_cfg:
            return resolve_config_path(Path(second), cwd=cwd_path), first
        return resolve_config_path(Path(first), cwd=cwd_path), second
    raise ConfigResolutionError(f"Too many arguments. Example: {command_hint}")
