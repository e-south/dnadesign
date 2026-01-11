"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/config_resolver.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

CANDIDATE_CONFIG_FILENAMES: tuple[str, ...] = (
    "cruncher.yaml",
    "cruncher.yml",
    "config.yaml",
    "config.yml",
)
WORKSPACE_ENV_VAR = "CRUNCHER_WORKSPACE"
WORKSPACE_ROOTS_ENV_VAR = "CRUNCHER_WORKSPACE_ROOTS"
DEFAULT_WORKSPACE_ENV_VAR = "CRUNCHER_DEFAULT_WORKSPACE"
CONFIG_ENV_VAR = "CRUNCHER_CONFIG"
NONINTERACTIVE_ENV_VAR = "CRUNCHER_NONINTERACTIVE"


class ConfigResolutionError(ValueError):
    """Raised when a config path cannot be resolved."""


@dataclass(frozen=True)
class WorkspaceCandidate:
    name: str
    root: Path
    config_path: Path
    catalog_path: Path


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_interactive() -> bool:
    if _env_truthy(NONINTERACTIVE_ENV_VAR) or _env_truthy("CI"):
        return False
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


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


def _candidate_configs_in_dir(directory: Path) -> list[Path]:
    return [directory / name for name in CANDIDATE_CONFIG_FILENAMES if (directory / name).is_file()]


def _resolve_config_in_dir(directory: Path, *, cwd: Path, log: bool, label: str) -> Path | None:
    matches = _candidate_configs_in_dir(directory)
    if len(matches) == 1:
        resolved = matches[0].resolve()
        if log:
            try:
                rel = resolved.relative_to(cwd)
                logger.info("Using config from %s: ./%s", label, rel.as_posix())
            except ValueError:
                logger.info("Using config from %s: %s", label, resolved)
        return resolved
    if len(matches) > 1:
        rendered = "\n".join(f"- {path.relative_to(directory).as_posix()}" for path in matches)
        raise ConfigResolutionError(
            f"Multiple config files found in {label} directory:\n{rendered}\nHint: pass --config PATH to disambiguate."
        )
    return None


def _find_git_root(cwd: Path) -> Path | None:
    for parent in (cwd, *cwd.parents):
        git_path = parent / ".git"
        if git_path.exists():
            return parent
    return None


def workspace_search_roots(cwd: Path | None = None) -> list[Path]:
    cwd_path = (cwd or Path.cwd()).expanduser().resolve()
    roots: list[Path] = []
    env_value = os.environ.get(WORKSPACE_ROOTS_ENV_VAR, "")
    if env_value:
        for raw in env_value.split(os.pathsep):
            if raw.strip():
                roots.append(_normalize_path(Path(raw.strip()), cwd_path))
    git_root = _find_git_root(cwd_path)
    if git_root:
        roots.append(git_root / "workspaces")
        roots.append(git_root / "workspace")
        roots.append(git_root / "src" / "dnadesign" / "cruncher" / "workspaces")
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        unique.append(resolved)
        seen.add(resolved)
    return unique


def discover_workspaces(cwd: Path | None = None) -> list[WorkspaceCandidate]:
    cwd_path = (cwd or Path.cwd()).expanduser().resolve()
    candidates: list[WorkspaceCandidate] = []
    seen_configs: set[Path] = set()
    for root in workspace_search_roots(cwd_path):
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            configs = _candidate_configs_in_dir(entry)
            if len(configs) != 1:
                continue
            config_path = configs[0].resolve()
            if config_path in seen_configs:
                continue
            seen_configs.add(config_path)
            candidates.append(
                WorkspaceCandidate(
                    name=entry.name,
                    root=entry.resolve(),
                    config_path=config_path,
                    catalog_path=(entry / ".cruncher" / "catalog.json").resolve(),
                )
            )
    candidates.sort(key=lambda item: (item.name, str(item.config_path)))
    return candidates


def _format_workspace_list(workspaces: Sequence[WorkspaceCandidate]) -> str:
    if not workspaces:
        return "- (none)"
    max_name = max(len(item.name) for item in workspaces)
    lines = []
    for idx, item in enumerate(workspaces, start=1):
        padded = item.name.ljust(max_name)
        lines.append(f"  [{idx}] {padded}  {item.config_path}")
    return "\n".join(lines)


def _resolve_workspace_path(path: Path, *, cwd: Path) -> Path:
    normalized = _normalize_path(path, cwd)
    if normalized.is_dir():
        matches = _candidate_configs_in_dir(normalized)
        if len(matches) == 1:
            return matches[0].resolve()
        if not matches:
            expected = ", ".join(CANDIDATE_CONFIG_FILENAMES)
            raise ConfigResolutionError(
                f"No config file found in workspace directory: {normalized}\nExpected one of: {expected}"
            )
        rendered = "\n".join(f"- {match.name}" for match in matches)
        raise ConfigResolutionError(
            f"Multiple config files found in workspace directory {normalized}:\n{rendered}\n"
            "Hint: pass --config PATH to disambiguate."
        )
    return _resolve_explicit_config(normalized, cwd=cwd)


def _resolve_workspace_selector(
    selector: str,
    *,
    cwd: Path,
    log: bool,
) -> Path:
    selector = selector.strip()
    workspaces = discover_workspaces(cwd)
    if selector.isdigit():
        index = int(selector)
        if index < 1 or index > len(workspaces):
            rendered = _format_workspace_list(workspaces)
            raise ConfigResolutionError(
                f"Workspace index {index} is out of range.\nDiscovered {len(workspaces)} workspace configs:\n{rendered}"
            )
        chosen = workspaces[index - 1]
        if log:
            logger.info(
                'Using workspace "%s" config: %s',
                chosen.name,
                chosen.config_path,
            )
        return chosen.config_path
    matches = [item for item in workspaces if item.name == selector]
    if len(matches) == 1:
        chosen = matches[0]
        if log:
            logger.info(
                'Using workspace "%s" config: %s',
                chosen.name,
                chosen.config_path,
            )
        return chosen.config_path
    if len(matches) > 1:
        rendered = "\n".join(f"- {item.config_path}" for item in matches)
        raise ConfigResolutionError(
            f"Workspace name '{selector}' is ambiguous. Matches:\n{rendered}\n"
            "Hint: use --workspace <index> or --config PATH to disambiguate."
        )
    path_candidate = Path(selector).expanduser()
    looks_like_path = path_candidate.suffix.lower() in {".yaml", ".yml"} or any(
        sep in selector for sep in (os.sep, os.altsep) if sep
    )
    if looks_like_path or path_candidate.exists():
        return _resolve_workspace_path(path_candidate, cwd=cwd)
    rendered = _format_workspace_list(workspaces)
    raise ConfigResolutionError(
        f"Workspace '{selector}' was not found.\nDiscovered {len(workspaces)} workspace configs:\n{rendered}\n"
        "Hint: use --workspace <index>, --workspace <name>, or --config PATH."
    )


def _prompt_for_workspace(
    workspaces: Sequence[WorkspaceCandidate],
) -> WorkspaceCandidate | None:
    if not _is_interactive() or not workspaces:
        return None
    from rich.console import Console
    from rich.prompt import Prompt

    console = Console()
    console.print("Discovered workspace configs:")
    console.print(_format_workspace_list(workspaces))
    choices = [str(idx) for idx in range(1, len(workspaces) + 1)]
    selection = Prompt.ask("Select workspace index", choices=choices)
    return workspaces[int(selection) - 1]


def resolve_config_path(config: Path | None, *, cwd: Path | None = None, log: bool = True) -> Path:
    """Resolve a config path from an explicit argument or from workspace discovery."""
    cwd_path = (cwd or Path.cwd()).expanduser().resolve()
    if config is not None:
        return _resolve_explicit_config(config, cwd=cwd_path)
    env_config = os.environ.get(CONFIG_ENV_VAR)
    if env_config:
        return _resolve_explicit_config(Path(env_config), cwd=cwd_path)
    selector = os.environ.get(WORKSPACE_ENV_VAR)
    if selector:
        return _resolve_workspace_selector(selector, cwd=cwd_path, log=log)
    resolved = _resolve_config_in_dir(cwd_path, cwd=cwd_path, log=log, label="CWD")
    if resolved is not None:
        return resolved
    for parent in cwd_path.parents:
        resolved = _resolve_config_in_dir(
            parent,
            cwd=cwd_path,
            log=log,
            label=f"parent directory {parent}",
        )
        if resolved is not None:
            return resolved
    workspaces = discover_workspaces(cwd_path)
    if not workspaces:
        expected = ", ".join(CANDIDATE_CONFIG_FILENAMES)
        roots = workspace_search_roots(cwd_path)
        roots_rendered = "\n".join(f"- {root}" for root in roots) if roots else "- (none)"
        raise ConfigResolutionError(
            "No config argument provided and no default config file was found in the current directory.\n"
            f"Expected one of: {expected}\n"
            "Hint: pass --config PATH, set CRUNCHER_CONFIG, or create a config.yaml "
            "(or cruncher.yaml) in this directory.\n"
            "Workspace discovery searched:\n"
            f"{roots_rendered}\n"
            f"To add roots: set {WORKSPACE_ROOTS_ENV_VAR}=/path/a{os.pathsep}/path/b"
        )
    if len(workspaces) == 1:
        chosen = workspaces[0]
        if log:
            logger.info(
                'Using workspace "%s" config: %s (Tip: set %s=%s to make this implicit.)',
                chosen.name,
                chosen.config_path,
                DEFAULT_WORKSPACE_ENV_VAR,
                chosen.name,
            )
        return chosen.config_path
    default_name = os.environ.get(DEFAULT_WORKSPACE_ENV_VAR)
    if default_name:
        matches = [item for item in workspaces if item.name == default_name]
        if len(matches) == 1:
            chosen = matches[0]
            if log:
                logger.info(
                    'Using workspace "%s" config: %s (from %s).',
                    chosen.name,
                    chosen.config_path,
                    DEFAULT_WORKSPACE_ENV_VAR,
                )
            return chosen.config_path
        rendered = _format_workspace_list(workspaces)
        raise ConfigResolutionError(
            f"{DEFAULT_WORKSPACE_ENV_VAR}={default_name} did not match a unique workspace.\n"
            f"Discovered {len(workspaces)} workspace configs:\n{rendered}\n"
            "Hint: update CRUNCHER_DEFAULT_WORKSPACE or pass --workspace."
        )
    chosen = _prompt_for_workspace(workspaces)
    if chosen is not None:
        if log:
            logger.info(
                'Using workspace "%s" config: %s',
                chosen.name,
                chosen.config_path,
            )
        return chosen.config_path
    rendered = _format_workspace_list(workspaces)
    sample = workspaces[0]
    raise ConfigResolutionError(
        "No config found in CWD or parents.\n"
        f"Discovered {len(workspaces)} workspace configs:\n{rendered}\n\n"
        "Choose one:\n"
        "  cruncher --workspace 1 catalog list\n"
        f"  cruncher -w {sample.name} catalog list\n"
        f"  cruncher --config {sample.config_path} catalog list"
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
