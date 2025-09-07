"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/usr/src/config.py

Load/save remote configuration (remotes.yaml). SSH only for v0.1.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import os
import sys

import yaml

from .errors import RemoteConfigError

_DEFAULT_SEARCH = [
    Path("./usr/remotes.yaml"),
    Path.home() / ".config" / "usr" / "remotes.yaml",
    Path.home() / ".usr" / "remotes.yaml",
]


@dataclass(frozen=True)
class SSHRemoteConfig:
    name: str
    host: str
    user: str
    base_dir: str
    ssh_key_env: Optional[str] = None

    @property
    def ssh_target(self) -> str:
        return f"{self.user}@{self.host}"

    def dataset_path(self, dataset: str) -> str:
        # Remote path for a dataset directory
        return str(Path(self.base_dir) / dataset)

    def rsync_url(self, dataset: str) -> str:
        # For rsync, append trailing slash to copy directory contents
        return f"{self.ssh_target}:{self.dataset_path(dataset)}/"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        raise RemoteConfigError(f"Failed to read config {path}: {e}")


def _dump_yaml(path: Path, obj: Dict) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=True)


def locate_config(custom: Optional[Path] = None) -> Path:
    if custom:
        return custom
    for p in _DEFAULT_SEARCH:
        if p.exists():
            return p
    # default to repo-local path
    return _DEFAULT_SEARCH[0]


def load_all(custom: Optional[Path] = None) -> Dict[str, SSHRemoteConfig]:
    path = locate_config(custom)
    data = _load_yaml(path)
    remotes = {}
    for name, rec in (data.get("remotes") or {}).items():
        if (rec or {}).get("type", "ssh") != "ssh":
            # We only support SSH for now (explicitly)
            continue
        try:
            remotes[name] = SSHRemoteConfig(
                name=name,
                host=rec["host"],
                user=rec["user"],
                base_dir=rec["base_dir"],
                ssh_key_env=rec.get("ssh_key_env"),
            )
        except KeyError as ke:
            raise RemoteConfigError(
                f"Remote '{name}' missing required key: {ke}"
            ) from None
    return remotes


def save_remote(cfg: SSHRemoteConfig, custom: Optional[Path] = None) -> Path:
    path = locate_config(custom)
    data = _load_yaml(path)
    data.setdefault("remotes", {})
    data["remotes"][cfg.name] = {
        "type": "ssh",
        "host": cfg.host,
        "user": cfg.user,
        "base_dir": cfg.base_dir,
        "ssh_key_env": cfg.ssh_key_env,
    }
    _dump_yaml(path, data)
    return path


def get_remote(name: str, custom: Optional[Path] = None) -> SSHRemoteConfig:
    remotes = load_all(custom)
    if name not in remotes:
        raise RemoteConfigError(
            f"Unknown remote '{name}'. Define it with 'usr remotes add {name} --type ssh ...'."
        )
    cfg = remotes[name]
    if cfg.ssh_key_env:
        # Validate env presence early (assertive programming)
        if cfg.ssh_key_env not in os.environ:
            raise RemoteConfigError(
                f"Environment variable '{cfg.ssh_key_env}' not set (SSH key path)."
            )
    return cfg
