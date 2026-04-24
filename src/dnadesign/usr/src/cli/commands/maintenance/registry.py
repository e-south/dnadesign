"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/commands/maintenance/registry.py

USR CLI maintenance registry command implementations.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ....dataset import Dataset


@dataclass(frozen=True)
class MaintenanceDeps:
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None]
    prompt: Callable[[str], str]


def cmd_registry_freeze(args, *, deps: MaintenanceDeps) -> None:
    ds_name = deps.resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    dataset = Dataset(args.root, ds_name)
    with dataset.maintenance(reason="registry_freeze"):
        snap = dataset.freeze_registry()
    print(f"[registry-freeze] wrote {snap}")
