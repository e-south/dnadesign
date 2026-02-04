"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/cli_commands/context.py

Shared CLI wiring context for command registration.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ContextManager

from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class CliContext:
    console: Console
    make_table: Callable[..., Table]
    display_path: Callable[..., str]
    resolve_config_path: Callable[..., tuple]
    load_config_or_exit: Callable[..., object]
    run_root_for: Callable[..., object]
    list_dir_entries: Callable[..., list[str]]
    workspace_command: Callable[..., str]
    suppress_pyarrow_sysctl_warnings: Callable[[], ContextManager[None]]
    resolve_outputs_path_or_exit: Callable[..., object]
    warn_full_pool_strategy: Callable[..., None]
    default_config_missing_message: str
