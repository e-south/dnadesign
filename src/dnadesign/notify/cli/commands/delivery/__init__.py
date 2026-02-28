"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/commands/delivery/__init__.py

Delivery command registration and payload-format adapters for Notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .providers import format_for_provider
from .send_cmd import register_send_command

__all__ = ["format_for_provider", "register_send_command"]
