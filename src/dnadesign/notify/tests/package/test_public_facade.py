"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/package/test_public_facade.py

Stable, lightweight Notify integration facade tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys


def test_notify_root_exports_integration_contracts_without_cli_runtime() -> None:
    code = "\n".join(
        [
            "import sys",
            "from dnadesign.notify import NotifyConfigError, parse_notify_profile_webhook",
            "print(NotifyConfigError.__module__)",
            "print(parse_notify_profile_webhook.__module__)",
            "print('dnadesign.notify.cli' in sys.modules)",
            "print('dnadesign.notify.runtime.runner' in sys.modules)",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == [
        "dnadesign.notify.core.errors",
        "dnadesign.notify.core.contracts",
        "False",
        "False",
    ]
