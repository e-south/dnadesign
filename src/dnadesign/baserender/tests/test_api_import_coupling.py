"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_api_import_coupling.py

Regression test preventing render-stack imports during API module import.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys


def test_public_api_catalog_names_are_exact_reexports() -> None:
    from dnadesign.baserender.src.public import api, catalog

    for name in (
        "get_adapter_descriptor",
        "get_render_contract_descriptor",
        "get_renderer_descriptor",
        "list_adapters",
        "list_render_contracts",
        "list_renderers",
    ):
        assert getattr(api, name) is getattr(catalog, name)


def test_api_module_import_does_not_preload_matplotlib() -> None:
    code = "import sys\nimport dnadesign.baserender.src.public.api\nprint('matplotlib' in sys.modules)\n"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip().endswith("False")


def test_package_root_import_does_not_preload_render_stack() -> None:
    code = "\n".join(
        [
            "import sys",
            "import dnadesign.baserender",
            "print('matplotlib' in sys.modules)",
            "print('numpy' in sys.modules)",
            "print('dnadesign.baserender.src.public.api' in sys.modules)",
            "print('dnadesign.baserender.src.render' in sys.modules)",
        ]
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.splitlines() == ["False", "False", "False", "False"]


def test_catalog_access_does_not_preload_adapters_or_numpy() -> None:
    code = "\n".join(
        [
            "import sys",
            "import dnadesign.baserender as baserender",
            "print('densegen_tfbs' in baserender.list_adapters())",
            "print('numpy' in sys.modules)",
            "print('dnadesign.baserender.src.adapters.densegen_tfbs' in sys.modules)",
            "print('dnadesign.baserender.src.public.api' in sys.modules)",
        ]
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert proc.stdout.splitlines() == ["True", "False", "False", "False"]
