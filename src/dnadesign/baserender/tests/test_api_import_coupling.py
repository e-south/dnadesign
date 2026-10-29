"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_api_import_coupling.py

Import-coupling and package-facade regression tests for BaseRender.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _assert_canonical_render_for_import_order(imports: list[str]) -> None:
    code = "\n".join(
        [
            "import importlib",
            *[f"importlib.import_module({module_name!r})" for module_name in imports],
            "baserender = importlib.import_module('dnadesign.baserender')",
            "internal = importlib.import_module('dnadesign.baserender.src')",
            "public = importlib.import_module('dnadesign.baserender.src.public')",
            "print(callable(baserender.render))",
            "print(baserender.render is public.render)",
            "print('render' in internal.__all__)",
        ]
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert proc.stdout.splitlines() == ["True", "True", "False"]


def test_public_api_catalog_names_are_exact_reexports() -> None:
    from dnadesign.baserender.src.public import api, catalog

    for name in (
        "get_adapter_descriptor",
        "get_render_contract_descriptor",
        "get_renderer_descriptor",
        "get_style_profile_descriptor",
        "get_transform_descriptor",
        "list_adapters",
        "list_render_contracts",
        "list_renderers",
        "list_style_profiles",
        "list_transforms",
    ):
        assert getattr(api, name) is getattr(catalog, name)


@pytest.mark.parametrize(
    "imports",
    [
        ["dnadesign.baserender"],
        ["dnadesign.baserender.src.render"],
    ],
    ids=["public-facade-first", "internal-render-package-first"],
)
def test_canonical_render_is_stable_across_import_orders(imports: list[str]) -> None:
    _assert_canonical_render_for_import_order(imports)


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
            "print('pydantic' in sys.modules)",
            "print('dnadesign.baserender.src.integrations.densegen.adapter' in sys.modules)",
            "print('dnadesign.baserender.src.public.api' in sys.modules)",
        ]
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert proc.stdout.splitlines() == ["True", "False", "False", "False", "False"]
