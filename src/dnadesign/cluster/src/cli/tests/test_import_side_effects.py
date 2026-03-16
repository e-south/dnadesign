"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/cli/tests/test_import_side_effects.py

Regression tests for cluster CLI bootstrap side effects.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cluster_cli_bootstrap_does_not_import_matplotlib() -> None:
    repo_root = Path(__file__).resolve().parents[6]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    probe = (
        "import sys; "
        "import dnadesign.cluster.src.cli.app as _app; "
        "print(any(name.startswith('matplotlib') for name in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_cluster_public_contracts_do_not_import_heavy_plotting_or_method_stacks() -> None:
    repo_root = Path(__file__).resolve().parents[6]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    probe = (
        "import sys; "
        "import dnadesign.cluster as _cluster; "
        "print(any(name.startswith('matplotlib') or name.startswith('scanpy') for name in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_cluster_analysis_helpers_do_not_import_plotting_until_needed() -> None:
    repo_root = Path(__file__).resolve().parents[6]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    env["MPLCONFIGDIR"] = "/tmp/cluster-test-mpl"
    probe = (
        "import importlib, sys; "
        "importlib.import_module('dnadesign.cluster.src.analysis.composition'); "
        "importlib.import_module('dnadesign.cluster.src.analysis.diversity'); "
        "importlib.import_module('dnadesign.cluster.src.analysis.numeric_per_cluster'); "
        "print(any(name.startswith('matplotlib') or name.startswith('seaborn') for name in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
