from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core import reporting
from dnadesign.densegen.src.core.reporting import collect_report_data
from dnadesign.densegen.tests.config_fixtures import write_minimal_config


def test_report_strict_raises_on_missing_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    write_minimal_config(cfg_path)

    loaded = load_config(cfg_path)
    with pytest.raises(ValueError, match="outputs"):
        collect_report_data(loaded.root, cfg_path, strict=True)


def test_reporting_facade_exposes_data_collection_only() -> None:
    assert hasattr(reporting, "collect_report_data")
    assert not hasattr(reporting, "write_report")
