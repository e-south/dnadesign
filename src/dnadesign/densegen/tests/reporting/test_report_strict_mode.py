from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core.reporting import write_report
from dnadesign.densegen.tests.config_fixtures import write_minimal_config


def test_report_strict_raises_on_missing_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    write_minimal_config(cfg_path)

    loaded = load_config(cfg_path)
    with pytest.raises(ValueError, match="outputs"):
        write_report(loaded.root, cfg_path, strict=True)
