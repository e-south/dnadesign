"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/tests/test_parse_mode.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from pathlib import Path

import yaml


def test_run_parse_smoke(cwd_tmp, mini_cfg_path):
    # force parse mode
    cfg_dict = yaml.safe_load(Path(mini_cfg_path).read_text())
    cfg_dict["cruncher"]["mode"] = "parse"
    Path(mini_cfg_path).write_text(yaml.safe_dump(cfg_dict))

    # run main() directly – faster than subprocess
    from dnadesign.cruncher.main import main

    main(str(mini_cfg_path))  # should complete with no exceptions

    assert any((cwd_tmp / "results").rglob("*_logo.png"))
