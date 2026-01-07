import logging
from pathlib import Path

import pytest

from dnadesign.cruncher.cli.config_resolver import (
    CANDIDATE_CONFIG_FILENAMES,
    ConfigResolutionError,
    parse_config_and_value,
    resolve_config_path,
)


def test_resolve_config_from_cwd_single(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config_path = tmp_path / "cruncher.yaml"
    config_path.write_text("cruncher: {}\n")
    caplog.set_level(logging.INFO, logger="dnadesign.cruncher.cli.config_resolver")

    resolved = resolve_config_path(None, cwd=tmp_path)

    assert resolved == config_path.resolve()
    assert any("Using config from CWD: ./cruncher.yaml" in record.message for record in caplog.records)


def test_resolve_config_from_cwd_none(tmp_path: Path) -> None:
    with pytest.raises(ConfigResolutionError) as excinfo:
        resolve_config_path(None, cwd=tmp_path, log=False)
    message = str(excinfo.value)
    assert "No config argument provided" in message
    for name in CANDIDATE_CONFIG_FILENAMES:
        assert name in message


def test_resolve_config_from_cwd_multiple(tmp_path: Path) -> None:
    (tmp_path / "cruncher.yaml").write_text("cruncher: {}\n")
    (tmp_path / "config.yaml").write_text("cruncher: {}\n")
    with pytest.raises(ConfigResolutionError) as excinfo:
        resolve_config_path(None, cwd=tmp_path, log=False)
    message = str(excinfo.value)
    assert "Multiple config files found" in message
    assert "cruncher.yaml" in message
    assert "config.yaml" in message


def test_resolve_config_explicit_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("cruncher: {}\n")

    resolved = resolve_config_path(config_path, cwd=tmp_path, log=False)

    assert resolved == config_path.resolve()


def test_parse_config_and_value_single_config_path_errors(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("cruncher: {}\n")

    with pytest.raises(ConfigResolutionError) as excinfo:
        parse_config_and_value(
            [str(config_path)],
            None,
            value_label="RUN",
            command_hint="cruncher report <run_name>",
            cwd=tmp_path,
        )
    message = str(excinfo.value)
    assert "Missing RUN" in message
    assert "config.yaml" in message


def test_parse_config_and_value_single_value_uses_cwd_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("cruncher: {}\n")

    resolved_config, value = parse_config_and_value(
        ["sample_run_1"],
        None,
        value_label="RUN",
        command_hint="cruncher report <run_name>",
        cwd=tmp_path,
    )

    assert resolved_config == config_path.resolve()
    assert value == "sample_run_1"
