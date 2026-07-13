"""Strict campaign config path-resolution tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.opal.src.core.config_resolve import resolve_campaign_config_path
from dnadesign.opal.src.core.utils import OpalError


def test_campaign_directory_resolves_only_configs_campaign_yaml(tmp_path: Path) -> None:
    canonical = tmp_path / "configs" / "campaign.yaml"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("schema_version: opal.campaign.v3\n", encoding="utf-8")

    assert resolve_campaign_config_path(tmp_path, allow_dir=True) == canonical.resolve()


@pytest.mark.parametrize("relative", ["campaign.yaml", "campaign.yml", "opal.yaml", "opal.yml"])
def test_campaign_directory_rejects_noncanonical_config_locations(tmp_path: Path, relative: str) -> None:
    alternate = tmp_path / relative
    alternate.write_text("schema_version: opal.campaign.v3\n", encoding="utf-8")

    with pytest.raises(OpalError, match="configs/campaign.yaml"):
        resolve_campaign_config_path(tmp_path, allow_dir=True)


def test_opal_config_names_environment_override_is_not_supported(tmp_path: Path, monkeypatch) -> None:
    alternate = tmp_path / "configs" / "custom.yaml"
    alternate.parent.mkdir(parents=True)
    alternate.write_text("schema_version: opal.campaign.v3\n", encoding="utf-8")
    monkeypatch.setenv("OPAL_CONFIG_NAMES", "custom.yaml")

    with pytest.raises(OpalError, match="configs/campaign.yaml"):
        resolve_campaign_config_path(tmp_path, allow_dir=True)
