"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_plot_output_dir.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.opal.src.cli.commands.plot import _resolve_output_dir
from dnadesign.opal.src.storage.workspace import CampaignWorkspace


def test_plot_output_dir_template(tmp_path):
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir(parents=True, exist_ok=True)
    ws = CampaignWorkspace(config_path=campaign_dir / "campaign.yaml", workdir=campaign_dir)
    out_cfg = {"dir": "{campaign}/plots/{kind}/{name}{round_suffix}"}
    out_dir = _resolve_output_dir(
        out_cfg,
        campaign_dir=campaign_dir,
        workspace=ws,
        plot_name="plot_a",
        plot_kind="scatter",
        round_suffix="_r2",
    )
    assert out_dir == (campaign_dir / "plots" / "scatter" / "plot_a_r2").resolve()
