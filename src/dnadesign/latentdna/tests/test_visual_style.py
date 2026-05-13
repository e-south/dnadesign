from dnadesign.latentdna.src.visual_style import scatter_style


def test_large_latent_scatter_style_prioritizes_visible_clouds() -> None:
    large = scatter_style(157_279)

    assert large.rasterized is True
    assert large.point_size >= 9.0
    assert large.alpha >= 0.60


def test_mid_sized_latent_scatter_style_remains_more_visible_than_large_clouds() -> None:
    mid = scatter_style(10_000)
    large = scatter_style(157_279)

    assert mid.point_size > large.point_size
    assert mid.alpha > large.alpha
