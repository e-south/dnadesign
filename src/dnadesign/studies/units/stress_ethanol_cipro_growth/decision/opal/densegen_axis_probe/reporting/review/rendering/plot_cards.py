"""Plot-card HTML helpers for DenseGen axis probe reviews."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .formatting import _e, _rel


def probe_plot_cards_html(plots: Sequence[Any], *, base_dir: Path) -> str:
    cards = []
    for path in plots:
        src = _rel(path, base_dir=base_dir)
        title = Path(str(path)).stem.replace("_", " ")
        cards.append(
            "<article>"
            f"<h3>{_e(title)}</h3>"
            f'<a href="{_e(src)}"><img src="{_e(src)}" alt="Probe plot: {_e(title)}"></a>'
            "</article>"
        )
    return "".join(cards)


def campaign_links_html(campaign_reviews: Sequence[Mapping[str, Any]], *, base_dir: Path) -> str:
    links = []
    for review in campaign_reviews:
        href = review.get("index_path") or review.get("review_path")
        links.append(
            "<li>"
            f'<a href="{_e(_rel(href, base_dir=base_dir))}"><code>{_e(review.get("run_key"))}</code></a>'
            f" round {_e(review.get('round_index'))} run <code>{_e(review.get('run_id'))}</code>"
            "</li>"
        )
    return "".join(links)


def configured_plot_cards_html(configured_plots: Sequence[Mapping[str, Any]], *, base_dir: Path) -> str:
    cards = [_configured_plot_card(entry, base_dir=base_dir) for entry in configured_plots]
    return "".join(cards) or "<p>No configured OPAL plot indexes found.</p>"


def _configured_plot_card(entry: Mapping[str, Any], *, base_dir: Path) -> str:
    plot_links = []
    plot_thumbs = []
    for plot in entry.get("plots") or []:
        if not isinstance(plot, Mapping):
            continue
        media_path = next(iter(plot.get("media_paths") or []), None)
        tidy_path = next(iter(plot.get("tidy_csv_paths") or []), None)
        manifest_path = plot.get("manifest_path")
        media_link = f'<a href="{_e(_rel(media_path, base_dir=base_dir))}">media</a>' if media_path else "media missing"
        tidy_link = f' · <a href="{_e(_rel(tidy_path, base_dir=base_dir))}">csv</a>' if tidy_path else ""
        manifest_link = (
            f' · <a href="{_e(_rel(manifest_path, base_dir=base_dir))}">manifest</a>' if manifest_path else ""
        )
        plot_links.append(
            "<li>"
            f"<code>{_e(plot.get('name'))}</code> ({_e(plot.get('kind'))}) "
            f"{media_link}{tidy_link}{manifest_link}"
            "</li>"
        )
        if media_path:
            media_src = _rel(media_path, base_dir=base_dir)
            caption = f"{plot.get('name')} ({plot.get('kind')})"
            plot_thumbs.append(
                "<figure>"
                f'<a href="{_e(media_src)}"><img src="{_e(media_src)}" '
                f'alt="{_e(entry.get("run_key"))}: {_e(caption)}"></a>'
                f"<figcaption>{_e(caption)}</figcaption>"
                "</figure>"
            )
    quality = entry.get("quality") or {}
    return (
        "<article>"
        f"<h3><code>{_e(entry.get('run_key'))}</code></h3>"
        f"<p>Status: <code>{_e(entry.get('status'))}</code>; "
        f"quality: <code>{_e(quality.get('status'))}</code>; "
        f"plots: <code>{_e(entry.get('plot_count'))}</code></p>"
        f"<ul>{''.join(plot_links) if plot_links else '<li>No manifest-backed configured plots.</li>'}</ul>"
        f'<div class="plot-thumb-grid">{"".join(plot_thumbs)}</div>'
        "</article>"
    )
