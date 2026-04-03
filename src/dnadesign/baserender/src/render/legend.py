"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/render/legend.py

Legend construction from feature tags and record-level display tag label metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..core import Record


def legend_entries_for_record(record: Record) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    labels = dict(record.display.tag_labels)
    excluded = {
        str(tag).strip()
        for tag in record.meta.get("legend_exclude_tags", ())
        if isinstance(tag, str) and str(tag).strip()
    }

    for feature in record.features:
        for tag in feature.tags:
            if tag in seen or tag in excluded:
                continue
            seen.add(tag)
            entries.append((tag, labels.get(tag, tag)))

    return entries
