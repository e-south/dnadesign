"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/legend.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .model import SeqRecord


def legend_entries_for_record(r: SeqRecord) -> list[tuple[str, str]]:
    """
    Build the legend for one record.
    - Non-σ TFs: 'tf:<name>' → '<name>' (dedup, stable order).
    - σ70: prefer dataset-declared strength from 'tf:sigma70_*';
           fall back to plugin ('sigma' tag or 'sigma_link' guide) only if needed.
    """
    entries: list[tuple[str, str]] = []
    seen_tfs: set[str] = set()
    sigma_from_dataset: str | None = None
    sigma_from_plugin: str | None = None

    for a in r.annotations:
        tag = str(a.tag or "")
        tag_lower = tag.lower()
        if tag_lower.startswith("tf:"):
            name = tag_lower[3:]
            if name.startswith("sigma70_"):
                for tok in name.split("_")[1:]:
                    if tok in {"low", "mid", "high"}:
                        if sigma_from_dataset is None:
                            sigma_from_dataset = tok
                        break
                # don't list sigma70_* again as a plain TF
                continue
            if name not in seen_tfs:
                seen_tfs.add(name)
                entries.append((f"tf:{name}", name))
        elif tag_lower == "sigma":
            st = (a.payload or {}).get("strength")
            if isinstance(st, str) and st.lower() in {"low", "mid", "high"}:
                if sigma_from_plugin is None:
                    sigma_from_plugin = st.lower()

    # Last-chance fallback: the plugin encodes strength on the sigma_link guide too.
    if sigma_from_dataset is None and sigma_from_plugin is None:
        for g in r.guides:
            if getattr(g, "kind", "") == "sigma_link":
                st = (g.payload or {}).get("strength")
                if isinstance(st, str) and st.lower() in {"low", "mid", "high"}:
                    sigma_from_plugin = st.lower()
                    break

    strength = sigma_from_dataset or sigma_from_plugin
    if strength:
        entries.append(("sigma", f"σ70 {strength}"))
    return entries
