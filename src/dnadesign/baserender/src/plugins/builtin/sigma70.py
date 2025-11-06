"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/plugins/builtin/sigma70.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Tuple

from ...contracts import PluginError
from ...model import Annotation, Guide, SeqRecord


@dataclass(frozen=True)
class SigmaVariant:
    name: str
    upstream: str  # -35
    downstream: str  # -10


_DEFAULT_VARIANTS: Sequence[SigmaVariant] = (
    SigmaVariant(name="sigma70_high", upstream="TTGACA", downstream="TATAAT"),
    SigmaVariant(name="sigma70_mid", upstream="ACCGCG", downstream="TATAAT"),
    SigmaVariant(name="sigma70_low", upstream="GCAGGT", downstream="TATAAT"),
)


@dataclass
class Sigma70Plugin:
    name: str = "sigma70"
    variants: Sequence[SigmaVariant] = field(
        default_factory=lambda: list(_DEFAULT_VARIANTS)
    )
    spacer_min: int = 16
    spacer_max: int = 18
    # How to label the spacer line:
    #  - "inner":    (spacer_min+1)-(spacer_max-1) in bp, e.g. 15..19 -> "16–18 bp"
    #  - "outer":    (spacer_min+1)-(spacer_max+1), original behavior (e.g. "16–20 bp")
    #  - "observed": per-record observed spacer in bp (see note below)
    #  - "custom":   use label_text verbatim
    label_mode: Literal["inner", "outer", "observed", "custom"] = "inner"
    label_text: Optional[str] = None
    # Default inner margin between the link and boxes (in bases). Can be overridden per-guide.
    inner_margin_bp: Optional[float] = None

    def __init__(
        self,
        variants: Sequence[dict] | None = None,
        spacer_min: int = 16,
        spacer_max: int = 18,
        label_mode: str = "inner",
        label_text: Optional[str] = None,
        inner_margin_bp: Optional[float] = None,
        **_: object,
    ):
        self.name = "sigma70"
        self.spacer_min = int(spacer_min)
        self.spacer_max = int(spacer_max)
        self.label_text = label_text
        lm = str(label_mode).lower().strip()
        self.label_mode = (
            "inner" if lm not in {"inner", "outer", "observed", "custom"} else lm
        )  # type: ignore[assignment]
        self.inner_margin_bp = (
            float(inner_margin_bp) if inner_margin_bp is not None else None
        )
        if variants:
            self.variants = tuple(SigmaVariant(**v) for v in variants)
        else:
            self.variants = tuple(_DEFAULT_VARIANTS)

    def _find_all(self, sub: str, s: str) -> List[int]:
        out: List[int] = []
        i = s.find(sub, 0)
        while i != -1:
            out.append(i)
            i = s.find(sub, i + 1)
        return out

    def apply(self, record: SeqRecord) -> SeqRecord:
        """
        Strict behavior:
        - Select at most ONE variant per sequence (first match by declared order).
        - Annotate exactly one paired (-35, -10) occurrence (first valid pair).
        - Sigma annotations are forward-only and occupy track 0 (payload.priority=0).
        - Always provide a 'sigma_link' guide with the spacer label. If σ boxes are
          already present in the dataset, the link aligns to their assigned track;
          otherwise it defaults to track 0 (the plugin's a track).
        """
        seq = record.sequence.upper()
        matches: List[Tuple[SigmaVariant, int, int]] = (
            []
        )  # (variant, start_35, start_10)

        for v in self.variants:
            ups = self._find_all(v.upstream, seq)
            dns = self._find_all(v.downstream, seq)
            for s35 in ups:
                for spacer in range(self.spacer_min, self.spacer_max + 1):
                    s10 = s35 + len(v.upstream) + spacer
                    if s10 in dns:
                        matches.append((v, s35, s10))
                        break
                if any(m[0] is v for m in matches):
                    break

        distinct_variants = {m[0].name for m in matches}
        if len(distinct_variants) > 1:
            raise PluginError(
                f"Multiple sigma70 variants matched in sequence '{record.id}': "
                f"{sorted(distinct_variants)}. Expected at most one."
            )

        anns: List[Annotation] = []
        guides: List[Guide] = []

        # Guard: avoid stacking the same k-mer when already present in the dataset
        # Match on exact (strand, start, length, letters), independent of tag.
        existing = {
            (a.strand, a.start, a.length, a.label.upper()) for a in record.annotations
        }

        if matches:
            v, s35, s10 = matches[0]
            # Single, unified tag for all sigma variants (one color in the plot).
            tag_base = "sigma"
            strength_raw = v.name.split("_", 1)[-1] if "_" in v.name else v.name
            # Keep 'low|mid|high' exactly as encoded by the variant name
            strength = strength_raw.lower()
            prio = {"priority": 0, "group": tag_base, "strength": strength}

            # Upstream (-35) and downstream (-10), same tag (shared hue)
            up_key = ("fwd", s35, len(v.upstream), v.upstream.upper())
            dn_key = ("fwd", s10, len(v.downstream), v.downstream.upper())
            if up_key not in existing:
                anns.append(
                    Annotation(
                        start=s35,
                        length=len(v.upstream),
                        strand="fwd",
                        label=v.upstream,
                        tag=tag_base,
                        payload=prio,
                    )
                )
            if dn_key not in existing:
                anns.append(
                    Annotation(
                        start=s10,
                        length=len(v.downstream),
                        strand="fwd",
                        label=v.downstream,
                        tag=tag_base,
                        payload=prio,
                    )
                )

            # Sandwiched span between boxes, same-height track; label must be 1-based range
            # Observed spacer between end(-35) and start(-10), in bases:
            observed = s10 - (s35 + len(v.upstream))
            # Human-facing label:
            if self.label_mode == "custom" and self.label_text:
                label_str = self.label_text
            elif self.label_mode == "observed":
                # Historically the label added +1; keep that convention for readability.
                label_str = f"{observed + 1} bp"
            elif self.label_mode == "outer":
                label_str = f"{self.spacer_min + 1}-{self.spacer_max + 1} bp"
            else:  # "inner" (default)
                label_str = f"{self.spacer_min + 1}-{self.spacer_max - 1} bp"

            link_track = 0

            guides.append(
                Guide(
                    kind="sigma_link",
                    start=s35,  # start of -35
                    end=s10,  # start of -10
                    label=label_str,
                    payload={
                        "track": link_track,
                        "up_len": len(v.upstream),
                        "group": tag_base,  # 'sigma' (for completeness)
                        "strength": strength,  # <-- expose plugin’s decision
                        # Prefer bp-based margin; renderer also supports legacy frac.
                        **(
                            {"inner_margin_bp": float(self.inner_margin_bp)}
                            if self.inner_margin_bp is not None
                            else {}
                        ),
                    },
                )
            )

        return record.with_extra(annotations=anns, guides=guides)
