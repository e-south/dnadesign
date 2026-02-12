"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/pipeline/sigma70.py

Sigma70 transform that emits generic kmer features and span_link effects.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from ..core import PluginError, Record, Span
from ..core.record import Display, Effect, Feature


@dataclass(frozen=True)
class SigmaVariant:
    name: str
    upstream: str
    downstream: str


_DEFAULT_VARIANTS: Sequence[SigmaVariant] = (
    SigmaVariant(name="sigma70_high", upstream="TTGACA", downstream="TATAAT"),
    SigmaVariant(name="sigma70_mid", upstream="ACCGCG", downstream="TATAAT"),
    SigmaVariant(name="sigma70_low", upstream="GCAGGT", downstream="TATAAT"),
)


@dataclass
class Sigma70Transform:
    variants: Sequence[SigmaVariant] = field(default_factory=lambda: list(_DEFAULT_VARIANTS))
    spacer_min: int = 16
    spacer_max: int = 18
    label_mode: Literal["inner", "outer", "observed", "custom"] = "inner"
    label_text: str | None = None
    inner_margin_bp: float | None = None
    on_multiple_matches: Literal["error", "first"] = "error"

    def __init__(
        self,
        variants: Sequence[dict] | None = None,
        spacer_min: int = 16,
        spacer_max: int = 18,
        label_mode: str = "inner",
        label_text: str | None = None,
        inner_margin_bp: float | None = None,
        on_multiple_matches: str = "error",
        **kwargs: object,
    ) -> None:
        if kwargs:
            raise PluginError(f"Unknown sigma70 parameter(s): {sorted(kwargs.keys())}")
        self.spacer_min = int(spacer_min)
        self.spacer_max = int(spacer_max)
        lm = str(label_mode).strip().lower()
        if lm not in {"inner", "outer", "observed", "custom"}:
            raise PluginError("sigma70.label_mode must be one of inner|outer|observed|custom")
        self.label_mode = lm  # type: ignore[assignment]
        self.label_text = label_text
        om = str(on_multiple_matches).strip().lower()
        if om not in {"error", "first"}:
            raise PluginError("sigma70.on_multiple_matches must be one of error|first")
        self.on_multiple_matches = om  # type: ignore[assignment]
        self.inner_margin_bp = float(inner_margin_bp) if inner_margin_bp is not None else None
        if variants is not None:
            self.variants = tuple(SigmaVariant(**v) for v in variants)
        else:
            self.variants = tuple(_DEFAULT_VARIANTS)

    @staticmethod
    def _find_all(seq: str, sub: str) -> list[int]:
        out: list[int] = []
        i = seq.find(sub, 0)
        while i != -1:
            out.append(i)
            i = seq.find(sub, i + 1)
        return out

    def apply(self, record: Record) -> Record:
        seq = record.sequence.upper()
        matches: list[tuple[SigmaVariant, int, int]] = []

        for variant in self.variants:
            ups = self._find_all(seq, variant.upstream)
            dns = self._find_all(seq, variant.downstream)
            for s35 in ups:
                for spacer in range(self.spacer_min, self.spacer_max + 1):
                    s10 = s35 + len(variant.upstream) + spacer
                    if s10 in dns:
                        matches.append((variant, s35, s10))
                        break

        if len(matches) > 1 and self.on_multiple_matches == "error":
            found = sorted({m[0].name for m in matches})
            raise PluginError(f"Multiple sigma70 matches found in record '{record.id}': {found}")
        if len(matches) > 1 and self.on_multiple_matches == "first":
            match = min(matches, key=lambda m: (m[1], m[2]))
        elif len(matches) == 1:
            match = matches[0]
        else:
            return record

        variant, s35, s10 = match
        strength_raw = variant.name.split("_", 1)[-1] if "_" in variant.name else variant.name
        strength = strength_raw.lower()

        # Resolve canonical kmer signatures to existing feature ids when available.
        existing_kmer_ids: dict[tuple[str, int, int, str | None, str], str | None] = {}
        for feature in record.features:
            key = (
                feature.kind,
                feature.span.start,
                feature.span.end,
                feature.span.strand,
                (feature.label or "").upper(),
            )
            existing_kmer_ids[key] = feature.id

        sigma_features: list[Feature] = []
        upstream_key = ("kmer", s35, s35 + len(variant.upstream), "fwd", variant.upstream.upper())
        upstream_id = existing_kmer_ids.get(upstream_key)
        if upstream_id is None:
            upstream_id = f"{record.id}:sigma70:-35"
            sigma_features.append(
                Feature(
                    id=upstream_id,
                    kind="kmer",
                    span=Span(start=s35, end=s35 + len(variant.upstream), strand="fwd"),
                    label=variant.upstream,
                    tags=("sigma",),
                    attrs={"strength": strength, "piece": "-35"},
                    render={"track": 0, "priority": 0},
                )
            )

        downstream_key = ("kmer", s10, s10 + len(variant.downstream), "fwd", variant.downstream.upper())
        downstream_id = existing_kmer_ids.get(downstream_key)
        if downstream_id is None:
            downstream_id = f"{record.id}:sigma70:-10"
            sigma_features.append(
                Feature(
                    id=downstream_id,
                    kind="kmer",
                    span=Span(start=s10, end=s10 + len(variant.downstream), strand="fwd"),
                    label=variant.downstream,
                    tags=("sigma",),
                    attrs={"strength": strength, "piece": "-10"},
                    render={"track": 0, "priority": 0},
                )
            )

        observed = s10 - (s35 + len(variant.upstream))
        if self.label_mode == "custom" and self.label_text:
            label = self.label_text
        elif self.label_mode == "observed":
            label = f"{observed + 1} bp"
        elif self.label_mode == "outer":
            label = f"{self.spacer_min + 1}-{self.spacer_max + 1} bp"
        else:
            label = f"{self.spacer_min + 1}-{self.spacer_max - 1} bp"

        effect = Effect(
            kind="span_link",
            target={"from_feature_id": upstream_id, "to_feature_id": downstream_id},
            params={
                "label": label,
                "inner_margin_bp": self.inner_margin_bp,
                "lane": "top",
            },
            render={"track": 0, "priority": 0},
        )

        display = Display(
            overlay_text=record.display.overlay_text,
            tag_labels={**dict(record.display.tag_labels), "sigma": f"σ70 {strength}"},
        )

        return Record(
            id=record.id,
            alphabet=record.alphabet,
            sequence=record.sequence,
            features=tuple(list(record.features) + sigma_features),
            effects=tuple(list(record.effects) + [effect]),
            display=display,
            meta=dict(record.meta),
        ).validate()
