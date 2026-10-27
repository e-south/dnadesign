"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/renderer.py

Renderer registry and pre-render contract enforcement for Record v1.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

from ..config import Style
from ..core import ContractError, Record, RenderingError, SchemaError, validate_record_kinds
from ..integrations import validate_record_renderer_compatibility
from .palette import Palette


class Renderer(Protocol):
    def preflight(
        self,
        record: Record,
        style: Style,
        palette: Palette,
        options: Mapping[str, object] | None = None,
    ) -> None: ...

    def render(
        self,
        record: Record,
        style: Style,
        palette: Palette,
        options: Mapping[str, object] | None = None,
    ): ...


RendererFactory = Callable[[], Renderer]


@dataclass(frozen=True)
class RendererDescriptor:
    name: str
    topology_kinds: tuple[str, ...]
    accepted_alphabets: tuple[str, ...]
    required_record_features: tuple[str, ...]
    optional_record_features: tuple[str, ...]
    docs_slug: str
    max_grid_records: int | None = None
    option_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class _RegisteredRenderer:
    descriptor: RendererDescriptor
    factory: RendererFactory


def _build_sequence_rows_renderer() -> Renderer:
    from .sequence_rows import SequenceRowsRenderer

    return SequenceRowsRenderer()


def _build_nucleotide_evidence_map_renderer() -> Renderer:
    from .sequence_rows import SequenceRowsRenderer

    return SequenceRowsRenderer()


def _build_hairpin_cartoon_renderer() -> Renderer:
    from .hairpin_cartoon import HairpinCartoonRenderer

    return HairpinCartoonRenderer()


def _build_topology_cartoon_renderer() -> Renderer:
    from .topology_cartoon import TopologyCartoonRenderer

    return TopologyCartoonRenderer()


def _build_junction_annealed_fragments_renderer() -> Renderer:
    from .junction_annealed_fragments import JunctionAnnealedFragmentsRenderer

    return JunctionAnnealedFragmentsRenderer()


def _build_junction_three_way_assembly_renderer() -> Renderer:
    from .junction_three_way_assembly import JunctionThreeWayAssemblyRenderer

    return JunctionThreeWayAssemblyRenderer()


@dataclass(frozen=True)
class _RendererRegistry:
    renderers: dict[str, _RegisteredRenderer]

    def get(self, name: str) -> Renderer:
        registered = self.renderers.get(name)
        if registered is None:
            raise RenderingError(f"Unknown renderer: {name}")
        return registered.factory()

    def descriptor(self, name: str) -> RendererDescriptor:
        registered = self.renderers.get(name)
        if registered is None:
            raise RenderingError(f"Unknown renderer: {name}")
        return registered.descriptor


_REGISTRY = _RendererRegistry(
    renderers={
        "sequence_rows": _RegisteredRenderer(
            descriptor=RendererDescriptor(
                name="sequence_rows",
                topology_kinds=("linear_ssdna", "linear_dsdna", "fragment_pool"),
                accepted_alphabets=("DNA", "IUPAC_DNA"),
                required_record_features=(),
                optional_record_features=("interval_annotation", "boundary_marker"),
                docs_slug="sequence-rows",
            ),
            factory=_build_sequence_rows_renderer,
        ),
        "nucleotide_evidence_map": _RegisteredRenderer(
            descriptor=RendererDescriptor(
                name="nucleotide_evidence_map",
                topology_kinds=(
                    "linear_ssdna",
                    "linear_dsdna",
                    "fragment_pool",
                    "circularized_linearized",
                    "hairpin_folded",
                    "branched_adapter",
                ),
                accepted_alphabets=("DNA", "IUPAC_DNA"),
                required_record_features=(),
                optional_record_features=("interval_annotation", "boundary_marker", "span_link"),
                docs_slug="nucleotide-evidence-map",
            ),
            factory=_build_nucleotide_evidence_map_renderer,
        ),
        "hairpin_cartoon": _RegisteredRenderer(
            descriptor=RendererDescriptor(
                name="hairpin_cartoon",
                topology_kinds=("hairpin_ssdna", "ssdna_hairpin"),
                accepted_alphabets=("DNA", "IUPAC_DNA"),
                required_record_features=("pair_map",),
                optional_record_features=("interval_annotation",),
                docs_slug="hairpin-cartoon",
            ),
            factory=_build_hairpin_cartoon_renderer,
        ),
        "topology_cartoon": _RegisteredRenderer(
            descriptor=RendererDescriptor(
                name="topology_cartoon",
                topology_kinds=("circular_dsdna_candidate", "branched_y", "fragment_pool", "circular_duplex"),
                accepted_alphabets=("DNA", "IUPAC_DNA"),
                required_record_features=(),
                optional_record_features=("interval_annotation",),
                docs_slug="topology-cartoon",
            ),
            factory=_build_topology_cartoon_renderer,
        ),
        "junction_annealed_fragments": _RegisteredRenderer(
            descriptor=RendererDescriptor(
                name="junction_annealed_fragments",
                topology_kinds=("fragment_pool",),
                accepted_alphabets=("DNA",),
                required_record_features=(),
                optional_record_features=(),
                docs_slug="junction-annealed-fragments",
                max_grid_records=1,
                option_keys=("fragment_ids",),
            ),
            factory=_build_junction_annealed_fragments_renderer,
        ),
        "junction_three_way_assembly": _RegisteredRenderer(
            descriptor=RendererDescriptor(
                name="junction_three_way_assembly",
                topology_kinds=("fragment_pool",),
                accepted_alphabets=("DNA",),
                required_record_features=(),
                optional_record_features=(),
                docs_slug="junction-three-way-assembly",
                max_grid_records=1,
                option_keys=("view", "junction_ids"),
            ),
            factory=_build_junction_three_way_assembly_renderer,
        ),
    }
)


def get_renderer(name: str) -> Renderer:
    return _REGISTRY.get(name)


def renderer_descriptors() -> tuple[RendererDescriptor, ...]:
    return tuple(_REGISTRY.descriptor(name) for name in sorted(_REGISTRY.renderers))


def get_renderer_descriptor(name: str) -> RendererDescriptor:
    return _REGISTRY.descriptor(name)


def validate_records_for_rendering(
    records: tuple[Record, ...] | list[Record],
    *,
    renderer_name: str,
    style: Style,
    palette: Palette,
    renderer_options: Mapping[str, object] | None = None,
) -> tuple[Record, ...]:
    """Validate one complete render batch before any renderer or figure is allocated."""

    try:
        descriptor = get_renderer_descriptor(renderer_name)
        options = dict(renderer_options or {})
        non_string_option_keys = [key for key in options if not isinstance(key, str)]
        if non_string_option_keys:
            raise SchemaError(f"renderer {renderer_name!r} option keys must be strings")
        unknown_options = sorted(set(options) - set(descriptor.option_keys))
        if unknown_options:
            raise SchemaError(
                f"renderer {renderer_name!r} received unknown options: {unknown_options}; "
                f"allowed options: {sorted(descriptor.option_keys)}"
            )
        validated_records: list[Record] = []
        for record in records:
            validated = record.validate()
            validate_record_kinds(validated)
            validate_record_renderer_compatibility(validated, renderer_name=renderer_name)
            validated_records.append(validated)
    except (ContractError, SchemaError) as exc:
        raise RenderingError(str(exc)) from exc

    renderer = get_renderer(renderer_name)
    for validated in validated_records:
        if options:
            renderer.preflight(validated, style, palette, options)
        else:
            renderer.preflight(validated, style, palette)
    return tuple(validated_records)


def render_record(
    record: Record,
    *,
    renderer_name: str,
    style: Style,
    palette: Palette,
    renderer_options: Mapping[str, object] | None = None,
):
    validated = validate_records_for_rendering(
        (record,),
        renderer_name=renderer_name,
        style=style,
        palette=palette,
        renderer_options=renderer_options,
    )[0]

    renderer = get_renderer(renderer_name)
    options = dict(renderer_options or {})
    if options:
        return renderer.render(validated, style, palette, options)
    return renderer.render(validated, style, palette)
