"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/publish.py

Publish YIU v4 bundles and BaseRender-ready view contracts.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.yiu.bundle_models import PayloadBundleManifest, PayloadVisualInventory
from dnadesign.cruncher.yiu.domain_models import NormalizedPayload
from dnadesign.cruncher.yiu.publish_inventory import (
    build_normalized_payload_dump,
    build_payload_bundle_manifest,
    build_payload_visual_inventory,
)
from dnadesign.cruncher.yiu.publish_io import (
    write_debug_render_jobs,
    write_normalized_payload_dump,
    write_payload_bundle_state,
    write_payload_bundle_views,
)
from dnadesign.cruncher.yiu.publish_layout import (
    build_payload_view_entries,
    resolve_payload_bundle_layout,
)
from dnadesign.cruncher.yiu.spec_models import YiuPayloadRenderingSpec
from dnadesign.cruncher.yiu.view_contracts import (
    build_assembled_payload_view_contract,
    build_payload_view_contract,
    build_split_payload_view_rows,
)


def publish_payload_bundle(
    *,
    spec: YiuPayloadRenderingSpec,
    normalized: NormalizedPayload,
    bundle_dir: Path,
) -> tuple[PayloadBundleManifest, PayloadVisualInventory]:
    layout = resolve_payload_bundle_layout(bundle_dir)

    payload_contract = build_payload_view_contract(normalized)
    split_payload_rows = build_split_payload_view_rows(normalized)
    assembled_payload_contract = build_assembled_payload_view_contract(normalized)
    normalized_payload_dump = build_normalized_payload_dump(spec=spec, normalized=normalized, layout=layout)

    view_entries = build_payload_view_entries(
        layout=layout,
        normalized=normalized,
    )

    inventory = build_payload_visual_inventory(
        spec=spec,
        normalized=normalized,
        layout=layout,
        view_entries=view_entries,
    )
    manifest = build_payload_bundle_manifest(normalized=normalized, inventory=inventory)

    write_payload_bundle_views(
        layout=layout,
        payload_contract=payload_contract,
        split_payload_rows=split_payload_rows,
        assembled_payload_contract=assembled_payload_contract,
    )
    write_normalized_payload_dump(layout=layout, normalized_payload_dump=normalized_payload_dump)
    if spec.output.emit_render_jobs_debug:
        write_debug_render_jobs(layout=layout, view_entries=view_entries)
    write_payload_bundle_state(layout=layout, manifest=manifest, inventory=inventory)
    return manifest, inventory
