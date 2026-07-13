"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/structure_views/backends/py3dmol.py

py3Dmol backend for browser structure views.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import html
import json

from dnadesign.thread.structure_views.models import (
    DNA_RESIDUE_NAMES,
    RNA_RESIDUE_NAMES,
    STANDARD_AMINO_ACID_RESIDUE_NAMES,
    MoleculeClass,
    StructureViewModel,
    StructureViewMoleculeStyle,
    StructureViewSelectionStyle,
    StructureViewSpec,
    filter_structure_text_by_molecule_classes,
    molecule_classes_in_structure_text,
)
from dnadesign.thread.structure_views.nucleic_geometry import (
    NucleicRibbonGeometry,
    extract_nucleic_ribbon_geometries,
)
from dnadesign.thread.structure_views.styles import (
    DNA_COLOR,
    NUCLEIC_ACID_RIBBON_THICKNESS,
    NUCLEIC_ACID_RIBBON_WIDTH,
    NUCLEIC_ACID_SPOKE_RADIUS,
    RNA_COLOR,
)

_NUCLEIC_ACID_CLASSES: frozenset[MoleculeClass] = frozenset({"dna", "rna"})
_DEFAULT_NUCLEIC_ACID_COLORS: dict[MoleculeClass, str] = {
    "dna": DNA_COLOR,
    "rna": RNA_COLOR,
}
_PY3DMOL_JS_URL = "https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"


def py3dmol_available() -> bool:
    """Return whether py3Dmol can be imported."""

    try:
        import py3Dmol  # noqa: F401
    except ImportError:
        return False
    return True


def render_py3dmol_structure_view(spec: StructureViewSpec) -> str:
    """Render a complete py3Dmol-backed HTML figure."""

    spec.validate()
    try:
        import py3Dmol
    except ImportError as exc:  # pragma: no cover - exercised when optional dependency is absent
        raise RuntimeError("py3Dmol is required for interactive browser structure views") from exc

    view = py3Dmol.view(width=int(spec.width), height=int(spec.height), js=_PY3DMOL_JS_URL)
    if hasattr(view, "setBackgroundColor"):
        view.setBackgroundColor(_py3dmol_background_color(spec.background_color))
    if spec.projection and hasattr(view, "setProjection"):
        view.setProjection(spec.projection)
    model_index_by_id = {}
    model_by_id = {}
    nucleic_geometry_by_model_class: dict[tuple[str, MoleculeClass], NucleicRibbonGeometry] = {}
    visible_molecule_classes = _visible_molecule_classes(spec)
    molecule_style_by_model_class = {
        (molecule_style.model_id, molecule_style.molecule_class): molecule_style
        for molecule_style in spec.molecule_styles
    }
    protein_surface_opacity_by_model_id = {
        molecule_style.model_id: molecule_style.opacity
        for molecule_style in spec.molecule_styles
        if molecule_style.molecule_class == "protein" and molecule_style.style == "surface"
    }
    view_style = "" if spec.hidden_molecule_classes else spec.view_style
    if view_style and hasattr(view, "setViewStyle"):
        view.setViewStyle({"style": view_style})
    for index, model in enumerate(spec.models):
        model_index_by_id[model.model_id] = index
        model_by_id[model.model_id] = model
        structure_text = filter_structure_text_by_molecule_classes(
            model.structure_text,
            structure_format=model.structure_format,
            visible_molecule_classes=visible_molecule_classes,
        )
        present_nucleic_classes = molecule_classes_in_structure_text(
            structure_text,
            structure_format=model.structure_format,
        )
        visible_nucleic_classes = tuple(
            molecule_class
            for molecule_class in ("dna", "rna")
            if molecule_class in visible_molecule_classes and molecule_class in present_nucleic_classes
        )
        model_nucleic_geometry = (
            extract_nucleic_ribbon_geometries(
                structure_text,
                structure_format=model.structure_format,
                molecule_classes=visible_nucleic_classes,
                source_label=model.model_id,
            )
            if visible_nucleic_classes
            else {}
        )
        nucleic_geometry_by_model_class.update(
            {(model.model_id, molecule_class): geometry for molecule_class, geometry in model_nucleic_geometry.items()}
        )
        view.addModel(structure_text, _py3dmol_model_format(model.structure_format))
        if "protein" in visible_molecule_classes:
            view.setStyle(_molecule_selection(index, "protein"), _style_for_model(spec, model))
        if (
            "dna" in visible_molecule_classes
            and (model.model_id, "dna") not in molecule_style_by_model_class
            and (model.model_id, "dna") in nucleic_geometry_by_model_class
        ):
            _apply_nucleic_ribbon_with_base_spokes(
                view,
                _molecule_selection(index, "dna"),
                geometry=nucleic_geometry_by_model_class[(model.model_id, "dna")],
                color=_DEFAULT_NUCLEIC_ACID_COLORS["dna"],
            )
        if (
            "rna" in visible_molecule_classes
            and (model.model_id, "rna") not in molecule_style_by_model_class
            and (model.model_id, "rna") in nucleic_geometry_by_model_class
        ):
            _apply_nucleic_ribbon_with_base_spokes(
                view,
                _molecule_selection(index, "rna"),
                geometry=nucleic_geometry_by_model_class[(model.model_id, "rna")],
                color=_DEFAULT_NUCLEIC_ACID_COLORS["rna"],
            )
    for molecule_style in spec.molecule_styles:
        if molecule_style.molecule_class not in visible_molecule_classes:
            continue
        molecule_selection = _molecule_selection(
            model_index_by_id[molecule_style.model_id], molecule_style.molecule_class
        )
        _apply_molecule_style(
            view,
            molecule_selection,
            molecule_style,
            geometry=nucleic_geometry_by_model_class.get((molecule_style.model_id, molecule_style.molecule_class)),
        )
    for index, model in enumerate(spec.models):
        if model.show_sidechains and "protein" in visible_molecule_classes:
            protein_style = molecule_style_by_model_class.get((model.model_id, "protein"))
            view.addStyle(_sidechain_selection(index), _style_for_sidechains(model, molecule_style=protein_style))
    for selection_style in spec.selection_styles:
        if selection_style.residue_scope not in visible_molecule_classes:
            continue
        _apply_selection_style(
            view,
            selection=_selection_query(
                model_index_by_id[selection_style.model_id],
                selection_style.residue_numbers,
                molecule_class=selection_style.residue_scope,
            ),
            spec=spec,
            selection_style=selection_style,
            model=model_by_id[selection_style.model_id],
            surface_highlight_opacity=protein_surface_opacity_by_model_id.get(selection_style.model_id),
            nucleic_geometry=nucleic_geometry_by_model_class.get(
                (selection_style.model_id, selection_style.residue_scope)
            ),
        )
    view.zoomTo()
    view.zoom(6.0)
    view.translate(0, 0)
    viewer_html = view._make_html()
    return _wrap_view_html(
        spec,
        viewer_html,
        scene_audit=_scene_audit(spec, nucleic_geometry_by_model_class),
    )


def _visible_molecule_classes(spec: StructureViewSpec) -> tuple[MoleculeClass, ...]:
    hidden = set(spec.hidden_molecule_classes)
    return tuple(molecule_class for molecule_class in ("protein", "dna", "rna") if molecule_class not in hidden)


def _style_for_model(spec: StructureViewSpec, model: StructureViewModel) -> dict[str, dict[str, object]]:
    style: dict[str, object] = {"color": model.color}
    if model.opacity < 1.0:
        style["opacity"] = float(model.opacity)
    return {spec.style: style}


def _py3dmol_model_format(structure_format: str) -> str:
    if structure_format == "mmcif":
        return "cif"
    return structure_format


def _py3dmol_background_color(color: str) -> str:
    """Return a background value that 3Dmol clears consistently."""

    normalized = str(color).strip()
    named_colors = {"#ffffff": "white", "#000000": "black"}
    if normalized.lower() in named_colors:
        return named_colors[normalized.lower()]
    return normalized


def _sidechain_selection(model_index: int) -> dict[str, object]:
    selection = _molecule_selection(model_index, "protein")
    selection["not"] = {"atom": ["N", "C", "O", "OXT"]}
    return selection


def _selection_query(
    model_index: int,
    residue_numbers: tuple[int, ...],
    *,
    molecule_class: MoleculeClass,
) -> dict[str, object]:
    selection = {"model": model_index, "resi": list(residue_numbers)}
    selection.update(_molecule_scope(molecule_class))
    return selection


def _molecule_selection(model_index: int, molecule_class: MoleculeClass) -> dict[str, object]:
    selection = {"model": model_index}
    selection.update(_molecule_scope(molecule_class))
    return selection


def _molecule_scope(molecule_class: MoleculeClass) -> dict[str, object]:
    if molecule_class == "protein":
        return {"resn": sorted(STANDARD_AMINO_ACID_RESIDUE_NAMES)}
    if molecule_class == "dna":
        return {"resn": sorted(DNA_RESIDUE_NAMES)}
    if molecule_class == "rna":
        return {"resn": sorted(RNA_RESIDUE_NAMES)}
    raise ValueError(f"Unsupported molecule class: {molecule_class}")


def _style_for_sidechains(
    model: StructureViewModel,
    *,
    molecule_style: StructureViewMoleculeStyle | None = None,
) -> dict[str, dict[str, object]]:
    color = molecule_style.color if molecule_style is not None else model.sidechain_color or model.color
    opacity = molecule_style.opacity if molecule_style is not None else 1.0
    return {"stick": _stick_payload(color=color, opacity=opacity, radius=float(model.sidechain_radius))}


def _apply_molecule_style(
    view: object,
    selection: dict[str, object],
    molecule_style: StructureViewMoleculeStyle,
    geometry: NucleicRibbonGeometry | None,
) -> None:
    if molecule_style.style == "surface":
        view.addSurface("VDW", _style_payload(color=molecule_style.color, opacity=molecule_style.opacity), selection)
        return
    if molecule_style.molecule_class in _NUCLEIC_ACID_CLASSES and molecule_style.style in {
        "",
        "backbone_ribbon_with_base_spokes",
    }:
        if geometry is None:
            view.setStyle(selection, {})
            return
        _apply_nucleic_ribbon_with_base_spokes(
            view,
            selection,
            geometry=geometry,
            color=molecule_style.color,
            opacity=molecule_style.opacity,
            ribbon_width=float(molecule_style.width),
            ribbon_thickness=float(molecule_style.thickness),
        )
        return
    view.setStyle(selection, _style_for_molecule_style(molecule_style))


def _apply_nucleic_ribbon_with_base_spokes(
    view: object,
    selection: dict[str, object],
    *,
    geometry: NucleicRibbonGeometry,
    color: str,
    opacity: float = 1.0,
    ribbon_width: float = NUCLEIC_ACID_RIBBON_WIDTH,
    ribbon_thickness: float = NUCLEIC_ACID_RIBBON_THICKNESS,
    spoke_radius: float = NUCLEIC_ACID_SPOKE_RADIUS,
) -> None:
    """Render an ordered C4-prime ribbon with one attached base spoke."""

    view.setStyle(selection, {})
    for mesh in geometry.ribbon_meshes(width=ribbon_width, thickness=ribbon_thickness):
        view.addCustom(
            {
                "vertexArr": [_point_payload(point) for point in mesh.vertices],
                "faceArr": list(mesh.faces),
                "color": color,
                "opacity": float(opacity),
            }
        )
    for residue in geometry.residues:
        view.addCylinder(
            {
                "start": _point_payload(residue.backbone_anchor),
                "end": _point_payload(residue.base_centroid),
                "radius": float(spoke_radius),
                "fromCap": 1,
                "toCap": 1,
                "color": color,
                "opacity": float(opacity),
            }
        )


def _point_payload(point: tuple[float, float, float]) -> dict[str, float]:
    return {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}


def _style_for_molecule_style(molecule_style: StructureViewMoleculeStyle) -> dict[str, dict[str, object]]:
    if molecule_style.style:
        if molecule_style.style == "stick":
            return {
                "stick": _stick_payload(
                    color=molecule_style.color,
                    opacity=molecule_style.opacity,
                    radius=float(molecule_style.radius),
                )
            }
        return {
            molecule_style.style: _style_payload(
                color=molecule_style.color,
                opacity=molecule_style.opacity,
            )
        }
    return {"cartoon": _style_payload(color=molecule_style.color, opacity=molecule_style.opacity)}


def _style_for_selection(
    spec: StructureViewSpec,
    selection_style: StructureViewSelectionStyle,
    *,
    model: StructureViewModel,
) -> dict[str, dict[str, object]]:
    style = _style_payload(color=selection_style.color, opacity=selection_style.opacity)
    styles = {spec.style: style}
    if selection_style.residue_scope == "protein" and selection_style.show_sidechains:
        styles["stick"] = _stick_payload(
            color=selection_style.color,
            opacity=selection_style.opacity,
            radius=max(0.22, float(model.sidechain_radius)),
        )
    return styles


def _apply_selection_style(
    view: object,
    *,
    selection: dict[str, object],
    spec: StructureViewSpec,
    selection_style: StructureViewSelectionStyle,
    model: StructureViewModel,
    surface_highlight_opacity: float | None,
    nucleic_geometry: NucleicRibbonGeometry | None,
) -> None:
    if selection_style.residue_scope in _NUCLEIC_ACID_CLASSES:
        if nucleic_geometry is None:
            raise ValueError(f"Missing nucleic ribbon geometry for selection {selection_style.selection_id}")
        _apply_nucleic_ribbon_with_base_spokes(
            view,
            selection,
            geometry=nucleic_geometry.filtered(selection_style.residue_numbers),
            color=selection_style.color,
            opacity=selection_style.opacity,
            ribbon_width=NUCLEIC_ACID_RIBBON_WIDTH * 1.08,
            ribbon_thickness=NUCLEIC_ACID_RIBBON_THICKNESS * 1.20,
            spoke_radius=NUCLEIC_ACID_SPOKE_RADIUS * 1.25,
        )
        return
    if surface_highlight_opacity is not None:
        view.addSurface(
            "VDW",
            _style_payload(color=selection_style.color, opacity=surface_highlight_opacity),
            selection,
        )
    view.addStyle(
        selection,
        _style_for_selection(
            spec,
            selection_style,
            model=model,
        ),
    )


def _style_payload(*, color: str, opacity: float) -> dict[str, object]:
    style: dict[str, object] = {"color": color}
    if opacity < 1.0:
        style["opacity"] = float(opacity)
    return style


def _stick_payload(*, color: str, opacity: float, radius: float) -> dict[str, object]:
    style = _style_payload(color=color, opacity=opacity)
    style["radius"] = float(radius)
    return style


def _wrap_view_html(spec: StructureViewSpec, viewer_html: str, *, scene_audit: dict[str, object]) -> str:
    title = html.escape(spec.title)
    subtitle = html.escape(spec.subtitle)
    description = html.escape(spec.description)
    interpretation_limit = html.escape(spec.interpretation_limit)
    view_id = _view_id(spec)
    subtitle_html = ""
    if subtitle:
        subtitle_html = (
            '<div style="font-size:0.86rem; line-height:1.25; margin:0.02rem auto 0.18rem auto; '
            'color:#57606a; max-width:58rem; text-align:center;">'
            f"{subtitle}"
            "</div>"
        )
    description_ids: list[str] = []
    metadata_html = ""
    if description:
        description_id = f"{view_id}-description"
        description_ids.append(description_id)
        metadata_html += f'<span id="{description_id}" class="structure-view-sr-only">{description}</span>'
    if interpretation_limit:
        interpretation_limit_id = f"{view_id}-interpretation-limit"
        description_ids.append(interpretation_limit_id)
        metadata_html += (
            f'<span id="{interpretation_limit_id}" class="structure-view-sr-only">{interpretation_limit}</span>'
        )
    legend = "".join(_legend_item(model) for model in spec.models)
    visible_molecule_classes = set(_visible_molecule_classes(spec))
    legend += "".join(
        _molecule_legend_item(molecule_style)
        for molecule_style in spec.molecule_styles
        if molecule_style.molecule_class in visible_molecule_classes
    )
    legend += "".join(
        _selection_legend_item(selection_style)
        for selection_style in spec.selection_styles
        if selection_style.residue_scope in visible_molecule_classes
    )
    srcdoc = html.escape(_viewer_document(spec, viewer_html, scene_audit=scene_audit), quote=True)
    iframe_title = html.escape(f"Interactive structure view: {spec.title}", quote=True)
    described_by = f' aria-describedby="{" ".join(description_ids)}"' if description_ids else ""
    return f"""
    <figure style="margin:0; width:100%; max-width:100%; min-width:0;">
      <style>
        .structure-view-sr-only {{
          position:absolute;
          width:1px;
          height:1px;
          padding:0;
          margin:-1px;
          overflow:hidden;
          clip:rect(0, 0, 0, 0);
          clip-path:inset(50%);
          white-space:nowrap;
          border:0;
        }}
      </style>
      <div style="background:#ffffff; width:100%; box-sizing:border-box;">
        <div style="font-weight:650; font-size:0.96rem; margin:0 auto 0.02rem auto;
                    color:#24292f; text-align:center; line-height:1.18;">{title}</div>
        {subtitle_html}
        {metadata_html}
        <iframe title="{iframe_title}"{described_by} srcdoc="{srcdoc}"
                sandbox="allow-scripts allow-same-origin"
                referrerpolicy="no-referrer"
                style="display:block; width:100%; height:{int(spec.height)}px;
                       border:0; background:#ffffff;"></iframe>
        <div style="display:flex; flex-wrap:wrap; gap:0.8rem; align-items:center;
                    justify-content:center; margin-top:0.45rem; font-size:0.86rem; color:#57606a;">
          {legend}
        </div>
      </div>
    </figure>
    """


def _view_id(spec: StructureViewSpec) -> str:
    digest = hashlib.sha256(f"{spec.title}\n{spec.description}".encode()).hexdigest()[:12]
    return f"structure-view-{digest}"


def _viewer_document(spec: StructureViewSpec, viewer_html: str, *, scene_audit: dict[str, object]) -> str:
    audit_json = json.dumps(scene_audit, sort_keys=True).replace("</", "<\\/")
    return f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          html, body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            background: #ffffff;
            width: 100%;
            height: 100%;
          }}
          [id^="3dmolviewer_"] {{
            width: 100% !important;
            height: 100% !important;
          }}
          canvas {{
            display: block !important;
          }}
        </style>
      </head>
      <body>
        {viewer_html}
        <script id="dnadesign-structure-scene-audit" type="application/json">{audit_json}</script>
        <script>
          window.__dnadesignStructureSceneAudit = JSON.parse(
            document.getElementById('dnadesign-structure-scene-audit').textContent
          );
        </script>
        {_viewer_interaction_script(spec.camera_memory_key)}
      </body>
    </html>
    """


def _scene_audit(
    spec: StructureViewSpec,
    geometry_by_model_class: dict[tuple[str, MoleculeClass], NucleicRibbonGeometry],
) -> dict[str, object]:
    style_by_model_class = {
        (style.model_id, style.molecule_class): style
        for style in spec.molecule_styles
        if style.molecule_class in _NUCLEIC_ACID_CLASSES
    }
    geometry_rows = [
        geometry.audit_row(
            model_id=model_id,
            ribbon_width=float(style_by_model_class[(model_id, molecule_class)].width)
            if (model_id, molecule_class) in style_by_model_class
            else NUCLEIC_ACID_RIBBON_WIDTH,
            ribbon_thickness=float(style_by_model_class[(model_id, molecule_class)].thickness)
            if (model_id, molecule_class) in style_by_model_class
            else NUCLEIC_ACID_RIBBON_THICKNESS,
        )
        for (model_id, molecule_class), geometry in sorted(geometry_by_model_class.items())
        if geometry.residues
    ]
    return {
        "schema_id": "dnadesign_py3dmol_scene_audit_v1",
        "representation": "backbone_ribbon_with_base_spokes",
        "surface_scope": "protein_only",
        "hidden_molecule_classes": list(spec.hidden_molecule_classes),
        "nucleic_geometry": geometry_rows,
    }


def _viewer_interaction_script(camera_memory_key: str) -> str:
    js_key = json.dumps(camera_memory_key.strip())
    return f"""
        <script>
        (function() {{
          const memoryKey = {js_key};
          function currentViewer() {{
            const container = document.querySelector('[id^="3dmolviewer_"]');
            if (!container) {{
              return null;
            }}
            const suffix = container.id.replace('3dmolviewer_', '');
            return window['viewer_' + suffix] || null;
          }}
          function applyCameraMemory() {{
            const viewer = currentViewer();
            const container = document.querySelector('[id^="3dmolviewer_"]');
            if (!viewer || !container) {{
              return;
            }}
            try {{
              const storedView = memoryKey ? window.localStorage.getItem(memoryKey) : "";
              if (storedView && typeof viewer.setView === 'function') {{
                const parsedView = JSON.parse(storedView);
                if (Array.isArray(parsedView)) {{
                  viewer.setView(parsedView);
                  viewer.render();
                }}
              }}
            }} catch (_error) {{}}
            const saveView = function() {{
              try {{
                if (memoryKey && typeof viewer.getView === 'function') {{
                  window.localStorage.setItem(memoryKey, JSON.stringify(viewer.getView()));
                }}
              }} catch (_error) {{}}
            }};
            if (container.dataset.twoFingerPan !== 'enabled') {{
              container.dataset.twoFingerPan = 'enabled';
              const panTargets = [];
              const panListenerOptions = {{passive: false, capture: true}};
              const panOnWheel = function(event) {{
                const translateScene = typeof viewer.translateScene === 'function'
                  ? viewer.translateScene.bind(viewer)
                  : null;
                const translate = typeof viewer.translate === 'function' ? viewer.translate.bind(viewer) : null;
                const pan = translateScene || translate;
                if (event.ctrlKey || !pan) {{
                  return;
                }}
                event.preventDefault();
                event.stopPropagation();
                const panScale = 0.22;
                pan(-event.deltaX * panScale, -event.deltaY * panScale);
                viewer.render();
                window.setTimeout(saveView, 80);
              }};
              const registerPanTarget = function(target) {{
                if (!target || panTargets.indexOf(target) >= 0) {{
                  return;
                }}
                target.addEventListener('wheel', panOnWheel, panListenerOptions);
                panTargets.push(target);
              }};
              const canvas = container.querySelector('canvas');
              registerPanTarget(container);
              registerPanTarget(canvas);
              registerPanTarget(document);
            }}
            ['mouseup', 'touchend', 'wheel'].forEach(function(eventName) {{
              container.addEventListener(eventName, function() {{
                window.setTimeout(saveView, 80);
              }}, {{passive: true}});
            }});
            window.setTimeout(saveView, 250);
          }}
          if (window.$3Dmolpromise && typeof window.$3Dmolpromise.then === 'function') {{
            window.$3Dmolpromise.then(function() {{
              window.setTimeout(applyCameraMemory, 0);
            }});
          }} else {{
            window.setTimeout(applyCameraMemory, 250);
          }}
        }})();
        </script>
    """


def _legend_item(model: StructureViewModel) -> str:
    label = html.escape(model.label or model.model_id)
    color = html.escape(model.color)
    return (
        f'<span><span style="display:inline-block; width:0.72rem; height:0.72rem; '
        f"background:{color}; border:1px solid #57606a; vertical-align:-0.08rem; "
        f'margin-right:0.25rem;"></span>{label}</span>'
    )


def _molecule_legend_item(molecule_style: StructureViewMoleculeStyle) -> str:
    label = html.escape(molecule_style.label)
    color = html.escape(molecule_style.color)
    molecule_class = html.escape(molecule_style.molecule_class)
    return (
        f'<span data-molecule-class="{molecule_class}"><span style="display:inline-block; width:0.72rem; '
        f"height:0.72rem; background:{color}; border:1px solid #57606a; vertical-align:-0.08rem; "
        f'margin-right:0.25rem;"></span>{label}</span>'
    )


def _selection_legend_item(selection_style: StructureViewSelectionStyle) -> str:
    label = html.escape(selection_style.label)
    color = html.escape(selection_style.color)
    selection_id = html.escape(selection_style.selection_id)
    return (
        f'<span data-selection-id="{selection_id}"><span style="display:inline-block; width:0.72rem; height:0.72rem; '
        f"background:{color}; border:1px solid #57606a; vertical-align:-0.08rem; "
        f'margin-right:0.25rem;"></span>{label}</span>'
    )
