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
)


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

    view = py3Dmol.view(width=int(spec.width), height=int(spec.height))
    if hasattr(view, "setBackgroundColor"):
        view.setBackgroundColor(spec.background_color)
    if spec.projection and hasattr(view, "setProjection"):
        view.setProjection(spec.projection)
    if spec.view_style and hasattr(view, "setViewStyle"):
        view.setViewStyle({"style": spec.view_style})
    model_index_by_id = {}
    for index, model in enumerate(spec.models):
        model_index_by_id[model.model_id] = index
        view.addModel(model.structure_text, _py3dmol_model_format(model.structure_format))
        view.setStyle({"model": index}, _style_for_model(spec, model))
    for molecule_style in spec.molecule_styles:
        view.addStyle(
            _molecule_selection(model_index_by_id[molecule_style.model_id], molecule_style.molecule_class),
            _style_for_molecule_style(molecule_style),
        )
    for selection_style in spec.selection_styles:
        view.addStyle(
            _selection_query(
                model_index_by_id[selection_style.model_id],
                selection_style.residue_numbers,
                molecule_class=selection_style.residue_scope,
            ),
            _style_for_selection(spec, selection_style),
        )
    for index, model in enumerate(spec.models):
        if model.show_sidechains:
            view.addStyle(_sidechain_selection(index), _style_for_sidechains(model))
    view.zoomTo()
    view.zoom(1.35)
    view.translate(0, 0)
    viewer_html = view._make_html()
    return _wrap_view_html(spec, viewer_html)


def _style_for_model(spec: StructureViewSpec, model: StructureViewModel) -> dict[str, dict[str, object]]:
    style: dict[str, object] = {"color": model.color}
    if model.opacity < 1.0:
        style["opacity"] = float(model.opacity)
    return {spec.style: style}


def _py3dmol_model_format(structure_format: str) -> str:
    if structure_format == "mmcif":
        return "cif"
    return structure_format


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


def _style_for_sidechains(model: StructureViewModel) -> dict[str, dict[str, object]]:
    return {
        "stick": {
            "color": model.sidechain_color or model.color,
            "radius": float(model.sidechain_radius),
        }
    }


def _style_for_molecule_style(molecule_style: StructureViewMoleculeStyle) -> dict[str, dict[str, object]]:
    style_name = molecule_style.style or ("cartoon" if molecule_style.molecule_class == "protein" else "stick")
    style: dict[str, object] = {"color": molecule_style.color}
    if style_name == "stick":
        style["radius"] = float(molecule_style.radius)
    if molecule_style.opacity < 1.0:
        style["opacity"] = float(molecule_style.opacity)
    return {style_name: style}


def _style_for_selection(
    spec: StructureViewSpec,
    selection_style: StructureViewSelectionStyle,
) -> dict[str, dict[str, object]]:
    style: dict[str, object] = {"color": selection_style.color}
    if selection_style.opacity < 1.0:
        style["opacity"] = float(selection_style.opacity)
    return {spec.style: style}


def _wrap_view_html(spec: StructureViewSpec, viewer_html: str) -> str:
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
    legend += "".join(_molecule_legend_item(molecule_style) for molecule_style in spec.molecule_styles)
    legend += "".join(_selection_legend_item(selection_style) for selection_style in spec.selection_styles)
    srcdoc = html.escape(_viewer_document(spec, viewer_html), quote=True)
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


def _viewer_document(spec: StructureViewSpec, viewer_html: str) -> str:
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
        {_viewer_interaction_script(spec.camera_memory_key)}
      </body>
    </html>
    """


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
                const panScale = 0.45;
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
