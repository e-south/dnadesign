"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook_structure_browser.py

Interactive structure-browser helpers for the Eco1 review notebook.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.pdb_alignment import (
    align_pdb_text_to_reference_ca,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.notebook_runtime import (
    resolve_manifest_path,
)
from dnadesign.thread.structure_views import (
    NUCLEIC_ACID_RIBBON_THICKNESS,
    NUCLEIC_ACID_RIBBON_WIDTH,
    PROTEIN_SURFACE_OPACITY,
    StructureViewModel,
    StructureViewMoleculeStyle,
    StructureViewSelectionStyle,
    StructureViewSpec,
    filter_structure_text_by_molecule_classes,
    molecule_classes_in_structure_text,
    render_structure_view_html,
    summarize_structure_atom_content,
)

from . import notebook_structure_rows as structure_rows
from .notebook_structure_dashboard import (
    structure_browser_panel_html,
    structure_metric_rows,
    structure_summary_rows,
)
from .structure_browser_common import (
    DNA_CLASS_COLOR,
    REFERENCE_COLOR,
    RESIDUE_CATEGORY_HIGHLIGHT_COLOR,
    RNA_CLASS_COLOR,
)

_STRUCTURE_VIEW_WIDTH = 760
_STRUCTURE_VIEW_HEIGHT = 520
_MOLECULE_CLASS_COLORS = {
    "dna": DNA_CLASS_COLOR,
    "rna": RNA_CLASS_COLOR,
}

load_structure_browser_rows = structure_rows.load_structure_browser_rows
load_structure_highlight_rows = structure_rows.load_structure_highlight_rows
structure_browser_lookup = structure_rows.structure_browser_lookup
structure_group_lookup = structure_rows.structure_group_lookup
structure_highlight_lookup = structure_rows.structure_highlight_lookup


def render_structure_browser(
    *,
    mo: Any,
    selected_row: dict[str, Any] | None,
    structure_ui: Any,
    structure_group_ui: Any | None = None,
    structure_highlight_ui: Any | None = None,
    selected_highlight_row: dict[str, Any] | None = None,
    structure_background_ui: Any | None = None,
    structure_mutation_ui: Any | None = None,
    structure_sidechain_ui: Any | None = None,
    structure_surface_ui: Any | None = None,
    structure_dna_visible_ui: Any | None = None,
    structure_rna_visible_ui: Any | None = None,
    show_reference_background: bool = True,
    show_mutation_differences: bool = False,
    show_sidechains: bool = True,
    show_protein_surface: bool = False,
    show_dna: bool = True,
    show_rna: bool = True,
    highlight_dna: bool = False,
    highlight_rna: bool = False,
) -> Any:
    """Render an interactive browser structure view for one selected fold model."""

    if selected_row is None:
        return mo.md("Interactive structure browsing is unavailable for this section.")
    browser_root = Path(str(selected_row["_browser_root"]))
    reference = dict(selected_row.get("_reference") or {})
    reference_path = resolve_manifest_path(browser_root, str(reference.get("local_path") or ""))
    query_path = resolve_manifest_path(browser_root, str(selected_row.get("local_path") or ""))
    if not reference_path.exists() or not query_path.exists():
        return mo.md("Interactive structure browsing is skipped because a selected PDB path is missing.")
    reference_format = str(reference.get("structure_format") or "pdb")
    query_format = str(selected_row.get("structure_format") or "pdb")
    reference_text = _read_structure_text(reference_path)
    query_text = _read_structure_text(query_path)
    reference_atom_content = summarize_structure_atom_content(reference_text, structure_format=reference_format)
    alignment_status = "raw_coordinates"
    browser_mapped_ca_rmsd: float | None = None
    alignment = dict(selected_row.get("_alignment") or {})
    view_mode = str(selected_row.get("structure_view_mode") or "")
    if view_mode == "reference_selection":
        query_text = ""
        alignment_status = "reference_selection"
    elif str(alignment.get("status") or "") == "enabled":
        alignment_reference_path = resolve_manifest_path(
            browser_root,
            str(alignment.get("reference_local_path") or reference.get("local_path") or ""),
        )
        if not alignment_reference_path.exists():
            return mo.md("Interactive structure alignment is skipped because the alignment reference PDB is missing.")
        alignment_reference_text = _read_structure_text(alignment_reference_path)
        try:
            query_text, browser_mapped_ca_rmsd = align_pdb_text_to_reference_ca(
                query_text=query_text,
                reference_text=alignment_reference_text,
                query_start_residue=int(alignment.get("query_start_residue", 3)),
                reference_start_residue=int(alignment.get("reference_start_residue", 1)),
                residue_count=int(alignment.get("residue_count", 309)),
            )
            alignment_status = "aligned_in_memory_to_reference_ca"
        except Exception as exc:  # pragma: no cover - defensive notebook rendering path
            return mo.md(f"Interactive structure alignment failed: `{type(exc).__name__}: {exc}`")
    query_atom_content = (
        None if not query_text else summarize_structure_atom_content(query_text, structure_format=query_format)
    )
    query_model_id = str(selected_row.get("source_candidate_id") or selected_row["candidate_id"])
    reference_model_id = str(reference.get("model_id") or "reference")
    selection_styles = _selection_styles(selected_row, show_sidechains=show_sidechains)
    selected_highlight_selection_styles = ()
    if selected_highlight_row is not None and selected_highlight_row is not selected_row:
        selected_highlight_selection_styles = _selection_styles(
            selected_highlight_row,
            show_sidechains=show_sidechains,
            model_id_override=query_model_id if view_mode != "reference_selection" else reference_model_id,
        )
    mutation_selection_styles = ()
    if show_mutation_differences and view_mode != "reference_selection":
        mutation_selection_styles = _mutation_selection_styles(
            selected_row,
            model_id=query_model_id,
            show_sidechains=show_sidechains,
        )
    reference_model = StructureViewModel(
        model_id=reference_model_id,
        structure_text=reference_text,
        structure_format=reference_format,
        label=str(reference.get("display_label") or "Reference"),
        color=str(reference.get("color") or "#d8d8d8"),
        opacity=0.82,
        show_sidechains=False,
    )
    models = []
    if view_mode == "reference_selection" or show_reference_background:
        models.append(reference_model)
    elif show_dna or show_rna:
        visible_nucleic_classes = tuple(
            molecule_class for molecule_class, is_visible in (("dna", show_dna), ("rna", show_rna)) if is_visible
        )
        models.append(
            StructureViewModel(
                model_id=reference_model_id,
                structure_text=filter_structure_text_by_molecule_classes(
                    reference_text,
                    structure_format=reference_format,
                    visible_molecule_classes=visible_nucleic_classes,
                ),
                structure_format=reference_format,
                label="Reference nucleic acids",
                color=str(reference.get("color") or "#d8d8d8"),
                opacity=0.82,
                show_sidechains=False,
            )
        )
    if view_mode != "reference_selection":
        models.append(
            StructureViewModel(
                model_id=query_model_id,
                structure_text=query_text,
                structure_format=query_format,
                label=str(selected_row.get("display_label") or selected_row["candidate_id"]),
                color=REFERENCE_COLOR,
                show_sidechains=False,
            )
        )
    selection_styles += selected_highlight_selection_styles + mutation_selection_styles
    try:
        html_panel = render_structure_view_html(
            StructureViewSpec(
                title=_structure_browser_title(selected_row),
                subtitle=_structure_browser_subtitle(selected_row),
                description=_structure_browser_description(selected_row),
                interpretation_limit=_structure_browser_interpretation_limit(selected_row),
                models=tuple(models),
                molecule_styles=_molecule_styles(
                    models,
                    row=selected_row,
                    show_protein_surface=show_protein_surface,
                    highlight_dna=highlight_dna,
                    highlight_rna=highlight_rna,
                ),
                selection_styles=selection_styles,
                hidden_molecule_classes=_hidden_molecule_classes(show_dna=show_dna, show_rna=show_rna),
                width=_STRUCTURE_VIEW_WIDTH,
                height=_STRUCTURE_VIEW_HEIGHT,
                camera_memory_key=_camera_memory_key(selected_row),
            )
        )
    except Exception as exc:  # pragma: no cover - defensive notebook rendering path
        return mo.md(f"Interactive structure viewer failed to render: `{type(exc).__name__}: {exc}`")
    metric_rows = structure_metric_rows(
        selected_row,
        selected_highlight_row=selected_highlight_row,
        alignment_status=alignment_status,
        browser_mapped_ca_rmsd=browser_mapped_ca_rmsd,
        reference_atom_content=reference_atom_content,
        query_atom_content=query_atom_content,
        show_sidechains=show_sidechains,
    )
    summary_rows = structure_summary_rows(
        selected_row,
        selected_highlight_row=selected_highlight_row,
        alignment_status=alignment_status,
        browser_mapped_ca_rmsd=browser_mapped_ca_rmsd,
        reference_atom_content=reference_atom_content,
        query_atom_content=query_atom_content,
        show_sidechains=show_sidechains,
        show_dna=show_dna,
        show_rna=show_rna,
        highlight_dna=highlight_dna,
        highlight_rna=highlight_rna,
    )
    selector_controls = [
        item
        for item in (
            structure_group_ui,
            structure_ui,
            structure_highlight_ui if view_mode != "reference_selection" else None,
        )
        if item is not None
    ]
    display_controls = [
        item
        for item in (
            structure_background_ui if view_mode != "reference_selection" else None,
            structure_mutation_ui if view_mode != "reference_selection" else None,
            structure_sidechain_ui,
            structure_surface_ui if has_declared_protein_surface(selected_row) else None,
            structure_dna_visible_ui,
            structure_rna_visible_ui,
        )
        if item is not None
    ]
    control_rows = [*selector_controls]
    if display_controls:
        control_rows.append(
            mo.hstack(
                display_controls,
                justify="start",
                align="start",
                wrap=True,
                gap=1.0,
            )
        )
    return mo.vstack(
        [
            mo.vstack(control_rows, gap=0.3),
            mo.Html(structure_browser_panel_html(html_panel, summary_rows, selected_row)),
            mo.accordion(
                {"Selected structure evidence": mo.ui.table(metric_rows, page_size=8)},
                multiple=False,
                lazy=True,
            ),
        ],
        gap=0.35,
    )


def _read_structure_text(path: Path) -> str:
    """Read structure text with an mtime-keyed notebook cache."""

    stat = path.stat()
    return _read_structure_text_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=128)
def _read_structure_text_cached(path: str, mtime_ns: int, size: int) -> str:
    del mtime_ns, size
    return Path(path).read_text(encoding="utf-8")


def _structure_browser_title(row: dict[str, Any]) -> str:
    label = str(row.get("display_label") or row.get("candidate_id") or "")
    return label


def _structure_browser_subtitle(row: dict[str, Any]) -> str:
    if str(row.get("structure_view_mode") or "") == "reference_selection":
        residue_count = row.get("selection_residue_count")
        group = str(row.get("group") or "Reference selection")
        if residue_count is not None:
            return f"{group} | {int(residue_count)} highlighted residues"
        return group
    return str(row.get("group") or "Candidate fold overlay")


def _structure_browser_description(row: dict[str, Any]) -> str:
    row_description = str(row.get("description") or "").strip()
    if row_description:
        return row_description
    return str(row.get("_deliverable_description") or "").strip()


def _structure_browser_interpretation_limit(row: dict[str, Any]) -> str:
    return str(row.get("_interpretation_limit") or "").strip()


def _camera_memory_key(row: dict[str, Any]) -> str:
    if str(row.get("structure_view_mode") or "") == "reference_selection":
        return "eco1-rt-repack:reference-complex:camera-v5"
    return "eco1-rt-repack:candidate-folds:camera-v5"


def _molecule_styles(
    models: list[StructureViewModel],
    *,
    row: dict[str, Any],
    show_protein_surface: bool,
    highlight_dna: bool,
    highlight_rna: bool,
) -> tuple[StructureViewMoleculeStyle, ...]:
    enabled = {
        "dna": highlight_dna,
        "rna": highlight_rna,
    }
    labels = {
        "dna": "DNA",
        "rna": "RNA",
    }
    styles = _declared_molecule_styles(
        row,
        model_ids={model.model_id for model in models},
        show_protein_surface=show_protein_surface,
    )
    declared_model_classes = {(style.model_id, style.molecule_class) for style in styles}
    for model in models:
        present_molecule_classes = molecule_classes_in_structure_text(
            model.structure_text,
            structure_format=model.structure_format,
        )
        for molecule_class, is_enabled in enabled.items():
            if (
                not is_enabled
                or molecule_class not in present_molecule_classes
                or (model.model_id, molecule_class) in declared_model_classes
            ):
                continue
            styles.append(
                StructureViewMoleculeStyle(
                    molecule_class=molecule_class,  # type: ignore[arg-type]
                    model_id=model.model_id,
                    label=labels[molecule_class],
                    color=_MOLECULE_CLASS_COLORS[molecule_class],
                )
            )
    return tuple(styles)


def _declared_molecule_styles(
    row: dict[str, Any],
    *,
    model_ids: set[str],
    show_protein_surface: bool,
) -> list[StructureViewMoleculeStyle]:
    styles: list[StructureViewMoleculeStyle] = []
    for item in row.get("molecule_styles") or []:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("model_id") or "")
        if model_id not in model_ids:
            continue
        if (
            not show_protein_surface
            and str(item.get("molecule_class") or "") == "protein"
            and str(item.get("style") or "") == "surface"
        ):
            continue
        molecule_class = str(item.get("molecule_class") or "protein")
        style = str(item.get("style") or "")
        color = str(item.get("color") or REFERENCE_COLOR)
        opacity = float(item.get("opacity", 1.0))
        if molecule_class == "protein" and style == "surface" and opacity != PROTEIN_SURFACE_OPACITY:
            raise ValueError(
                f"Eco1 protein surfaces must use alpha {PROTEIN_SURFACE_OPACITY:.2f}; observed {opacity:.2f}"
            )
        expected_nucleic_color = {"dna": DNA_CLASS_COLOR, "rna": RNA_CLASS_COLOR}.get(molecule_class)
        if expected_nucleic_color is not None and color != expected_nucleic_color:
            raise ValueError(
                f"Eco1 {molecule_class.upper()} representations must use {expected_nucleic_color}; observed {color}"
            )
        styles.append(
            StructureViewMoleculeStyle(
                molecule_class=molecule_class,  # type: ignore[arg-type]
                model_id=model_id,
                label=str(item.get("label") or "Molecule"),
                color=color,
                opacity=opacity,
                style=style,  # type: ignore[arg-type]
                radius=float(item.get("radius", 0.18)),
                width=float(item.get("width", NUCLEIC_ACID_RIBBON_WIDTH)),
                thickness=float(item.get("thickness", NUCLEIC_ACID_RIBBON_THICKNESS)),
            )
        )
    return styles


def has_declared_protein_surface(row: dict[str, Any] | None) -> bool:
    """Return whether a structure scene declares a toggleable protein surface."""

    if row is None:
        return False
    return any(
        isinstance(item, dict)
        and str(item.get("molecule_class") or "") == "protein"
        and str(item.get("style") or "") == "surface"
        for item in row.get("molecule_styles") or []
    )


def _hidden_molecule_classes(*, show_dna: bool, show_rna: bool) -> tuple[str, ...]:
    hidden: list[str] = []
    if not show_dna:
        hidden.append("dna")
    if not show_rna:
        hidden.append("rna")
    return tuple(hidden)


def _selection_styles(
    row: dict[str, Any],
    *,
    show_sidechains: bool,
    model_id_override: str = "",
) -> tuple[StructureViewSelectionStyle, ...]:
    styles: list[StructureViewSelectionStyle] = []
    for item in row.get("selection_styles") or []:
        if not isinstance(item, dict):
            continue
        styles.append(
            StructureViewSelectionStyle(
                selection_id=str(item.get("selection_id") or ""),
                model_id=model_id_override or str(item.get("model_id") or ""),
                label=str(item.get("label") or ""),
                residue_numbers=tuple(int(value) for value in item.get("residue_numbers") or []),
                color=str(item.get("color") or RESIDUE_CATEGORY_HIGHLIGHT_COLOR),
                opacity=float(item.get("opacity", 1.0)),
                residue_scope=str(item.get("residue_scope") or "protein"),  # type: ignore[arg-type]
                show_sidechains=show_sidechains,
            )
        )
    return tuple(styles)


def _mutation_selection_styles(
    row: dict[str, Any],
    *,
    model_id: str,
    show_sidechains: bool,
) -> tuple[StructureViewSelectionStyle, ...]:
    residue_numbers = tuple(int(value) for value in row.get("mutation_residue_numbers") or [])
    if not residue_numbers:
        return ()
    return (
        StructureViewSelectionStyle(
            selection_id="candidate_differences",
            model_id=model_id,
            label="Candidate differences",
            residue_numbers=residue_numbers,
            color=RESIDUE_CATEGORY_HIGHLIGHT_COLOR,
            show_sidechains=show_sidechains,
        ),
    )
