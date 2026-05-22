from __future__ import annotations

from .data_cells import DATA_CELLS
from .layout_cells import LAYOUT_CELLS
from .plot_cells import PLOT_CONFIG_CELLS
from .record_cells import RECORD_CELLS
from .run_cells import RUN_CELLS
from .setup_cells import SETUP_CELLS
from .summary_cells import SUMMARY_CELLS

NOTEBOOK_TEMPLATE_FRAGMENTS = (
    SETUP_CELLS,
    SUMMARY_CELLS,
    RUN_CELLS,
    RECORD_CELLS,
    PLOT_CONFIG_CELLS,
    "__PLOT_GALLERY_CELLS__",
    DATA_CELLS,
    LAYOUT_CELLS,
)


def render_notebook_source(*, plot_gallery_cells: str) -> str:
    template = "\n\n".join(NOTEBOOK_TEMPLATE_FRAGMENTS).strip("\n")
    template = template.replace("\n\n\n@app.cell", "\n\n@app.cell")
    return template.replace("__PLOT_GALLERY_CELLS__", plot_gallery_cells)
