"""
LatentDNA CLI application.
"""

from __future__ import annotations

import typer

from .commands.agreement import app as agreement_app
from .commands.alignment import app as alignment_app
from .commands.cluster import app as cluster_app
from .commands.deliverable import app as deliverable_app
from .commands.distance import app as distance_app
from .commands.enrich import app as enrich_app
from .commands.export import app as export_app
from .commands.inspect import app as inspect_app
from .commands.neighbors import app as neighbors_app
from .commands.notebook import app as notebook_app
from .commands.plot import app as plot_app
from .commands.projection import app as projection_app
from .commands.recipe import app as recipe_app
from .commands.runs import app as runs_app
from .commands.sample import app as sample_app
from .commands.scalar import app as scalar_app
from .commands.snapshot import app as snapshot_app
from .commands.validate import app as validate_app
from .commands.view import app as view_app
from .commands.workspace import app as workspace_app

app = typer.Typer(help="Artifact-first downstream latent analysis for dnadesign.")
app.add_typer(workspace_app, name="workspace")
app.add_typer(validate_app, name="validate")
app.add_typer(inspect_app, name="inspect")
app.add_typer(snapshot_app, name="snapshot")
app.add_typer(alignment_app, name="alignment")
app.add_typer(view_app, name="view")
app.add_typer(scalar_app, name="scalar")
app.add_typer(sample_app, name="sample")
app.add_typer(neighbors_app, name="neighbors")
app.add_typer(cluster_app, name="cluster")
app.add_typer(notebook_app, name="notebook")
app.add_typer(projection_app, name="projection")
app.add_typer(distance_app, name="distance")
app.add_typer(enrich_app, name="enrich")
app.add_typer(agreement_app, name="agreement")
app.add_typer(plot_app, name="plot")
app.add_typer(export_app, name="export")
app.add_typer(recipe_app, name="recipe")
app.add_typer(deliverable_app, name="deliverable")
app.add_typer(runs_app, name="runs")
