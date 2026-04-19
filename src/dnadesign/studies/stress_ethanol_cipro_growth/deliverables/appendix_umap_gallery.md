# UMAP Gallery

The gallery shows the persisted UMAP layouts for the candidate spaces. Candidate ranking stays on the primary review path.

### appendix_umap_gallery | UMAP gallery

#### Plot details

**Data.** Each panel shows the persisted UMAP projection for one candidate representation space.

**Preprocessing.** Uses the stored projection artifacts and their recorded fitting metadata. These UMAP layouts come from the stored view matrices rather than the standardized cosine-geometry contract used in the primary ladder.

**Definition.** The plotted coordinates are the persisted UMAP embeddings for each candidate space. Hue changes only recolor the same coordinates.

**Decision use.** The layout can surface density artifacts or obvious grouping failures worth checking after the primary summaries.

**Limits.** Apparent visual separation on UMAP is not evidence that a representation is better for pre-assay triage or future supervised modeling.
