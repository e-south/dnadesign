# UMAP Gallery

The gallery shows persisted UMAP layouts for candidate and reference-oriented
spaces. Candidate ranking stays on the primary review path.

### reference_core60_strength_umap | Reference strength UMAP

#### Plot details

**Data.** The 48 reference core60 rows with completed Evo 2 7B intermediate
embedding sidecars and promoter-standard metadata where available.

**Preprocessing.** Uses the materialized reference core60 intermediate
embedding view and persisted UMAP projection. Metadata are carried from USR
reference promoter records through Construct and Infer sidecars.

**Definition.** Each point is one reference core60 embedding in persisted UMAP
coordinates. Color encodes `promoter_standard__strength_value_numeric` when a
numeric strength annotation is present.

**Decision use.** The plot checks whether Anderson iGEM, W Collection, and
Native MG1655 standards occupy interpretable neighborhoods before their labels
are used as appendix landmarks.

**Limits.** Strength values are sparse and may not share a collection-wide
scale. UMAP axes are exploratory and should not be treated as calibrated
distances.

### reference_core60_pca_scree | Reference PCA scree

#### Plot details

**Data.** PCA reducer summary for the reference core60 intermediate embedding
matrix.

**Preprocessing.** Uses the persisted PCA reducer fit on the materialized
reference core60 view.

**Definition.** The curve plots the per-component explained-variance ratio and
cumulative explained variance across retained principal components.

**Decision use.** The scree curve exposes whether the reference core60 space is
low-rank or dominated by one component before interpreting collection labels or
strength metadata.

**Limits.** The diagnostic is sample-size limited and does not identify which
sequence features drive each component.

### appendix_umap_gallery | UMAP gallery

#### Plot details

**Data.** Each panel shows the persisted UMAP projection for one candidate
representation space. The manifest records whether the expanded reference set
is matched in each panel.

**Preprocessing.** Uses the stored projection artifacts and their recorded fitting metadata. These UMAP layouts come from the stored view matrices rather than the standardized cosine-geometry contract used in the primary ladder.

**Definition.** The plotted coordinates are the persisted UMAP embeddings for each candidate space. Hue changes only recolor the same coordinates.

**Decision use.** The layout can surface density artifacts or obvious grouping failures worth checking after the primary summaries.

**Limits.** Apparent visual separation on UMAP is not evidence that a representation is better for pre-assay triage or future supervised modeling.
