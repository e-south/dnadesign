## <study-id>

Replace every placeholder before relying on this file for current-status checks.
If a branch of work is not active, mark it `n/a` explicitly instead of leaving a
placeholder behind.

- Last verified:
- Owner:
- Affiliated dataset registry: `datasets.yaml`
- Route map: `routes.md` or `n/a`
- Study execution map: `pipeline.yaml` or `n/a`
- USR root:

### Current datasets

- DenseGen anchor source: `<dataset>` (`present|planned`, shared source or `n/a`)
- Wildtype or manual controls: `<dataset>` (`<rows>` rows) or `n/a`
- Construct template seed: `<dataset>` (`<rows>` rows) or `n/a`
- Anchor-only handoff: `<dataset>` (`present|planned`, shared infer plane or `n/a`)
- Construct-expanded handoff: `<dataset>` (`present|planned`, shared infer plane or `n/a`)
- Canonical consolidated feature dataset: `<dataset>` (`present|planned` or `n/a`)

### Current phase

- Declared phase: `<phase-id>`
- DenseGen growth: `<pending|running|complete|parallel_optional>`
- Merged anchor set: `<pending|running|complete|n/a>`
- Construct context expansion: `<pending|running|complete|n/a>`
- Next in-progress surface: `<doc path or workspace>`
- Preferred infer family: `<model family>` or `n/a`
- Supported infer families: `<family>`, `<family>` or `n/a`

### Current row counts

- `<dataset>`: `<rows>`
- `<dataset>`: `<rows>`
- `<dataset>`: `<rows>`
- `<dataset>`: `n/a` (`planned`) or `<rows>`
- DenseGen source row target: `<rows>`
- Current DenseGen row gap: `<rows>`
- Shared handoff metadata posture: `densegen__plan` and `densegen__required_regulators` are complete for all DenseGen-derived handoff rows, or describe the exact missing metadata surface

### Current downstream posture

- LatentDNA: `configured|planned|not configured`; `<one-line readiness note>`
- Cluster: `configured|planned|not configured`; `<one-line readiness note>`
- OPAL: `configured|planned|not configured`; `<one-line readiness note>`
- Use `routes.md` for owner tool, entry artifact, primary doc or workspace, and first command per downstream branch

### Next actions

- `<action>`
- `<action>`
