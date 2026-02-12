# Record v1 Contract

`Record` is the canonical render input for baserender vNext.

## Required fields
- `id: str`
- `alphabet: DNA | RNA | PROTEIN`
- `sequence: str`
- `features: Feature[]`
- `effects: Effect[]`
- `display: Display`
- `meta: mapping`

## Feature
- `id: str | null`
- `kind: str` (must be registered)
- `span: { start: int, end: int, strand: fwd|rev|null }`
- `label: str | null`
- `tags: str[]`
- `attrs: mapping`
- `render: mapping` with strict keys (`track`, `priority`, `lane`)

## Effect
- `kind: str` (must be registered)
- `target: mapping`
- `params: mapping`
- `render: mapping` with strict keys (`track`, `priority`, `lane`)

## Display
- `overlay_text: str | null`
- `tag_labels: mapping[str, str]`

## Strictness
- Unknown feature/effect kinds are fatal.
- Unknown render-hint keys are fatal.
- No silent coercions.
- Kmer contract requires sequence/label/strand consistency.
