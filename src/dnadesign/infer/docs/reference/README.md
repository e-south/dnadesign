## Infer Reference

### Commands

- `infer run`: run jobs from config or preset.
- `infer extract`: ad-hoc extract (single or preset multi-output).
- `infer generate`: ad-hoc generation.
- `infer presets`: list/show preset definitions.
- `infer adapters`: inspect registered adapters and functions.
- `infer validate`: config and USR validation checks.

Canonical user-facing examples are in [`../../README.md`](../../README.md).

### Configuration and Contracts

- Schema models: `src/dnadesign/infer/config.py`
- Runtime orchestration: `src/dnadesign/infer/engine.py`
- USR write-back naming contract and tests:
  - `src/dnadesign/infer/writers/usr.py`
  - `src/dnadesign/infer/tests/test_infer_usr_docs_contract.py`
  - `src/dnadesign/infer/tests/test_usr_writeback_contract.py`
