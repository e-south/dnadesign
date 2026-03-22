## CI Mechanical Gates

Use these gates when users ask for automated quality assurance around SCC workflows.

### Core gates

1. Skill contract gate

```bash
bash scripts/audit-sge-hpc-ops-skill.sh
```

2. Submission policy gate (template-level)

```bash
scripts/qa-sge-submit-preflight.sh \
  --template <qsub-template-1> \
  --template <qsub-template-2> \
  --require-project-flag
```

3. Session-status gate

```bash
scripts/sge-session-status.sh --warn-over-running 3
```

For deterministic fixture validation:

```bash
scripts/sge-session-status.sh --qstat-file <fixture> --warn-over-running 3
```

4. Submission-shape advisor gate

```bash
scripts/sge-submit-shape-advisor.sh \
  --planned-submits <N> \
  --warn-over-running 3
```

5. Operator brief gate

```bash
scripts/sge-operator-brief.sh \
  --planned-submits <N> \
  --warn-over-running 3
```

6. Active-jobs reporter gate

```bash
scripts/sge-active-jobs.sh --qstat-file <fixture> --max-jobs 12
```

7. Determinism gate

Run the skill audit three times and compare outputs.

```bash
tmpdir="$(mktemp -d)"
for i in 1 2 3; do
  bash scripts/audit-sge-hpc-ops-skill.sh > "$tmpdir/run$i.log"
done
shasum "$tmpdir"/run*.log
```

### dnadesign repo gates

Use these after editing the repo-local skill or BU SCC operator docs.

```bash
uv run pytest -q src/dnadesign/densegen/tests/docs/test_bu_scc_docs_contracts.py
uv run python -m dnadesign.devtools.docs_checks --repo-root .
```

### Optional policy gate for strict environments

Use this when org policy requires shorter shared-queue runtimes by default.

```bash
scripts/qa-sge-submit-preflight.sh \
  --template <qsub-template> \
  --require-project-flag \
  --max-runtime-hours 12
```

### Failure handling

- Any gate failure blocks publish/sync for skill changes.
- If the failure is a stale policy claim, refresh BU source evidence before rerun.
- If the failure is a template contract issue, patch template or route decision before rerun.
- If the failure is a queue-pressure warning breach, include explicit confirmation checkpoint in execution plan before rerun.
- If the failure is a queue-fairness breach, replace burst submit plan with advisor-compliant array or `-hold_jid` shape.
