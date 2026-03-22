## Automation QA Preflight

Use this reference to enforce mechanical checks before any real `qsub` in agentic workflows.

### Why this exists

- Prevent avoidable process-reaper terminations.
- Keep automated runs aligned with BU SCC best-practice expectations.
- Provide CI-runnable safety checks for single-job, array, and multi-job flows.

### Flow classes

- `single_batch`: one submit template and one expected job id.
- `array_batch`: one template with `-t` and task-indexed behavior.
- `multi_batch_pipeline`: two or more submit artifacts with dependencies or watcher + worker pairing.
- `interactive_handoff`: interactive shell context used for diagnostics while long runs move to batch.

### Required QA checks by flow class

| Check | single_batch | array_batch | multi_batch_pipeline | interactive_handoff |
| --- | --- | --- | --- | --- |
| Module-safe shebang for module workflows (`#!/bin/bash -l`) | required | required | required | n/a |
| Explicit runtime (`h_rt`) | required | required | required | required for queued interactive submits |
| Project/account attribution | required | required | required | required for queued interactive submits |
| Array-task usage (`SGE_TASK_ID`) | n/a | required | required when arrays used | n/a |
| Thread-slot alignment (`OMP_NUM_THREADS=$NSLOTS`) | required when threaded | required when threaded | required when threaded | n/a |
| Dependency sequencing (`-hold_jid`) | optional | optional | required when ordering is needed | n/a |
| Storage placement (`/project` or `/projectnb`) | required | required | required | required for handoff-created outputs |
| Active-job pressure check (`running_jobs > 3` warning + acknowledgement) | required | required | required | required for queued interactive submits |
| Submission-shape advisor check (`array`/`hold_jid` decision) | required when multi-submit | required when multi-submit | required | required when queued submits are planned |
| Queue fairness check (no line-skipping or queue bypass) | required | required | required | required |
| Reaper-risk classification | required | required | required | required |

### Active-job pressure default

- Use `3` as the default warning threshold for running jobs.
- If `running_jobs > 3` and user requests additional submit actions, warn and require explicit confirmation.
- In threshold-exceeded cases, prefer array conversion or dependency chaining before raw parallel submit expansion.

### Queue fairness default

- Respect the queue and do not skip the line.
- Avoid submit bursts when the advisor recommends `array` or `hold_jid`.
- If user asks for queue-bypass style behavior, refuse and provide compliant alternatives.
- Reject batch templates that include `#$ -now y`.

### Reaper-risk classification

Use this taxonomy before submit and at incident triage time.

- `login_cpu_risk`:
  - workload includes long/high-CPU commands intended for login node execution.
  - mitigation: route to batch or interactive scheduler methods.
- `slot_overuse_risk`:
  - runtime process/thread count likely exceeds requested slots.
  - mitigation: align slots with thread/process configuration.
- `idle_gpu_risk`:
  - workflow requests GPUs without confirmed GPU-bound runtime activity.
  - mitigation: verify GPU usage path before submit.
- `unassigned_gpu_risk`:
  - workflow could bypass assigned devices.
  - mitigation: constrain runtime to assigned GPUs and avoid out-of-allocation access.

### Standard commands

```bash
scripts/qa-sge-submit-preflight.sh \
  --template <PATH_TO_QSUB> \
  --require-project-flag
```

```bash
scripts/sge-session-status.sh --warn-over-running 3
```

```bash
scripts/sge-submit-shape-advisor.sh \
  --planned-submits <N> \
  --warn-over-running 3
```

For pipeline workflows, run preflight once per template and include one session-status report plus one advisor output before each submit phase.

### Required evidence artifacts

- QA preflight output per template (pass or fail with reasons)
- route and flow class decision (`workflow_id` plus flow class)
- session-status report with threshold status
- shape-advisor output (`advisor`, `reason`, `recommended_action`)
- operator-brief output (`submit_gate`, `next_action`)
- queue-fairness acknowledgement
- reaper-risk classification outcome
- submit fingerprint used for dedupe and retry control
