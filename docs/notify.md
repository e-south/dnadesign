# Notifications

`dnadesign` includes a tool-agnostic notifier CLI for sending webhook updates from batch jobs or local runs.

## CLI

```
notify send \
  --provider slack \
  --status success \
  --tool densegen \
  --run-id demo_meme_three_tfs \
  --url-env DENSEGEN_WEBHOOK_URL \
  --message "Run complete"
```

Supported providers:
- `generic` (JSON payload)
- `slack` (text payload)
- `discord` (text payload)

Exactly one of `--url` or `--url-env` is required. The notifier fails fast on missing inputs.

## Metadata

Include additional metadata with a JSON file:

```
notify send \
  --provider generic \
  --status failure \
  --tool infer \
  --run-id evo2_001 \
  --url-env INFER_WEBHOOK_URL \
  --meta outputs/meta/run_manifest.json
```

The notifier expects a JSON object in the file and attaches it to the `meta` field.

## Usage Patterns

The notifier is intentionally tool-agnostic. Wire it up in scripts, notebooks, or batch jobs
without adding tool-specific flags.

Example: wrap a pipeline step and notify on success/failure.

```bash
#!/usr/bin/env bash
set -euo pipefail

CONFIG="/path/to/config.yaml"

if uv run dense -- run -c "$CONFIG" --fresh; then
  notify send \
    --provider slack \
    --status success \
    --tool densegen \
    --run-id demo_meme_three_tfs \
    --url-env DENSEGEN_WEBHOOK_URL \
    --message "DenseGen run completed"
else
  notify send \
    --provider slack \
    --status failure \
    --tool densegen \
    --run-id demo_meme_three_tfs \
    --url-env DENSEGEN_WEBHOOK_URL \
    --message "DenseGen run failed"
  exit 1
fi
```

Example: milestone notifications.

```bash
notify send --provider generic --status started --tool densegen --run-id demo --url-env DENSEGEN_WEBHOOK_URL \
  --message "Stage-A build starting"
uv run dense -- stage-a build-pool -c "$CONFIG"
notify send --provider generic --status running --tool densegen --run-id demo --url-env DENSEGEN_WEBHOOK_URL \
  --message "Stage-B solve starting"
uv run dense -- run -c "$CONFIG" --resume
notify send --provider generic --status success --tool densegen --run-id demo --url-env DENSEGEN_WEBHOOK_URL \
  --message "Run finished"
```

## Testing Without External Notifications

Use `--dry-run` to validate payloads without sending anything:

```
notify send \
  --provider slack \
  --status running \
  --tool densegen \
  --run-id demo \
  --url https://example.invalid/webhook \
  --dry-run
```

If you want to exercise the HTTP path without hitting Slack/Discord, run a local webhook server:

```
python - <<'PY'
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        print(body.decode("utf-8"))
        self.send_response(200)
        self.end_headers()

HTTPServer(("127.0.0.1", 18080), Handler).serve_forever()
PY
```

Then in another shell:

```
notify send \
  --provider generic \
  --status success \
  --tool densegen \
  --run-id demo \
  --url http://127.0.0.1:18080
```

## Batch Schedulers (Generic)

Batch schedulers already provide exit codes and stderr/stdout logs. The simplest integration
is to call `notify send` after the main command and on error with a `trap`:

```bash
#!/usr/bin/env bash
set -euo pipefail

notify_failure() {
  notify send --provider slack --status failure --tool densegen --run-id demo \
    --url-env DENSEGEN_WEBHOOK_URL --message "Job failed (exit $?)"
}
trap notify_failure ERR

uv run dense -- run -c "$CONFIG" --fresh

notify send --provider slack --status success --tool densegen --run-id demo \
  --url-env DENSEGEN_WEBHOOK_URL --message "Job complete"
```

If you want to catch interactive session drops or explicit termination, add a signal trap:

```bash
trap 'notify send --provider slack --status failure --tool densegen --run-id demo \
  --url-env DENSEGEN_WEBHOOK_URL --message "Session terminated"' HUP TERM INT
```

## Example: Slurm (sbatch)

Slurm accepts `#SBATCH` directives in job scripts. Common options include `--output`, plus
`--mail-type` and `--mail-user` for scheduler email notifications.

```
#!/bin/bash
#SBATCH --job-name=densegen_demo
#SBATCH --output=densegen_demo.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=you@example.edu

set -euo pipefail

CONFIG="/path/to/config.yaml"
uv run dense -- run -c "$CONFIG" --fresh

notify send --provider slack --status success --tool densegen --run-id demo \
  --url-env DENSEGEN_WEBHOOK_URL --message "Slurm job finished"
```

## Example: BU SCC (SGE/qsub)

BU SCC uses Sun Grid Engine and the `qsub` command for batch submission. Jobs emit stdout
and stderr files in the working directory by default, and directives can be specified via
`#$` lines in a job script.

Minimal SGE script with a notifier hook:

```bash
#!/bin/bash -l
#$ -N densegen_demo
#$ -j y
#$ -o densegen_demo.out
#$ -m e
#$ -M you@bu.edu

set -euo pipefail

CONFIG="/path/to/config.yaml"

uv run dense -- run -c "$CONFIG" --fresh

notify send --provider slack --status success --tool densegen --run-id demo \
  --url-env DENSEGEN_WEBHOOK_URL --message "SCC job finished"
```

This keeps email handled by the scheduler while webhooks remain tool-agnostic and portable.

## Dry Run

To preview the payload:

```
notify send \
  --provider slack \
  --status running \
  --tool densegen \
  --run-id demo \
  --url https://example.com/webhook \
  --dry-run
```
