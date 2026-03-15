#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${CONSTRUCT_RUNBOOK_MODE:-dry-run}"
CONFIG="${CONSTRUCT_RUNBOOK_CONFIG:-$WORKSPACE_DIR/config.slot_a.window.yaml}"
MANIFEST="$WORKSPACE_DIR/inputs/seed_manifest.yaml"
USR_ROOT="${CONSTRUCT_RUNBOOK_USR_ROOT:-$WORKSPACE_DIR/outputs/usr_datasets}"
PROJECT_ROOT="${CONSTRUCT_RUNBOOK_PROJECT_ROOT:-__CONSTRUCT_PROJECT_ROOT__}"

usage() {
  cat <<'EOF'
Usage:
  ./runbook.sh --mode <seed|validate|dry-run|run|validate-all> [--config <path>]

Modes:
  seed          Bootstrap the curated demo USR datasets only.
  validate      Seed demo datasets, then runtime-validate one config.
  dry-run       Seed demo datasets, runtime-validate one config, then dry-run it.
  run           Seed demo datasets, runtime-validate one config, then materialize it.
  validate-all  Seed demo datasets, then runtime-validate all packaged configs.

Environment overrides:
  CONSTRUCT_RUNBOOK_MODE
  CONSTRUCT_RUNBOOK_CONFIG
  CONSTRUCT_RUNBOOK_USR_ROOT
  CONSTRUCT_RUNBOOK_PROJECT_ROOT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

seed_demo() {
  uv run --project "$PROJECT_ROOT" construct seed promoter-swap-demo \
    --root "$USR_ROOT" \
    --manifest "$MANIFEST"
}

validate_one() {
  uv run --project "$PROJECT_ROOT" construct validate config --config "$1" --runtime
}

dry_run_one() {
  uv run --project "$PROJECT_ROOT" construct run --config "$1" --dry-run
}

run_one() {
  uv run --project "$PROJECT_ROOT" construct run --config "$1"
}

validate_all() {
  local configs=(
    "$WORKSPACE_DIR/config.slot_a.window.yaml"
    "$WORKSPACE_DIR/config.slot_a.full.yaml"
    "$WORKSPACE_DIR/config.slot_b.window.yaml"
    "$WORKSPACE_DIR/config.slot_b.full.yaml"
  )
  local cfg
  for cfg in "${configs[@]}"; do
    validate_one "$cfg"
  done
}

seed_demo

case "$MODE" in
  seed)
    ;;
  validate)
    validate_one "$CONFIG"
    ;;
  dry-run)
    validate_one "$CONFIG"
    dry_run_one "$CONFIG"
    ;;
  run)
    validate_one "$CONFIG"
    run_one "$CONFIG"
    ;;
  validate-all)
    validate_all
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac
