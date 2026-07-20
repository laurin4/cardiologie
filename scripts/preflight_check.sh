#!/usr/bin/env bash
set -euo pipefail

# Pre-run sanity checks for the clinical extraction framework.
#
# Usage:
#   scripts/preflight_check.sh [REPORTS_SOURCE]
#
# Verifies the Python interpreter, that the report source exists, and that the
# registered tasks import cleanly.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-"$ROOT_DIR/cardiology_venv/bin/python"}"
REPORTS="${1:-$ROOT_DIR/examples/demo_reports}"

echo "=== Extraction framework preflight ==="
echo "Project root: $ROOT_DIR"
echo "Python:       $PYTHON_BIN"
echo "Reports:      $REPORTS"
echo

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python not executable at $PYTHON_BIN"
  echo "Hint: create/activate a venv or set PYTHON_BIN explicitly."
  exit 1
fi

if [[ ! -e "$REPORTS" ]]; then
  echo "ERROR: report source not found: $REPORTS"
  exit 1
fi

cd "$ROOT_DIR"

echo "[1/2] Registered tasks import cleanly"
"$PYTHON_BIN" - <<'PY'
from configs.tasks import available_tasks, load_task
for name in available_tasks():
    task = load_task(name)
    print(f"  ok: {name} ({len(task.fields)} fields, {len(task.evidence_groups)} evidence groups)")
PY
echo

echo "[2/2] Validate input reports"
"$PYTHON_BIN" -m src.validation.validate_inputs --reports "$REPORTS"
echo

echo "Preflight completed."
