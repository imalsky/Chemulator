#!/usr/bin/env bash
# Local launcher for the current single-config, single-device workflow.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-}"
FLOWMAP_CONFIG="${FLOWMAP_CONFIG:-$ROOT/config/config_job0.jsonc}"

if [ -n "$CONDA_ENV" ]; then
  CONDA_EXE="$(command -v conda || true)"
  [ -n "$CONDA_EXE" ] || { echo "conda not found but CONDA_ENV was provided"; exit 2; }
  CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" || { echo "failed to activate env '$CONDA_ENV'"; exit 2; }
fi

PY="${PYTHON:-$(command -v python)}"
[ -n "$PY" ] || { echo "python not found"; exit 2; }

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export FLOWMAP_CONFIG

"$PY" - <<'PY'
from importlib.util import find_spec
from pathlib import Path
import os

cfg_path = Path(os.environ["FLOWMAP_CONFIG"]).expanduser().resolve()
if not cfg_path.is_file():
    raise FileNotFoundError(f"Missing config file: {cfg_path}")

missing = [name for name in ("torch", "numpy", "h5py") if find_spec(name) is None]
if missing:
    raise RuntimeError(
        "Missing required Python packages: "
        + ", ".join(missing)
        + ". Install them in the active environment before running run.sh."
    )

print(f"Using config: {cfg_path}")
PY

TIME_CMD=()
if /usr/bin/time -v true >/dev/null 2>&1; then
  TIME_CMD=(/usr/bin/time -v)
elif /usr/bin/time -l true >/dev/null 2>&1; then
  TIME_CMD=(/usr/bin/time -l)
fi

if [ "${#TIME_CMD[@]}" -gt 0 ]; then
  exec "${TIME_CMD[@]}" "$PY" -m src.main
fi
exec "$PY" -m src.main
