#!/usr/bin/env bash
# One-time setup for the v5 poker bots on Linux:  bash setup.sh
# Creates a private Python environment in .venv (inside this folder only),
# installs the Python packages (numpy, rich, CPU-only torch, cython, eval7,
# pytest) and compiles the C++ engine. Nothing outside this folder is changed.
# Safe to run again; delete .venv first for a clean reinstall.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
echo "Setting up v5 in $HERE"

missing=()
command -v g++ >/dev/null || missing+=("g++ (build-essential)")
command -v python3 >/dev/null || missing+=("python3")
if command -v python3 >/dev/null; then
  python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || missing+=("python 3.9 or newer")
  python3 -c "import venv, ensurepip" 2>/dev/null || missing+=("python3-venv")
  python3 -c "import sysconfig, os; os.stat(os.path.join(sysconfig.get_paths()['include'], 'Python.h'))" 2>/dev/null \
    || missing+=("python3-dev")
fi
if [ ${#missing[@]} -gt 0 ]; then
  echo "Missing: ${missing[*]}"
  echo "Ubuntu or Debian:  sudo apt install build-essential python3-venv python3-dev"
  echo "Fedora:            sudo dnf install gcc-c++ python3-devel"
  echo "(On a shared machine, ask the administrator.) Then run 'bash setup.sh' again."
  exit 1
fi

if [ -d .venv ] && [ ! -x .venv/bin/python ]; then
  echo "Removing an incomplete .venv"
  rm -rf .venv
fi
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
PY="$HERE/.venv/bin/python"
"$PY" -m pip install --upgrade pip wheel setuptools
# CPU-only torch is much smaller. Fall back to the default index if it is unreachable.
if ! "$PY" -c "import torch" 2>/dev/null; then
  "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu || "$PY" -m pip install torch
fi
"$PY" -m pip install numpy rich cython pytest
"$PY" -m pip install --no-build-isolation eval7

echo "Compiling the C++ engine..."
"$PY" -c "from pokerbot.mccfr.native import build; build(force=True)"
"$PY" -c "import torch, numpy, rich, eval7; import pokerbot, pokerbot.mccfr; print('Python packages OK')"
./native/pbcfr memory --config /dev/null --players 2 --stacks 10 >/dev/null && echo "Engine OK"

echo
echo "Setup finished."
[ -f cfr_config.txt ] || echo "Next: copy your cfr_config.txt (and abstraction/, checkpoints/, blueprints/) into $HERE"
echo "In every new terminal, start with:"
echo "  cd $HERE && source .venv/bin/activate"
