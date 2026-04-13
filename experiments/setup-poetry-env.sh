#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENTS_DIR="${ROOT_DIR}/experiments"
HNSWLIB_DIR="${ROOT_DIR}/hnswlib-original"
POETRY_BOOTSTRAP_VENV="${HOME}/.cache/flatnav-poetry-venv"
REQUIRED_POETRY_VERSION="1.8.2"

# Add local executables and module directories for this repo.
export PATH="${HOME}/.local/bin:${POETRY_BOOTSTRAP_VENV}/bin:${PATH}"
PYTHON_MODULE_PATHS="${EXPERIMENTS_DIR}:${ROOT_DIR}/python-bindings/src:${ROOT_DIR}"
export PYTHONPATH="${PYTHON_MODULE_PATHS}${PYTHONPATH:+:${PYTHONPATH}}"

# Prefer user-installed Poetry to avoid distro package conflicts.
POETRY_BIN="${HOME}/.local/bin/poetry"
if [[ -x "${POETRY_BIN}" ]] && "${POETRY_BIN}" --version 2>/dev/null | grep -q "${REQUIRED_POETRY_VERSION}"; then
  :
elif command -v poetry >/dev/null 2>&1 && "$(command -v poetry)" --version 2>/dev/null | grep -q "${REQUIRED_POETRY_VERSION}"; then
  POETRY_BIN="$(command -v poetry)"
elif [[ -x "${POETRY_BOOTSTRAP_VENV}/bin/poetry" ]] && "${POETRY_BOOTSTRAP_VENV}/bin/poetry" --version 2>/dev/null | grep -q "${REQUIRED_POETRY_VERSION}"; then
  POETRY_BIN="${POETRY_BOOTSTRAP_VENV}/bin/poetry"
else
  echo "No compatible Poetry ${REQUIRED_POETRY_VERSION} found. Bootstrapping isolated Poetry env..."
  if ! python3 -m venv "${POETRY_BOOTSTRAP_VENV}"; then
    python3 -m virtualenv "${POETRY_BOOTSTRAP_VENV}"
  fi
  "${POETRY_BOOTSTRAP_VENV}/bin/pip" install --upgrade pip "poetry==${REQUIRED_POETRY_VERSION}"
  POETRY_BIN="${POETRY_BOOTSTRAP_VENV}/bin/poetry"
fi

if ! "${POETRY_BIN}" --version >/dev/null 2>&1; then
  echo "Failed to initialize Poetry."
  exit 1
fi

cd "${EXPERIMENTS_DIR}"
if ! "${POETRY_BIN}" install --no-root; then
  echo "Poetry install failed; attempting to refresh lock file and retry..."
  "${POETRY_BIN}" lock --no-update
  "${POETRY_BIN}" install --no-root
fi

if [[ ! -d "${HNSWLIB_DIR}" ]]; then
  git clone https://github.com/BlaiseMuhirwa/hnswlib-original.git "${HNSWLIB_DIR}"
fi

cd "${HNSWLIB_DIR}/python_bindings"
"${POETRY_BIN}" install --no-root
"${POETRY_BIN}" run python setup.py bdist_wheel

cd "${EXPERIMENTS_DIR}"
# Install forked hnswlib into the Poetry virtualenv without mutating pyproject.toml.
"${POETRY_BIN}" run pip install --no-deps --force-reinstall ../hnswlib-original/python_bindings/dist/*.whl
"${POETRY_BIN}" install --no-root

echo "Poetry environment is ready."
echo "PYTHONPATH configured as: ${PYTHONPATH}"
echo "Run: cd experiments && poetry run python run-benchmark.py --help"
echo "To persist in your shell, add this line to your profile:"
echo "export PYTHONPATH=${PYTHON_MODULE_PATHS}:\${PYTHONPATH}" 
