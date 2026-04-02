#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
MODE="${1:-}"

install_packages() {
  local pip_cmd="$1"
  "${pip_cmd}" install --upgrade pip setuptools wheel
  "${pip_cmd}" install \
    pyvisa \
    pyvisa-py \
    pyusb \
    psutil \
    zeroconf \
    matplotlib
}

setup_global_shims() {
  local py_bin pip_bin
  py_bin="$(command -v python3)"
  pip_bin="$(command -v pip3)"
  mkdir -p "${VENV_DIR}/bin"
  ln -sf "${py_bin}" "${VENV_DIR}/bin/python3"
  ln -sf "${py_bin}" "${VENV_DIR}/bin/python"
  ln -sf "${pip_bin}" "${VENV_DIR}/bin/pip3"
  ln -sf "${pip_bin}" "${VENV_DIR}/bin/pip"
  cat > "${VENV_DIR}/pyvenv.cfg" <<EOF
home = $(dirname "$(dirname "${py_bin}")")
include-system-site-packages = true
version = $("${py_bin}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
EOF
}

choose_mode() {
  if [[ -n "${MODE}" ]]; then
    return
  fi
  echo "Choose install mode:"
  echo "  1) Virtualenv (.venv) [recommended]"
  echo "  2) Global (system Python)"
  read -r -p "Enter 1 or 2: " choice
  case "${choice}" in
    1) MODE="venv" ;;
    2) MODE="global" ;;
    *) echo "Invalid choice."; exit 2 ;;
  esac
}

choose_mode

case "${MODE}" in
  venv|--venv)
    echo "Creating virtual environment at ${VENV_DIR} ..."
    python3 -m venv "${VENV_DIR}"
    echo "Installing into ${VENV_DIR} ..."
    install_packages "${VENV_DIR}/bin/pip"
    echo
    echo "Setup complete (venv mode)."
    echo "Use the environment with:"
    echo "  source .venv/bin/activate"
    ;;
  global|--global)
    echo "Installing into global Python environment ..."
    install_packages "pip3"
    echo "Creating .venv shim wrappers to system python/pip ..."
    setup_global_shims
    echo
    echo "Setup complete (global mode)."
    echo "Shebang compatibility shims created in .venv/bin."
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Usage: ./setup.sh [venv|global|--venv|--global]"
    exit 2
    ;;
esac
