#!/usr/bin/env bash
# Install a working GMR Python environment, including optional Pico/XRoboToolkit
# Python bindings for live Pico body tracking.

set -euo pipefail

ENV_NAME="gmr"
PYTHON_VERSION="3.10"
USE_CONDA=1
INSTALL_SYSTEM_DEPS=1
INSTALL_PICO=1
STRICT_PICO=0
PICO_SDK_PATH=""
PICO_SDK_PACKAGE="xrobotoolkit_sdk"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() {
  printf '[install-gmr] %s\n' "$*"
}

warn() {
  printf '[install-gmr][warn] %s\n' "$*" >&2
}

die() {
  printf '[install-gmr][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --env-name NAME          Conda env name to create/reuse. Default: ${ENV_NAME}
  --python VERSION         Python version for new conda envs. Default: ${PYTHON_VERSION}
  --no-conda               Install into the currently active Python instead of creating a conda env.
  --no-system-deps         Skip Ubuntu apt packages used by MuJoCo/OpenGL viewers.
  --no-pico                Skip Pico/XRoboToolkit Python dependency installation.
  --strict-pico            Fail if xrobotoolkit_sdk cannot be imported after installation.
  --pico-sdk-path PATH     Install Pico SDK from a local wheel/sdist/directory.
  --pico-sdk-package NAME  Pip package name for Pico SDK. Default: ${PICO_SDK_PACKAGE}
  --repo-root PATH         Repository root. Default: auto-detected.
  -h, --help               Show this help.

Examples:
  $0
  $0 --env-name cerebrix --pico-sdk-path /path/to/xrobotoolkit_sdk.whl
  $0 --no-conda --no-system-deps

Notes:
  The Pico live scripts import xrobotoolkit_sdk and also require XRoboToolkit
  PC Service to be installed and running on the machine connected to the Pico.
  This script installs the Python side; it cannot install the vendor PC Service.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="${2:-}"
      [[ -n "${ENV_NAME}" ]] || die "--env-name requires a value"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="${2:-}"
      [[ -n "${PYTHON_VERSION}" ]] || die "--python requires a value"
      shift 2
      ;;
    --no-conda)
      USE_CONDA=0
      shift
      ;;
    --no-system-deps)
      INSTALL_SYSTEM_DEPS=0
      shift
      ;;
    --no-pico)
      INSTALL_PICO=0
      shift
      ;;
    --strict-pico)
      STRICT_PICO=1
      shift
      ;;
    --pico-sdk-path)
      PICO_SDK_PATH="${2:-}"
      [[ -n "${PICO_SDK_PATH}" ]] || die "--pico-sdk-path requires a value"
      shift 2
      ;;
    --pico-sdk-package)
      PICO_SDK_PACKAGE="${2:-}"
      [[ -n "${PICO_SDK_PACKAGE}" ]] || die "--pico-sdk-package requires a value"
      shift 2
      ;;
    --repo-root)
      REPO_ROOT="${2:-}"
      [[ -n "${REPO_ROOT}" ]] || die "--repo-root requires a value"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -f "${REPO_ROOT}/setup.py" ]] || die "setup.py not found under repo root: ${REPO_ROOT}"

install_system_deps() {
  [[ "${INSTALL_SYSTEM_DEPS}" -eq 1 ]] || return 0

  if ! command -v apt-get >/dev/null 2>&1; then
    warn "apt-get not found; skipping system packages"
    return 0
  fi

  local apt=(apt-get)
  if [[ "${EUID}" -ne 0 ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
      warn "sudo not found; skipping system packages"
      return 0
    fi
    apt=(sudo apt-get)
  fi

  log "installing Ubuntu packages for MuJoCo/OpenGL viewers"
  "${apt[@]}" update
  "${apt[@]}" install -y \
    git \
    build-essential \
    libgl1 \
    libglvnd0 \
    libglfw3 \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libice6 \
    libosmesa6 \
    patchelf
}

setup_python() {
  if [[ "${USE_CONDA}" -eq 1 ]]; then
    command -v conda >/dev/null 2>&1 || die "conda not found; install conda/mamba or rerun with --no-conda"

    local conda_base
    conda_base="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${conda_base}/etc/profile.d/conda.sh"

    if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
      log "reusing conda env: ${ENV_NAME}"
    else
      log "creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
      conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
    fi

    log "installing conda runtime library compatibility package"
    conda install -n "${ENV_NAME}" -c conda-forge libstdcxx-ng -y

    PYTHON_CMD=(conda run -n "${ENV_NAME}" python)
  else
    command -v python >/dev/null 2>&1 || die "python not found"
    PYTHON_CMD=(python)
    log "using active Python: $("${PYTHON_CMD[@]}" -c 'import sys; print(sys.executable)')"
  fi

  PIP_CMD=("${PYTHON_CMD[@]}" -m pip)
}

pip_install() {
  "${PIP_CMD[@]}" install "$@"
}

install_python_deps() {
  log "upgrading pip/setuptools/wheel"
  pip_install --upgrade pip setuptools wheel

  log "installing GMR editable package from ${REPO_ROOT}"
  pip_install --upgrade-strategy only-if-needed -e "${REPO_ROOT}"

  log "pinning numpy<2 for current GMR/cerebrix compatibility"
  pip_install 'numpy<2'
}

install_pico_deps() {
  [[ "${INSTALL_PICO}" -eq 1 ]] || return 0

  if [[ -n "${PICO_SDK_PATH}" ]]; then
    [[ -e "${PICO_SDK_PATH}" ]] || die "Pico SDK path does not exist: ${PICO_SDK_PATH}"
    log "installing Pico SDK from ${PICO_SDK_PATH}"
    pip_install "${PICO_SDK_PATH}"
  elif "${PYTHON_CMD[@]}" -c 'import xrobotoolkit_sdk' >/dev/null 2>&1; then
    log "xrobotoolkit_sdk is already importable"
  else
    log "installing Pico SDK Python package: ${PICO_SDK_PACKAGE}"
    if ! pip_install "${PICO_SDK_PACKAGE}"; then
      local msg="could not install ${PICO_SDK_PACKAGE}; rerun with --pico-sdk-path /path/to/vendor_wheel.whl once the XRoboToolkit Python binding is available"
      if [[ "${STRICT_PICO}" -eq 1 ]]; then
        die "${msg}"
      fi
      warn "${msg}"
    fi
  fi
}

verify_install() {
  log "verifying GMR imports"
  "${PYTHON_CMD[@]}" - <<'PY'
import importlib

mods = [
    "numpy",
    "scipy",
    "mujoco",
    "mink",
    "rich",
    "qpsolvers",
    "general_motion_retargeting",
    "general_motion_retargeting.utils.pico_xrt",
    "general_motion_retargeting.utils.pico_xrobot",
    "general_motion_retargeting.utils.xrobot_streamer",
]
for name in mods:
    importlib.import_module(name)
print("GMR core imports OK")
PY

  if [[ "${INSTALL_PICO}" -eq 1 ]]; then
    if "${PYTHON_CMD[@]}" -c 'import xrobotoolkit_sdk; print("xrobotoolkit_sdk import OK")'; then
      :
    elif [[ "${STRICT_PICO}" -eq 1 ]]; then
      die "xrobotoolkit_sdk import failed"
    else
      warn "xrobotoolkit_sdk is not importable; offline Pico jsonl retargeting works, but live Pico scripts need the vendor SDK and PC Service"
    fi
  fi
}

main() {
  install_system_deps
  setup_python
  install_python_deps
  install_pico_deps
  verify_install

  log "done"
  if [[ "${USE_CONDA}" -eq 1 ]]; then
    log "activate with: conda activate ${ENV_NAME}"
  fi
  log "offline Pico retarget: python scripts/pico_to_robot.py --source pico_xrobot --pico_file <xrobot.jsonl> --robot vt_human_v2"
  log "live Pico viewer: python scripts/live_pico_xrobot_viewer.py"
}

main "$@"
