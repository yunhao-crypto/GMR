#!/usr/bin/env bash
# Install a working GMR Python environment, including optional Pico/XRoboToolkit
# Python bindings for live Pico body tracking.

set -euo pipefail

ENV_NAME="gmr"
PYTHON_VERSION="3.10"
USE_CONDA=1
INSTALL_SYSTEM_DEPS=1
INSTALL_PICO=1
INSTALL_AORTA=1
STRICT_PICO=0
PICO_SDK_PATH=""
PICO_SDK_PACKAGE="xrobotoolkit_sdk"
AORTA_RELEASE="2026.7.9"
AORTA_GITHUB_REPO="VitaDynamics/aorta"
AORTA_RELEASE_BASE_URL="https://github.com/VitaDynamics/aorta/releases/download/2026.7.9"
AORTA_SDK_WHL=""
AORTA_MSGS_WHL=""
AORTA_SDK_PACKAGE="aorta_sdk"
AORTA_MSGS_PACKAGE="aorta_msgs"
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

find_conda_exe() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi
  for candidate in \
    "${HOME}/miniconda3/bin/conda" \
    "${HOME}/anaconda3/bin/conda" \
    "/opt/conda/bin/conda"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
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
  --no-aorta               Skip Aorta Python SDK/message wheel installation.
  --strict-pico            Fail if xrobotoolkit_sdk cannot be imported after installation.
  --pico-sdk-path PATH     Install Pico SDK from a local wheel/sdist/directory.
  --pico-sdk-package NAME  Pip package name for Pico SDK. Default: ${PICO_SDK_PACKAGE}
  --aorta-release VERSION  Aorta release version. Default: ${AORTA_RELEASE}
  --aorta-github-repo REPO GitHub repo used by gh release download. Default: ${AORTA_GITHUB_REPO}
  --aorta-release-url URL  Base URL containing aorta wheel files.
                            Default: ${AORTA_RELEASE_BASE_URL}
  --aorta-sdk-wheel PATH   Install aorta_sdk from a local wheel/sdist/directory.
  --aorta-msgs-wheel PATH  Install aorta_msgs from a local wheel/sdist/directory.
  --aorta-sdk-package NAME Pip package fallback for Aorta SDK. Default: ${AORTA_SDK_PACKAGE}
  --aorta-msgs-package NAME
                            Pip package fallback for Aorta messages. Default: ${AORTA_MSGS_PACKAGE}
  --repo-root PATH         Repository root. Default: auto-detected.
  -h, --help               Show this help.

Examples:
  $0
  $0 --env-name cerebrix --pico-sdk-path /path/to/xrobotoolkit_sdk.whl
  $0 --aorta-sdk-wheel /path/aorta_sdk-2026.7.9-py3-none-linux_x86_64.whl \\
     --aorta-msgs-wheel /path/aorta_msgs-2026.7.9-py3-none-any.whl
  $0 --no-conda --no-system-deps

Notes:
  The Pico live scripts import xrobotoolkit_sdk and also require XRoboToolkit
  PC Service to be installed and running on the machine connected to the Pico.
  This script installs the Python side; it cannot install the vendor PC Service.
  Aorta release assets may require GitHub credentials if the repository is
  private. In that case, download the two wheel files and pass the local paths.
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
    --no-aorta)
      INSTALL_AORTA=0
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
    --aorta-release)
      AORTA_RELEASE="${2:-}"
      [[ -n "${AORTA_RELEASE}" ]] || die "--aorta-release requires a value"
      AORTA_RELEASE_BASE_URL="https://github.com/VitaDynamics/aorta/releases/download/${AORTA_RELEASE}"
      shift 2
      ;;
    --aorta-github-repo)
      AORTA_GITHUB_REPO="${2:-}"
      [[ -n "${AORTA_GITHUB_REPO}" ]] || die "--aorta-github-repo requires a value"
      shift 2
      ;;
    --aorta-release-url)
      AORTA_RELEASE_BASE_URL="${2:-}"
      [[ -n "${AORTA_RELEASE_BASE_URL}" ]] || die "--aorta-release-url requires a value"
      shift 2
      ;;
    --aorta-sdk-wheel)
      AORTA_SDK_WHL="${2:-}"
      [[ -n "${AORTA_SDK_WHL}" ]] || die "--aorta-sdk-wheel requires a value"
      shift 2
      ;;
    --aorta-msgs-wheel)
      AORTA_MSGS_WHL="${2:-}"
      [[ -n "${AORTA_MSGS_WHL}" ]] || die "--aorta-msgs-wheel requires a value"
      shift 2
      ;;
    --aorta-sdk-package)
      AORTA_SDK_PACKAGE="${2:-}"
      [[ -n "${AORTA_SDK_PACKAGE}" ]] || die "--aorta-sdk-package requires a value"
      shift 2
      ;;
    --aorta-msgs-package)
      AORTA_MSGS_PACKAGE="${2:-}"
      [[ -n "${AORTA_MSGS_PACKAGE}" ]] || die "--aorta-msgs-package requires a value"
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
    local conda_exe
    conda_exe="$(find_conda_exe)" || die "conda not found; install conda/mamba or rerun with --no-conda"

    local conda_base
    conda_base="$("${conda_exe}" info --base)"
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

has_aorta_lowcmd_schema() {
  "${PYTHON_CMD[@]}" - <<'PY' >/dev/null 2>&1
import importlib

for name in (
    "low_cmd_schema_meta",
    "aorta_msgs.low_cmd_schema_meta",
    "aorta_msgs.lowlevel.low_cmd_schema_meta",
    "lowlevel.low_cmd_schema_meta",
):
    try:
        importlib.import_module(name)
        raise SystemExit(0)
    except ImportError:
        pass
raise SystemExit(1)
PY
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

install_aorta_from_gh_release() {
  local tag="$1"

  command -v gh >/dev/null 2>&1 || return 1

  local tmp_dir
  tmp_dir="$(mktemp -d)"
  if gh release download "${tag}" \
    --repo "${AORTA_GITHUB_REPO}" \
    --pattern "aorta_msgs-${AORTA_RELEASE}-py3-none-any.whl" \
    --pattern "aorta_sdk-${AORTA_RELEASE}-py3-none-linux_x86_64.whl" \
    --dir "${tmp_dir}"; then
    local msgs_wheel="${tmp_dir}/aorta_msgs-${AORTA_RELEASE}-py3-none-any.whl"
    local sdk_wheel="${tmp_dir}/aorta_sdk-${AORTA_RELEASE}-py3-none-linux_x86_64.whl"
    if [[ -e "${msgs_wheel}" && -e "${sdk_wheel}" ]] && pip_install "${msgs_wheel}" "${sdk_wheel}"; then
      rm -rf "${tmp_dir}"
      return 0
    fi
  fi

  rm -rf "${tmp_dir}"
  return 1
}

install_aorta_deps() {
  [[ "${INSTALL_AORTA}" -eq 1 ]] || return 0

  local sdk_url="${AORTA_RELEASE_BASE_URL}/aorta_sdk-${AORTA_RELEASE}-py3-none-linux_x86_64.whl"
  local msgs_url="${AORTA_RELEASE_BASE_URL}/aorta_msgs-${AORTA_RELEASE}-py3-none-any.whl"
  local fallback_base_url="https://github.com/VitaDynamics/aorta/releases/download/v${AORTA_RELEASE}"
  local fallback_sdk_url="${fallback_base_url}/aorta_sdk-${AORTA_RELEASE}-py3-none-linux_x86_64.whl"
  local fallback_msgs_url="${fallback_base_url}/aorta_msgs-${AORTA_RELEASE}-py3-none-any.whl"

  if [[ -n "${AORTA_SDK_WHL}" || -n "${AORTA_MSGS_WHL}" ]]; then
    [[ -n "${AORTA_SDK_WHL}" ]] || die "--aorta-sdk-wheel is required when using local Aorta wheels"
    [[ -n "${AORTA_MSGS_WHL}" ]] || die "--aorta-msgs-wheel is required when using local Aorta wheels"
    [[ -e "${AORTA_SDK_WHL}" ]] || die "Aorta SDK wheel path does not exist: ${AORTA_SDK_WHL}"
    [[ -e "${AORTA_MSGS_WHL}" ]] || die "Aorta messages wheel path does not exist: ${AORTA_MSGS_WHL}"
    log "installing Aorta wheels from local paths"
    pip_install "${AORTA_MSGS_WHL}" "${AORTA_SDK_WHL}"
    return 0
  fi

  if "${PYTHON_CMD[@]}" -c 'import aorta' >/dev/null 2>&1 && has_aorta_lowcmd_schema; then
    log "Aorta SDK and low_cmd schema metadata are already importable"
    return 0
  fi

  if ! "${PYTHON_CMD[@]}" -c 'import aorta' >/dev/null 2>&1 || ! has_aorta_lowcmd_schema; then
    log "installing Aorta wheels from ${AORTA_RELEASE_BASE_URL}"
    if ! pip_install "${msgs_url}" "${sdk_url}"; then
      warn "could not install Aorta wheels from release URL; trying ${fallback_base_url}"
      if ! pip_install "${fallback_msgs_url}" "${fallback_sdk_url}"; then
        warn "could not install Aorta wheels from v-prefixed release URL; trying gh release download"
        if ! install_aorta_from_gh_release "${AORTA_RELEASE}"; then
          if ! install_aorta_from_gh_release "v${AORTA_RELEASE}"; then
            warn "could not install Aorta wheels via gh; trying pip package names"
            if ! pip_install "${AORTA_MSGS_PACKAGE}" "${AORTA_SDK_PACKAGE}"; then
              warn "could not install Aorta packages; rerun with --aorta-sdk-wheel and --aorta-msgs-wheel if the GitHub release is private"
            fi
          fi
        fi
      fi
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

  if [[ "${INSTALL_AORTA}" -eq 1 ]]; then
    "${PYTHON_CMD[@]}" - <<'PY'
import importlib

import aorta

candidates = [
    "low_cmd_schema_meta",
    "aorta_msgs.low_cmd_schema_meta",
    "aorta_msgs.lowlevel.low_cmd_schema_meta",
    "lowlevel.low_cmd_schema_meta",
]
for name in candidates:
    try:
        importlib.import_module(name)
        print(f"Aorta LowCmd schema import OK: {name}")
        break
    except ImportError:
        pass
else:
    raise SystemExit(
        "Aorta SDK imports, but low_cmd schema metadata was not found; "
        "check the aorta_msgs wheel contents or pass --aorta-schema-module "
        "to the live publisher script."
    )
print("Aorta SDK import OK")
PY
  fi
}

main() {
  install_system_deps
  setup_python
  install_python_deps
  install_aorta_deps
  install_pico_deps
  verify_install

  log "done"
  if [[ "${USE_CONDA}" -eq 1 ]]; then
    log "activate with: conda activate ${ENV_NAME}"
  fi
  log "offline Pico retarget: python scripts/pico_to_robot.py --source pico_xrobot --pico_file <xrobot.jsonl> --robot vt_human_v2"
  log "live Pico viewer: python scripts/live_pico_xrobot_viewer.py"
  log "live Pico -> Aorta: ./scripts/run_live_pico_aorta.sh --robot-sn <robot_sn> --robot-ip <robot_ip>"
}

main "$@"
