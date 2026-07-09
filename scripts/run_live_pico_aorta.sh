#!/usr/bin/env bash
# Start live Pico/XRoboToolkit retargeting and publish Aorta LowCmd by default.

set -euo pipefail

ENV_NAME="gmr"
USE_CONDA=1
AORTA_GROUP_VALUE="${AORTA_GROUP:-default}"
ZENOH_CONFIG_URI="${ZENOH_SESSION_CONFIG_URI:-}"
ROBOT_SN=""
ROBOT_ENDPOINT=""
ROBOT_IP=""
ROBOT_PORT="7447"
AORTA_TOPIC="/locomotion/external_command"
LOWCMD_LAYOUT="extended44"
SRC_HUMAN="xrobot"
ROBOT="vt_human_v2"
ACTUAL_HUMAN_HEIGHT="1.6"
FPS="30.0"
RECORD_PATH=""
CHECK_ENV=1
DRY_RUN=0
DISPLAY_VALUE="${DISPLAY:-}"
XAUTHORITY_VALUE="${XAUTHORITY:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ZENOH_CONFIG="${REPO_ROOT}/config/dev_machine_to_bot.json5"
EXTRA_ARGS=()
GENERATED_ZENOH_CONFIG=""

log() {
  printf '[run-pico-aorta] %s\n' "$*"
}

warn() {
  printf '[run-pico-aorta][warn] %s\n' "$*" >&2
}

die() {
  printf '[run-pico-aorta][error] %s\n' "$*" >&2
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

detect_display() {
  if [[ -n "${DISPLAY_VALUE}" ]]; then
    return 0
  fi
  local socket
  socket="$(find /tmp/.X11-unix -maxdepth 1 -type s -name 'X*' 2>/dev/null | sort | head -n 1 || true)"
  if [[ -n "${socket}" ]]; then
    DISPLAY_VALUE=":${socket##*/X}"
  fi
}

detect_xauthority() {
  if [[ -n "${XAUTHORITY_VALUE}" ]]; then
    return 0
  fi
  local uid
  uid="$(id -u)"
  for candidate in \
    "${HOME}/.Xauthority" \
    "/run/user/${uid}/gdm/Xauthority" \
    "/run/user/${uid}/Xauthority"; do
    if [[ -r "${candidate}" ]]; then
      XAUTHORITY_VALUE="${candidate}"
      return 0
    fi
  done
}

usage() {
  cat <<EOF
Usage: $0 [options] [-- extra live_pico_xrobot_viewer.py args]

Options:
  --env-name NAME             Conda env to activate. Default: ${ENV_NAME}
  --no-conda                  Use the current shell Python instead of conda activate.
  --zenoh-config PATH_OR_URI  Export ZENOH_SESSION_CONFIG_URI.
                               Default: config/dev_machine_to_bot.json5
  --robot-sn SN               Override Zenoh namespace for the target robot.
  --robot-namespace SN        Alias of --robot-sn.
  --robot-ip IP               Override connect endpoint as tcp/IP:7447.
  --robot-port PORT           Port used with --robot-ip. Default: ${ROBOT_PORT}
  --robot-endpoint ENDPOINT   Override connect endpoint, e.g. tcp/10.100.23.163:7447.
  --aorta-group GROUP         Export AORTA_GROUP. Default: ${AORTA_GROUP_VALUE}
  --aorta-topic TOPIC         Aorta LowCmd topic. Default: ${AORTA_TOPIC}
  --lowcmd-layout NAME        LowCmd layout: extended44 or legacy22. Default: ${LOWCMD_LAYOUT}
  --src-human NAME            Retarget source: xrobot or pico_xrobot. Default: ${SRC_HUMAN}
  --robot NAME                Robot model. Default: ${ROBOT}
  --actual-human-height M     Human height for GMR scaling. Default: ${ACTUAL_HUMAN_HEIGHT}
  --fps FPS                   Viewer/render loop FPS. Default: ${FPS}
  --record-path PATH          Record retargeted qpos JSONL while streaming.
  --display DISPLAY           Export DISPLAY for MuJoCo GUI. Default: auto-detect local X display.
  --xauthority PATH           Export XAUTHORITY for MuJoCo GUI. Default: auto-detect when available.
  --skip-env-check            Skip import checks before starting.
  --dry-run                   Print the resolved command without running it.
  -h, --help                  Show this help.

Examples:
  $0
  $0 --robot-sn 80100090026FF100001 --robot-ip 10.100.23.163
  $0 --zenoh-config /path/to/robot_connect.json5 --robot-endpoint tcp/10.100.23.163:7447
  $0 --env-name gmr --aorta-group default --zenoh-config file:///tmp/connect.json5
  ZENOH_SESSION_CONFIG_URI=/tmp/connect.json5 $0 -- --show-human
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="${2:-}"
      [[ -n "${ENV_NAME}" ]] || die "--env-name requires a value"
      shift 2
      ;;
    --no-conda)
      USE_CONDA=0
      shift
      ;;
    --zenoh-config)
      ZENOH_CONFIG_URI="${2:-}"
      [[ -n "${ZENOH_CONFIG_URI}" ]] || die "--zenoh-config requires a value"
      shift 2
      ;;
    --robot-sn|--robot-namespace)
      ROBOT_SN="${2:-}"
      [[ -n "${ROBOT_SN}" ]] || die "$1 requires a value"
      shift 2
      ;;
    --robot-ip)
      ROBOT_IP="${2:-}"
      [[ -n "${ROBOT_IP}" ]] || die "--robot-ip requires a value"
      shift 2
      ;;
    --robot-port)
      ROBOT_PORT="${2:-}"
      [[ -n "${ROBOT_PORT}" ]] || die "--robot-port requires a value"
      shift 2
      ;;
    --robot-endpoint)
      ROBOT_ENDPOINT="${2:-}"
      [[ -n "${ROBOT_ENDPOINT}" ]] || die "--robot-endpoint requires a value"
      shift 2
      ;;
    --aorta-group)
      AORTA_GROUP_VALUE="${2:-}"
      [[ -n "${AORTA_GROUP_VALUE}" ]] || die "--aorta-group requires a value"
      shift 2
      ;;
    --aorta-topic)
      AORTA_TOPIC="${2:-}"
      [[ -n "${AORTA_TOPIC}" ]] || die "--aorta-topic requires a value"
      shift 2
      ;;
    --lowcmd-layout)
      LOWCMD_LAYOUT="${2:-}"
      [[ -n "${LOWCMD_LAYOUT}" ]] || die "--lowcmd-layout requires a value"
      shift 2
      ;;
    --src-human)
      SRC_HUMAN="${2:-}"
      [[ -n "${SRC_HUMAN}" ]] || die "--src-human requires a value"
      shift 2
      ;;
    --robot)
      ROBOT="${2:-}"
      [[ -n "${ROBOT}" ]] || die "--robot requires a value"
      shift 2
      ;;
    --actual-human-height)
      ACTUAL_HUMAN_HEIGHT="${2:-}"
      [[ -n "${ACTUAL_HUMAN_HEIGHT}" ]] || die "--actual-human-height requires a value"
      shift 2
      ;;
    --fps)
      FPS="${2:-}"
      [[ -n "${FPS}" ]] || die "--fps requires a value"
      shift 2
      ;;
    --record-path)
      RECORD_PATH="${2:-}"
      [[ -n "${RECORD_PATH}" ]] || die "--record-path requires a value"
      shift 2
      ;;
    --display)
      DISPLAY_VALUE="${2:-}"
      [[ -n "${DISPLAY_VALUE}" ]] || die "--display requires a value"
      shift 2
      ;;
    --xauthority)
      XAUTHORITY_VALUE="${2:-}"
      [[ -n "${XAUTHORITY_VALUE}" ]] || die "--xauthority requires a value"
      shift 2
      ;;
    --skip-env-check)
      CHECK_ENV=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -f "${REPO_ROOT}/setup.py" ]] || die "setup.py not found under repo root: ${REPO_ROOT}"

cleanup() {
  if [[ -n "${GENERATED_ZENOH_CONFIG}" && -e "${GENERATED_ZENOH_CONFIG}" ]]; then
    rm -f "${GENERATED_ZENOH_CONFIG}"
  fi
}
trap cleanup EXIT

if [[ "${USE_CONDA}" -eq 1 ]]; then
  CONDA_EXE="$(find_conda_exe)" || die "conda not found; run install_gmr_env.sh or use --no-conda"
  CONDA_BASE="$("${CONDA_EXE}" info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  log "activating conda env: ${ENV_NAME}"
  conda activate "${ENV_NAME}"
else
  command -v python >/dev/null 2>&1 || die "python not found"
  log "using current Python: $(python -c 'import sys; print(sys.executable)')"
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export AORTA_GROUP="${AORTA_GROUP_VALUE}"
detect_display
detect_xauthority
if [[ -n "${DISPLAY_VALUE}" ]]; then
  export DISPLAY="${DISPLAY_VALUE}"
fi
if [[ -n "${XAUTHORITY_VALUE}" ]]; then
  export XAUTHORITY="${XAUTHORITY_VALUE}"
fi
if [[ -n "${ROBOT_IP}" && -n "${ROBOT_ENDPOINT}" ]]; then
  die "use only one of --robot-ip or --robot-endpoint"
fi
if [[ -n "${ROBOT_IP}" ]]; then
  ROBOT_ENDPOINT="tcp/${ROBOT_IP}:${ROBOT_PORT}"
fi

if [[ -z "${ZENOH_CONFIG_URI}" && -f "${DEFAULT_ZENOH_CONFIG}" ]]; then
  ZENOH_CONFIG_URI="${DEFAULT_ZENOH_CONFIG}"
fi

if [[ -n "${ZENOH_CONFIG_URI}" ]]; then
  ZENOH_CONFIG_PATH="${ZENOH_CONFIG_URI}"
  if [[ "${ZENOH_CONFIG_URI}" == file://localhost/* ]]; then
    ZENOH_CONFIG_PATH="${ZENOH_CONFIG_URI#file://localhost}"
  elif [[ "${ZENOH_CONFIG_URI}" == file:///* ]]; then
    ZENOH_CONFIG_PATH="${ZENOH_CONFIG_URI#file://}"
  elif [[ "${ZENOH_CONFIG_URI}" == *://* ]]; then
    if [[ -n "${ROBOT_SN}" || -n "${ROBOT_ENDPOINT}" ]]; then
      die "robot overrides require a local --zenoh-config path, not URI: ${ZENOH_CONFIG_URI}"
    fi
    ZENOH_CONFIG_PATH=""
  fi

  if [[ -n "${ZENOH_CONFIG_PATH}" ]]; then
    [[ -e "${ZENOH_CONFIG_PATH}" ]] || die "Zenoh config path does not exist: ${ZENOH_CONFIG_PATH}"
  fi

  if [[ -n "${ROBOT_SN}" || -n "${ROBOT_ENDPOINT}" ]]; then
    [[ -n "${ZENOH_CONFIG_PATH}" ]] || die "robot overrides require a local Zenoh config template"
    GENERATED_ZENOH_CONFIG="$(mktemp "${TMPDIR:-/tmp}/gmr-zenoh.XXXXXX.json5")"
    TEMPLATE_PATH="${ZENOH_CONFIG_PATH}" \
      OUTPUT_PATH="${GENERATED_ZENOH_CONFIG}" \
      ROBOT_SN="${ROBOT_SN}" \
      ROBOT_ENDPOINT="${ROBOT_ENDPOINT}" \
      python - <<'PY'
import os
import re
from pathlib import Path

template = Path(os.environ["TEMPLATE_PATH"]).read_text()
robot_sn = os.environ["ROBOT_SN"]
robot_endpoint = os.environ["ROBOT_ENDPOINT"]

if robot_sn:
    template, count = re.subn(
        r'(^\s*namespace\s*:\s*)"[^"]*"',
        rf'\g<1>"{robot_sn}"',
        template,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit("Could not replace top-level namespace in Zenoh config")

if robot_endpoint:
    template, count = re.subn(
        r'(^\s*endpoints\s*:\s*)\[[^\]]*\]',
        rf'\g<1>["{robot_endpoint}"]',
        template,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit("Could not replace connect.endpoints in Zenoh config")

Path(os.environ["OUTPUT_PATH"]).write_text(template)
PY
    ZENOH_CONFIG_URI="${GENERATED_ZENOH_CONFIG}"
  fi
  export ZENOH_SESSION_CONFIG_URI="${ZENOH_CONFIG_URI}"
else
  warn "ZENOH_SESSION_CONFIG_URI is not set; Aorta will use Zenoh's default local peer configuration"
fi

case "${LOWCMD_LAYOUT}" in
  extended44|legacy22)
    ;;
  *)
    die "--lowcmd-layout must be extended44 or legacy22"
    ;;
esac

case "${SRC_HUMAN}" in
  xrobot|pico_xrobot)
    ;;
  *)
    die "--src-human must be xrobot or pico_xrobot"
    ;;
esac

if [[ "${CHECK_ENV}" -eq 1 ]]; then
  log "checking Python imports"
  python - <<'PY'
import importlib

required = [
    "general_motion_retargeting",
    "general_motion_retargeting.utils.aorta_lowcmd_publisher",
    "xrobotoolkit_sdk",
    "aorta",
]
for name in required:
    importlib.import_module(name)

schema_candidates = [
    "low_cmd_schema_meta",
    "aorta_msgs.low_cmd_schema_meta",
    "aorta_msgs.lowlevel.low_cmd_schema_meta",
    "lowlevel.low_cmd_schema_meta",
]
for name in schema_candidates:
    try:
        importlib.import_module(name)
        print(f"low_cmd schema import OK: {name}")
        break
    except ImportError:
        pass
else:
    raise SystemExit(
        "Could not import low_cmd schema metadata. Install aorta_msgs or pass "
        "--aorta-schema-module through extra args after --."
    )
print("environment import checks OK")
PY
fi

cmd=(
  python "${REPO_ROOT}/scripts/live_pico_xrobot_viewer.py"
  --robot "${ROBOT}"
  --actual-human-height "${ACTUAL_HUMAN_HEIGHT}"
  --fps "${FPS}"
  --publish-aorta
  --aorta-topic "${AORTA_TOPIC}"
  --aorta-group "${AORTA_GROUP}"
  --lowcmd-layout "${LOWCMD_LAYOUT}"
  --src-human "${SRC_HUMAN}"
)

if [[ -n "${RECORD_PATH}" ]]; then
  cmd+=(--record-path "${RECORD_PATH}")
fi
cmd+=("${EXTRA_ARGS[@]}")

log "AORTA_GROUP=${AORTA_GROUP}"
if [[ -n "${DISPLAY:-}" ]]; then
  log "DISPLAY=${DISPLAY}"
fi
if [[ -n "${XAUTHORITY:-}" ]]; then
  log "XAUTHORITY=${XAUTHORITY}"
fi
if [[ -n "${ZENOH_SESSION_CONFIG_URI:-}" ]]; then
  log "ZENOH_SESSION_CONFIG_URI=${ZENOH_SESSION_CONFIG_URI}"
fi
if [[ -n "${ROBOT_SN}" ]]; then
  log "robot namespace=${ROBOT_SN}"
fi
if [[ -n "${ROBOT_ENDPOINT}" ]]; then
  log "robot endpoint=${ROBOT_ENDPOINT}"
fi
log "starting live Pico retargeting with Aorta publishing"
printf '[run-pico-aorta] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" -eq 0 ]]; then
  exec "${cmd[@]}"
fi
