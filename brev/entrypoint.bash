#! /bin/bash
# Entrypoint of the tutorial containers. It runs as root, creates the tutorial user from
# TUTORIAL_UID and TUTORIAL_GID (1000 by default) and runs the requested service as that user.
#
#   entrypoint.bash jupyter [default-url]   JupyterLab on port 8888 (the image's default)
#   entrypoint.bash test [pytest args]      the test suite; without arguments, every test file
#   entrypoint.bash shell                   an interactive shell in the notebooks directory
#   entrypoint.bash nsight                  the Nsight Systems or Nsight Compute streamer, from
#                                           its own image (see docker-compose.yml)

set -euo pipefail

ROOT=/sc2026-tutorial
SERVICE=${1:-jupyter}
shift || true

if [ "$(id -u)" = 0 ]; then
    uid=${TUTORIAL_UID:-1000}
    gid=${TUTORIAL_GID:-1000}
    TUTORIAL_USER=$(getent passwd "${uid}" | cut -d: -f1 || true)
    if [ -z "${TUTORIAL_USER}" ]; then
        TUTORIAL_USER=student
        getent group "${gid}" >/dev/null || groupadd --gid "${gid}" "${TUTORIAL_USER}"
        useradd --uid "${uid}" --gid "${gid}" --create-home --shell /bin/bash "${TUTORIAL_USER}"
    fi
    # The Nsight streamer's entrypoint runs chown "$USER:$USER", so a group of that name must exist.
    getent group "${TUTORIAL_USER}" >/dev/null || groupadd "${TUTORIAL_USER}"
    # Profiling as a non-root user needs these; they fail harmlessly where the container cannot set them.
    sysctl -w kernel.perf_event_paranoid=0 >/dev/null 2>&1 || true
    sysctl -w kernel.kptr_restrict=0 >/dev/null 2>&1 || true
    AS_USER=(gosu "${TUTORIAL_USER}")
else
    TUTORIAL_USER=$(id -un)
    AS_USER=()
fi
export USER=${TUTORIAL_USER}
HOME=$(getent passwd "${TUTORIAL_USER}" | cut -d: -f6)
export HOME

case "${SERVICE}" in
jupyter)
    # jupyter-server-proxy reaches only localhost; forward the streamer ports of the other
    # containers there (as root; the user needs no access to the forwards). A connection fails
    # quietly when its streamer is not running.
    socat TCP-LISTEN:8080,fork,reuseaddr TCP:nsys:8080 &
    socat TCP-LISTEN:8081,fork,reuseaddr TCP:ncu:8081 &
    args=(--config="${ROOT}/brev/jupyter_server_config.py" --ServerApp.preferred_dir="${ROOT}/notebooks")
    if [ -n "${1:-}" ]; then
        args+=(--LabApp.default_url="${1}")
    fi
    cd "${ROOT}/notebooks"
    exec "${AS_USER[@]}" python -m jupyter lab "${args[@]}"
    ;;
test)
    nvidia-smi || { echo "The tests need a GPU and the NVIDIA container runtime." >&2; exit 1; }
    cd "${ROOT}"
    # Package tests first, so a broken image fails before the notebooks run.
    if [ $# -eq 0 ]; then
        set -- test/test_packages.py test/test_notebooks.py test/test_cosmic_web_notebooks.py
    fi
    exec "${AS_USER[@]}" pytest "$@"
    ;;
shell)
    cd "${ROOT}/notebooks"
    exec "${AS_USER[@]}" bash -l
    ;;
nsight)
    # The streamer image switches to $USER itself and builds HOME as /home/$USER.
    if [ ! -e "/home/${TUTORIAL_USER}" ]; then
        ln -s "${HOME}" "/home/${TUTORIAL_USER}"
    fi
    if [ "${HOME}" != /home/nvidia ]; then
        cp /home/nvidia/.bashrc "${HOME}/.bashrc"
        mkdir -p "${HOME}/.config/NVIDIA Corporation"
        cp "/home/nvidia/.config/NVIDIA Corporation/"*.ini "${HOME}/.config/NVIDIA Corporation/" 2>/dev/null || true
        chown -R "${TUTORIAL_USER}:$(id -gn "${TUTORIAL_USER}")" "${HOME}"
    fi
    # Nsight Compute opens its file dialogs on the notebooks.
    mkdir -p "${HOME}/.config/NVIDIA Corporation"
    printf '[CorePlugin.Environment]\nCorePlugin.DefaultDocumentsFolder=%s/notebooks\n' "${ROOT}" \
        > "${HOME}/.config/NVIDIA Corporation/NVIDIA Nsight Compute.ini"
    # No login page: the streamer applies default credentials when these two are unset and none
    # when they are empty. The browser reaches the streamer through its TURN relay, advertised at
    # the instance's public address, with credentials made for this container.
    export WEB_USERNAME=${WEB_USERNAME:-}
    export WEB_PASSWORD=${WEB_PASSWORD:-}
    if ! command -v curl >/dev/null && [ -z "${SELKIES_TURN_HOST:-}" ]; then
        apt-get update -qq && apt-get install -y -qq curl
    fi
    export NVIDIA_DRIVER_CAPABILITIES=all
    export HOST_IP=${HOST_IP:-0.0.0.0}
    export SELKIES_ADDR=${SELKIES_ADDR:-0.0.0.0}
    export SELKIES_TURN_HOST=${SELKIES_TURN_HOST:-$(curl -sSL ifconfig.me)}
    export TURN_USERNAME=${TURN_USERNAME:-$(head -c 24 /dev/urandom | base64 | tr -dc a-zA-Z0-9)}
    export TURN_PASSWORD=${TURN_PASSWORD:-$(head -c 48 /dev/urandom | base64 | tr -dc a-zA-Z0-9)}
    for var in NVIDIA_DRIVER_CAPABILITIES HOST_IP SELKIES_ADDR SELKIES_TURN_HOST TURN_USERNAME TURN_PASSWORD \
               HTTP_PORT TURN_PORT WEB_USERNAME WEB_PASSWORD; do
        if [ -n "${!var+x}" ]; then
            printf 'export %s=%q\n' "${var}" "${!var}" >> "${HOME}/.bashrc"
        fi
    done
    source /setup/entrypoint.sh "$@"
    ;;
*)
    echo "Usage: entrypoint.bash {jupyter [default-url]|test [pytest args]|shell|nsight}" >&2
    exit 1
    ;;
esac
