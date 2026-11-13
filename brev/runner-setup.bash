#! /bin/bash
#
# Turn a fresh Brev GPU instance into the self-hosted GitHub Actions runner that builds the image.
#
# Usage:
#   sudo ./brev/runner-setup.bash <repository-url> <registration-token>
#
# The registration token comes from the repository's Settings > Actions > Runners > New self-hosted
# runner page and expires after an hour. The runner registers with the label sc26-builder, which
# .github/workflows/image.yml selects, and runs as a systemd service under the user "runner", who
# is in the docker group. Docker, Docker Compose, Buildx and the NVIDIA container toolkit must
# already be present, as they are on Brev GPU instances.

set -eu

RUNNER_VERSION=2.338.2
RUNNER_SHA256=baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40
RUNNER_USER=runner
RUNNER_HOME=/opt/actions-runner
RUNNER_LABELS=sc26-builder

if [ $# -ne 2 ]; then
    echo "Usage: sudo $0 <repository-url> <registration-token>" >&2
    exit 1
fi
REPO_URL=${1}
TOKEN=${2}
if [ "$(id -u)" -ne 0 ]; then
    echo "Error: run this script with sudo" >&2
    exit 1
fi

echo "Checking the container tools"
docker info >/dev/null
docker buildx version
docker compose version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! id -u "${RUNNER_USER}" >/dev/null 2>&1; then
    useradd --create-home --shell /bin/bash "${RUNNER_USER}"
fi
usermod -aG docker "${RUNNER_USER}"

echo "Installing actions-runner ${RUNNER_VERSION} in ${RUNNER_HOME}"
mkdir -p "${RUNNER_HOME}"
TARBALL=$(mktemp /tmp/actions-runner.XXXXXX.tar.gz)
curl -fsSL -o "${TARBALL}" \
    "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
echo "${RUNNER_SHA256}  ${TARBALL}" | sha256sum -c -
tar -xzf "${TARBALL}" -C "${RUNNER_HOME}"
rm -f "${TARBALL}"
chown -R "${RUNNER_USER}:${RUNNER_USER}" "${RUNNER_HOME}"
"${RUNNER_HOME}/bin/installdependencies.sh"

echo "Registering with ${REPO_URL}"
cd "${RUNNER_HOME}"
sudo -u "${RUNNER_USER}" ./config.sh --unattended \
    --url "${REPO_URL}" --token "${TOKEN}" \
    --name "$(hostname)" --labels "${RUNNER_LABELS}" --work _work --replace
./svc.sh install "${RUNNER_USER}"
./svc.sh start
./svc.sh status
echo "Runner $(hostname) registered with label ${RUNNER_LABELS}"
