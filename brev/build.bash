#! /bin/bash
#
# Build the tutorial image, test it the way Brev runs it, and optionally push it.
#
# Usage:
#   brev/build.bash [--push] [--latest] [--chain] [--owner <ghcr-owner>] [--tag <tag>]
#
# The image is built from brev/dockerfile and tagged ghcr.io/<owner>/sc2026-tutorial:<tag>
# (owner pyhpc and tag git-<sha7> by default). It is then tested from a fresh volume as the
# tutorial user, which is how a Brev instance runs it: the package tests always, every notebook
# with --chain, and the Compose stack is started from it until JupyterLab and both Nsight
# streamers answer. With --push the image goes to the registry together with its build cache,
# the latest tag moves too with --latest, and the images of earlier builds are removed from the
# host. build/summary.md records the tags, the digest, the size and the free disk.
#
# Options:
#   --push          Push the image and the build cache to the registry (docker login first)
#   --latest        Also tag the image latest (pushed with --push)
#   --chain         Run every notebook in the image after the package tests
#   --owner OWNER   Registry namespace below ghcr.io (default: pyhpc; CI passes the repository owner)
#   --tag TAG       Image tag (default: git-<sha7> of HEAD)

set -eu

REPO_ROOT=$(cd "$(dirname "${0}")/.."; pwd -P)

PUSH=false
LATEST=false
CHAIN=false
OWNER=pyhpc
TAG=""
while [ $# -gt 0 ]; do
    case "${1}" in
        --push)   PUSH=true ;;
        --latest) LATEST=true ;;
        --chain)  CHAIN=true ;;
        --owner)  OWNER=${2:?--owner needs a value}; shift ;;
        --tag)    TAG=${2:?--tag needs a value}; shift ;;
        *) echo "Usage: $0 [--push] [--latest] [--chain] [--owner <ghcr-owner>] [--tag <tag>]" >&2; exit 1 ;;
    esac
    shift
done
if [ -z "${TAG}" ]; then
    TAG="git-$(git -C "${REPO_ROOT}" rev-parse --short=7 HEAD)"
fi

IMAGE_REPO="ghcr.io/${OWNER,,}/sc2026-tutorial"
IMAGE="${IMAGE_REPO}:${TAG}"
CACHE="type=registry,ref=${IMAGE_REPO}:buildcache"
BUILDER=sc2026-tutorial
export BUILDKIT_PROGRESS="${BUILDKIT_PROGRESS:-plain}"

# The default Docker builder cannot export a registry build cache; a docker-container builder can.
# brev/buildkitd.toml keeps the cached layers, the LLVM build among them, on the disk between builds.
if ! docker buildx inspect "${BUILDER}" >/dev/null 2>&1; then
    docker buildx create --name "${BUILDER}" --driver docker-container \
        --buildkitd-config "${REPO_ROOT}/brev/buildkitd.toml" --bootstrap >/dev/null
fi

# One build: the image is loaded into the daemon for the tests and the push, and the layers go
# to the registry cache whether or not the tests pass.
CACHE_TO=()
if [ "${PUSH}" = true ]; then
    CACHE_TO=(--cache-to "${CACHE},mode=max,image-manifest=true,compression=zstd,compression-level=3")
fi
echo "Building ${IMAGE}"
docker buildx build --builder "${BUILDER}" -f "${REPO_ROOT}/brev/dockerfile" -t "${IMAGE}" \
    --cache-from "${CACHE}" "${CACHE_TO[@]}" --provenance=false --load "${REPO_ROOT}"
SIZE_GB=$(docker image inspect --format '{{.Size}}' "${IMAGE}" | awk '{printf "%.1f", $1 / 1e9}')
echo "Built ${IMAGE}: ${SIZE_GB} GB uncompressed"

# The tests run in a jupyter container of the Compose stack, on a volume seeded from this image.
echo "Testing ${IMAGE}"
export SC26_IMAGE="${IMAGE}"
compose() {
    docker compose -f "${REPO_ROOT}/brev/docker-compose.yml" -f "${REPO_ROOT}/brev/docker-compose.test.yml" \
        -p sc2026-tutorial-test "$@"
}
compose down -v --remove-orphans >/dev/null 2>&1 || true
if [ "${CHAIN}" = true ]; then
    compose run --rm --no-deps -T jupyter test
else
    compose run --rm --no-deps -T jupyter test test/test_packages.py
fi
compose down -v --remove-orphans >/dev/null

# The stack as Brev starts it: a fresh volume seeded from the image, JupyterLab and the two Nsight
# streamers up and answering. docker-compose.test.yml moves the published ports to free host ports.
answers() {
    local service=${1} port=${2} url code
    url="http://$(compose port "${service}" "${port}")/"
    for _ in $(seq 1 90); do
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "${url}" || true)
        case "${code}" in 200|301|302|401|403) echo "${service} answers on ${url} (${code})"; return 0 ;; esac
        sleep 2
    done
    echo "Error: ${service} does not answer on ${url}" >&2
    compose ps >&2
    compose logs --tail 50 "${service}" >&2
    return 1
}
echo "Starting the stack from ${IMAGE}"
compose up -d --quiet-pull
answers jupyter 8888
answers nsys 8080
answers ncu 8081
compose down -v --remove-orphans >/dev/null

TAGS=("${IMAGE}")
if [ "${LATEST}" = true ]; then
    TAGS+=("${IMAGE_REPO}:latest")
    docker tag "${IMAGE}" "${IMAGE_REPO}:latest"
fi
DIGEST=""
if [ "${PUSH}" = true ]; then
    echo "Pushing ${TAGS[*]}"
    for tag in "${TAGS[@]}"; do
        docker push "${tag}"
    done
    DIGEST=$(docker buildx imagetools inspect "${IMAGE}" --format '{{.Manifest.Digest}}')
    echo "Pushed ${IMAGE_REPO}@${DIGEST}"
    # Earlier builds are in the registry; on the host they only fill the disk.
    docker images --format '{{.Repository}}:{{.Tag}}' "${IMAGE_REPO}" | grep -vxF -e "${IMAGE}" -e "${IMAGE_REPO}:latest" \
        | xargs -r docker image rm >/dev/null || true
    docker image prune -f >/dev/null
fi
DISK_FREE=$(df -h --output=avail / | tail -n 1 | tr -d ' ')

mkdir -p "${REPO_ROOT}/build"
{
    echo "## Image"
    echo
    echo "| Field | Value |"
    echo "|---|---|"
    echo "| Tags | ${TAGS[*]} |"
    echo "| Size | ${SIZE_GB} GB uncompressed |"
    echo "| Pushed | ${PUSH} |"
    if [ -n "${DIGEST}" ]; then
        echo "| Digest | \`${IMAGE_REPO}@${DIGEST}\` |"
    fi
    echo "| Notebooks run | ${CHAIN} |"
    echo "| Disk free | ${DISK_FREE} |"
} > "${REPO_ROOT}/build/summary.md"
echo "Wrote build/summary.md"
