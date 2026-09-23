#!/usr/bin/env bash
# Publish the private GHCR image for a semantic-release version, then retain it.
set -euo pipefail

version="${1:?semantic-release version is required}"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
    echo "Invalid semantic-release version: $version" >&2
    exit 2
fi

image="${PROWL_IMAGE_NAME:-ghcr.io/imxeren/prowl}"
registry_and_path="${image#ghcr.io/}"
if [[ "$registry_and_path" == "$image" || "$registry_and_path" != */* ]]; then
    echo "PROWL_IMAGE_NAME must be a ghcr.io owner/package reference" >&2
    exit 2
fi
owner="${PROWL_GHCR_OWNER:-${registry_and_path%%/*}}"
package="${PROWL_GHCR_PACKAGE:-${registry_and_path#*/}}"
owner_type="${PROWL_GHCR_OWNER_TYPE:-user}"
actor="${GITHUB_ACTOR:?GITHUB_ACTOR is required}"
token="${GITHUB_TOKEN:?GITHUB_TOKEN is required}"

visibility_args=(--owner "$owner" --package "$package" --owner-type "$owner_type")
python .github/scripts/ghcr_package.py "${visibility_args[@]}" verify-private --allow-missing

printf '%s' "$token" | docker login ghcr.io --username "$actor" --password-stdin
trap 'docker logout ghcr.io >/dev/null 2>&1 || true' EXIT

moving_tag="latest"
if [[ "$version" == *-* ]]; then
    moving_tag="dev"
fi

build_tag="prowl:release-${version}"

PROWL_FONT_GITHUB_TOKEN="${PROWL_FONT_GITHUB_TOKEN:-}" \
    bash .github/scripts/build-image.sh \
    --load \
    --label "org.opencontainers.image.source=${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-IMXEren/prowl}" \
    --label "org.opencontainers.image.version=${version}" \
    --tag "$build_tag"

# Smoke-test before anything reaches the registry, so a broken image is never
# published. Pushing a locally built image also avoids the provenance/SBOM
# attestation manifests that a buildx push would add as untagged versions.
docker run --rm --entrypoint python "$build_tag" -c "import prowl"
if [[ -n "${PROWL_FONT_GITHUB_TOKEN:-}" ]]; then
    docker run --rm --entrypoint sh "$build_tag" -c \
        'test -n "$(find /usr/share/fonts/windows -type f -print -quit)"'
fi

docker tag "$build_tag" "${image}:${version}"
docker tag "$build_tag" "${image}:${moving_tag}"
docker push "${image}:${version}"
docker push "${image}:${moving_tag}"

# A public container package can never be made private again, so this check is
# the last line of defence and has to name the remediation explicitly.
if ! python .github/scripts/ghcr_package.py "${visibility_args[@]}" verify-private; then
    echo "::error title=Published image is not private::Delete ${image} and re-create it as private; GitHub cannot make a public package private again." >&2
    exit 1
fi

# Retention is cumulative housekeeping, so a failure must not mark an already
# published release as failed. It must still be impossible to miss, and the
# next release prunes whatever this run left behind.
if ! python .github/scripts/ghcr_package.py "${visibility_args[@]}" retain \
    --current-tag "$version" \
    --keep-prerelease 3 \
    --keep-stable 2 \
    --execute; then
    echo "::warning title=GHCR retention failed::Could not prune old ${image} versions; the next release retries." >&2
fi
