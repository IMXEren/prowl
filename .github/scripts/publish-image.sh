#!/usr/bin/env bash
# Build and smoke-test the amd64 candidate without publishing release tags.
# Use a PAT, not GITHUB_TOKEN, so a new GHCR package is unlinked and private.
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
token="${PROWL_GHCR_TOKEN:?PROWL_GHCR_TOKEN is required to publish the image}"
package_args=(--owner "$owner" --package "$package" --owner-type "$owner_type")

if ! GITHUB_TOKEN="$token" python .github/scripts/ghcr_package.py "${package_args[@]}" \
    verify-private --allow-missing; then
    echo "::error title=Refusing to publish::${image} exists but is not private." >&2
    exit 1
fi

metadata="$(mktemp)"
cleanup() {
    rm -f "$metadata"
    docker logout ghcr.io >/dev/null 2>&1 || true
}
trap cleanup EXIT
printf '%s' "$token" | docker login ghcr.io --username "$actor" --password-stdin

moving_tag="latest"
if [[ "$version" == *-* ]]; then
    moving_tag="dev"
fi

# Plain platform manifests can be combined without nested attestation indexes.
PROWL_FONT_GITHUB_TOKEN="${PROWL_FONT_GITHUB_TOKEN:-}" \
    bash .github/scripts/build-image.sh \
    --platform linux/amd64 \
    --provenance=false \
    --sbom=false \
    --output "type=image,name=${image},push=true,push-by-digest=true,name-canonical=true" \
    --label "org.opencontainers.image.source=${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-IMXEren/prowl}" \
    --label "org.opencontainers.image.version=${version}" \
    --metadata-file "$metadata"

amd64_digest="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["containerimage.digest"])' "$metadata")"
if [[ -z "$amd64_digest" ]]; then
    echo "Could not read the amd64 image digest from the build metadata" >&2
    exit 1
fi
candidate="${image}@${amd64_digest}"
docker run --rm --entrypoint python "$candidate" -c "import prowl"
if [[ -n "${PROWL_FONT_GITHUB_TOKEN:-}" ]]; then
    docker run --rm --entrypoint sh "$candidate" -c \
        'test -n "$(find /usr/share/fonts/windows -type f -print -quit)"'
fi

if ! GITHUB_TOKEN="$token" python .github/scripts/ghcr_package.py "${package_args[@]}" verify-private; then
    echo "::error title=Published image is not private::Delete ${image} and re-create it as private; GitHub cannot make a public package private again." >&2
    exit 1
fi

# The descriptor exists only after this candidate passes its checks.
python - "$version" "$moving_tag" "$image" "$amd64_digest" <<'PY'
import json
import sys

version, moving_tag, image, digest = sys.argv[1:5]
descriptor = {
    "image": image,
    "version": version,
    "moving_tag": moving_tag,
    "amd64_digest": digest,
}
with open("release-image.json", "w", encoding="utf-8") as handle:
    json.dump(descriptor, handle, indent=2)
    handle.write("\n")
print(f"Recorded amd64 candidate {image}@{digest} for {version}; release tags unchanged")
PY
