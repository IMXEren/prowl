#!/usr/bin/env bash
# Build the arm64 image natively and merge it into the released manifest.
#
# The amd64 image is published by semantic-release on an amd64 runner; this runs
# on a native arm64 runner so no emulation is involved, then folds both platform
# manifests into one index under the tags that release already created.
set -euo pipefail

descriptor="release-image.json"
if [[ ! -f "$descriptor" ]]; then
    echo "No released image descriptor; this run published nothing to merge."
    exit 0
fi

read -r image version moving_tag amd64_digest < <(python - "$descriptor" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
print(data["image"], data["version"], data["moving_tag"], data["amd64_digest"])
PY
)

if [[ -z "$image" || -z "$version" || -z "$moving_tag" || -z "$amd64_digest" ]]; then
    echo "Released image descriptor is incomplete: $descriptor" >&2
    exit 1
fi

metadata=""
cleanup() {
    rm -f "$metadata"
    docker logout ghcr.io >/dev/null 2>&1 || true
}
trap cleanup EXIT

actor="${GITHUB_ACTOR:?GITHUB_ACTOR is required}"
token="${PROWL_GHCR_TOKEN:?PROWL_GHCR_TOKEN is required to publish the image}"

# Build exactly the commit that was released, not whatever the branch has moved on to.
git checkout --quiet "v${version}"

printf '%s' "$token" | docker login ghcr.io --username "$actor" --password-stdin

# Attestations are disabled so the pushed digest is the plain arm64 manifest.
# An attestation would make the digest an index, and merging an index into an
# index nests it and surfaces as a bogus unknown/unknown platform.
metadata="$(mktemp)"
PROWL_FONT_GITHUB_TOKEN="${PROWL_FONT_GITHUB_TOKEN:-}" \
    bash .github/scripts/build-image.sh \
    --platform linux/arm64 \
    --provenance=false \
    --sbom=false \
    --output "type=image,name=${image},push=true,push-by-digest=true,name-canonical=true" \
    --label "org.opencontainers.image.source=${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-IMXEren/prowl}" \
    --label "org.opencontainers.image.version=${version}" \
    --metadata-file "$metadata"

arm64_digest="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["containerimage.digest"])' "$metadata")"
if [[ -z "$arm64_digest" ]]; then
    echo "Could not read the arm64 image digest from the build metadata" >&2
    exit 1
fi

# imagetools resolves sources through a registry, so each digest needs its repository.
docker buildx imagetools create \
    --tag "${image}:${version}" \
    --tag "${image}:${moving_tag}" \
    "${image}@${amd64_digest}" \
    "${image}@${arm64_digest}"

echo "Merged linux/amd64 ${amd64_digest} and linux/arm64 ${arm64_digest} into ${image}:${version} and ${image}:${moving_tag}"
