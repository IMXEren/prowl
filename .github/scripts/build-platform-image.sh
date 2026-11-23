#!/usr/bin/env bash
# Build and smoke-test one native candidate without publishing release tags.
# Use a PAT, not GITHUB_TOKEN, so a new GHCR package is unlinked and private.
set -euo pipefail

arch="${1:?native architecture is required}"
case "$arch" in
    amd64|arm64) ;;
    *) echo "Unsupported native architecture: $arch" >&2; exit 2 ;;
esac

read -r image version < <(python - <<'PY'
import json

with open("release-image.json", encoding="utf-8") as handle:
    data = json.load(handle)
print(data["image"], data["version"])
PY
)

# Build the released commit, not the latest branch head.
git checkout --quiet "v${version}"
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

# Plain platform manifests can be combined without nested attestation indexes.
PROWL_FONT_GITHUB_TOKEN="${PROWL_FONT_GITHUB_TOKEN:-}" \
    bash .github/scripts/build-image.sh \
    --platform "linux/${arch}" \
    --provenance=false \
    --sbom=false \
    --output "type=image,name=${image},push=true,push-by-digest=true,name-canonical=true" \
    --label "org.opencontainers.image.source=${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-IMXEren/prowl}" \
    --label "org.opencontainers.image.version=${version}" \
    --metadata-file "$metadata"

digest="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["containerimage.digest"])' "$metadata")"
if [[ -z "$digest" ]]; then
    echo "Could not read the ${arch} image digest from the build metadata" >&2
    exit 1
fi
candidate="${image}@${digest}"
docker run --rm --entrypoint python "$candidate" -c "import prowl"
if [[ -n "${PROWL_FONT_GITHUB_TOKEN:-}" ]]; then
    docker run --rm --entrypoint sh "$candidate" -c \
        'test -n "$(find /usr/share/fonts/windows -type f -print -quit)"'
fi

if ! GITHUB_TOKEN="$token" python .github/scripts/ghcr_package.py "${package_args[@]}" verify-private; then
    echo "::error title=Published image is not private::Delete ${image} and re-create it as private; GitHub cannot make a public package private again." >&2
    exit 1
fi

# Export a digest only after its candidate passes every check.
printf '%s\n' "$digest" > "image-${arch}.digest"
echo "Recorded ${arch} candidate ${candidate}; release tags unchanged"
