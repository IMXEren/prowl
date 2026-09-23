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

# Attestations are disabled so the pushed manifest is the plain linux/amd64 image
# rather than an index. The arm64 job merges this digest into a platform index,
# and merging an index into an index nests it into a bogus unknown/unknown entry.
PROWL_FONT_GITHUB_TOKEN="${PROWL_FONT_GITHUB_TOKEN:-}" \
    bash .github/scripts/build-image.sh \
    --load \
    --provenance=false \
    --sbom=false \
    --label "org.opencontainers.image.source=${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-IMXEren/prowl}" \
    --label "org.opencontainers.image.version=${version}" \
    --tag "$build_tag"

# Smoke-test before anything reaches the registry, so a broken image is never
# published.
docker run --rm --entrypoint python "$build_tag" -c "import prowl"
if [[ -n "${PROWL_FONT_GITHUB_TOKEN:-}" ]]; then
    docker run --rm --entrypoint sh "$build_tag" -c \
        'test -n "$(find /usr/share/fonts/windows -type f -print -quit)"'
fi

docker tag "$build_tag" "${image}:${version}"
docker tag "$build_tag" "${image}:${moving_tag}"
docker push "${image}:${version}"
docker push "${image}:${moving_tag}"

# Record what was published so the arm64 job can merge into the same manifest
# instead of guessing which tags this run created. The name is deliberately not
# hidden: upload-artifact excludes dotfiles unless include-hidden-files is set.
amd64_digest="$(docker buildx imagetools inspect "${image}:${version}" --format '{{.Manifest.Digest}}')"
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
print(f"Recorded {image}:{version} ({digest}) for the arm64 merge")
PY

# A public container package can never be made private again, so this check is
# the last line of defence and has to name the remediation explicitly.
if ! python .github/scripts/ghcr_package.py "${visibility_args[@]}" verify-private; then
    echo "::error title=Published image is not private::Delete ${image} and re-create it as private; GitHub cannot make a public package private again." >&2
    exit 1
fi

# A multi-platform image is one tagged index plus one untagged version per
# platform manifest, so retention must know which children a retained image
# still references before it removes anything untagged.
retain_args=(--current-tag "$version" --keep-prerelease 3 --keep-stable 2)
kept_tags="$(python .github/scripts/ghcr_package.py "${visibility_args[@]}" kept-tags \
    --current-tag "$version" --keep-prerelease 3 --keep-stable 2)" || kept_tags=""

if [[ -n "$kept_tags" ]]; then
    protected=()
    resolved=true
    while IFS= read -r kept_tag; do
        [[ -n "$kept_tag" ]] || continue
        if children="$(docker buildx imagetools inspect --raw "${image}:${kept_tag}" 2>/dev/null \
            | python -c 'import json,sys; print("\n".join(m["digest"] for m in json.load(sys.stdin).get("manifests", [])))')"; then
            while IFS= read -r child; do
                [[ -n "$child" ]] || continue
                protected+=(--protect-digest "$child")
            done <<<"$children"
        else
            resolved=false
            echo "::warning title=Retention could not resolve platform manifests::Could not read the platform manifests of ${image}:${kept_tag}; untagged manifests are left in place." >&2
        fi
    done <<<"$kept_tags"

    # Only prune untagged manifests when every retained parent was readable, so
    # a lookup failure can never orphan a manifest a retained image still needs.
    if [[ "$resolved" == true ]]; then
        retain_args+=(--prune-untagged)
    fi
    if [[ ${#protected[@]} -gt 0 ]]; then
        retain_args+=("${protected[@]}")
    fi
fi

# Retention is cumulative housekeeping, so a failure must not mark an already
# published release as failed. It must still be impossible to miss, and the
# next release prunes whatever this run left behind.
if ! python .github/scripts/ghcr_package.py "${visibility_args[@]}" retain "${retain_args[@]}" --execute; then
    echo "::warning title=GHCR retention failed::Could not prune old ${image} versions; the next release retries." >&2
fi
