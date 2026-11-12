#!/usr/bin/env bash
# Publish release tags only after both native candidates pass their smoke tests.
set -euo pipefail

read -r image version moving_tag amd64_digest arm64_digest < <(python - <<'PY'
import json
from pathlib import Path

with open("release-image.json", encoding="utf-8") as handle:
    data = json.load(handle)
amd64_digest = Path("image-amd64.digest").read_text(encoding="utf-8").strip()
arm64_digest = Path("image-arm64.digest").read_text(encoding="utf-8").strip()
print(data["image"], data["version"], data["moving_tag"], amd64_digest, arm64_digest)
PY
)
if [[ -z "$image" || -z "$version" || -z "$moving_tag" || -z "$amd64_digest" || -z "$arm64_digest" ]]; then
    echo "Both verified platform digests are required before publishing release tags" >&2
    exit 1
fi

actor="${GITHUB_ACTOR:?GITHUB_ACTOR is required}"
token="${PROWL_GHCR_TOKEN:?PROWL_GHCR_TOKEN is required to publish the image}"
registry_and_path="${image#ghcr.io/}"
owner="${PROWL_GHCR_OWNER:-${registry_and_path%%/*}}"
package="${PROWL_GHCR_PACKAGE:-${registry_and_path#*/}}"
owner_type="${PROWL_GHCR_OWNER_TYPE:-user}"
package_args=(--owner "$owner" --package "$package" --owner-type "$owner_type")

GITHUB_TOKEN="$token" python .github/scripts/ghcr_package.py "${package_args[@]}" verify-private
printf '%s' "$token" | docker login ghcr.io --username "$actor" --password-stdin
trap 'docker logout ghcr.io >/dev/null 2>&1 || true' EXIT

docker buildx imagetools create \
    --tag "${image}:${version}" \
    --tag "${image}:${moving_tag}" \
    "${image}@${amd64_digest}" \
    "${image}@${arm64_digest}"

echo "Published linux/amd64 and linux/arm64 as ${image}:${version} and ${image}:${moving_tag}"

# Resolve retained indexes before pruning their untagged platform manifests.
retain_args=(--current-tag "$version" --keep-prerelease 3 --keep-stable 2)
kept_tags="$(GITHUB_TOKEN="$token" python .github/scripts/ghcr_package.py "${package_args[@]}" kept-tags \
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

    if [[ "$resolved" == true ]]; then
        retain_args+=(--prune-untagged)
    fi
    if [[ ${#protected[@]} -gt 0 ]]; then
        retain_args+=("${protected[@]}")
    fi
fi

# Retention failures do not undo an already-published multi-platform image.
if ! GITHUB_TOKEN="$token" python .github/scripts/ghcr_package.py "${package_args[@]}" retain \
    "${retain_args[@]}" --execute; then
    echo "::warning title=GHCR retention failed::Could not prune old ${image} versions; the next release retries." >&2
fi
