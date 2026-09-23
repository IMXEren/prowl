#!/usr/bin/env bash
# Refuse release publication unless the destination package is private.
#
# The image is published with a personal access token, which creates an unlinked
# package, and an unlinked package is private. A package that does not exist yet
# is therefore allowed: the publish step creates it private. A package that
# already exists must report private, because a container package cannot be
# relied on to be made private again once it is public.
set -euo pipefail

image="${PROWL_IMAGE_NAME:-ghcr.io/imxeren/prowl}"
registry_and_path="${image#ghcr.io/}"
if [[ "$registry_and_path" == "$image" || "$registry_and_path" != */* ]]; then
    echo "PROWL_IMAGE_NAME must be a ghcr.io owner/package reference" >&2
    exit 2
fi
owner="${PROWL_GHCR_OWNER:-${registry_and_path%%/*}}"
package="${PROWL_GHCR_PACKAGE:-${registry_and_path#*/}}"
owner_type="${PROWL_GHCR_OWNER_TYPE:-user}"

if GITHUB_TOKEN="${PROWL_GHCR_TOKEN:?PROWL_GHCR_TOKEN is required}" \
    python .github/scripts/ghcr_package.py \
    --owner "$owner" \
    --package "$package" \
    --owner-type "$owner_type" \
    verify-private \
    --allow-missing; then
    exit 0
fi

cat >&2 <<EOF
::error title=Refusing to publish::${image} exists but is not private.

A container package cannot be relied on to be made private again once it is public, so publishing to it is refused. Set the package to Private in its settings, or delete it and let the next release create it unlinked and private.
EOF
exit 1
