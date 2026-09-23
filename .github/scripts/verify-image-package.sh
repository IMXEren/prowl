#!/usr/bin/env bash
# Refuse release publication if an existing destination package is not private.
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

python .github/scripts/ghcr_package.py \
    --owner "$owner" \
    --package "$package" \
    --owner-type "$owner_type" \
    verify-private \
    --allow-missing
