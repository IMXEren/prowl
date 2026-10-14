#!/usr/bin/env bash
# Record image metadata after semantic-release creates the release tag.
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

moving_tag="latest"
if [[ "$version" == *-* ]]; then
    moving_tag="dev"
fi

python - "$version" "$moving_tag" "$image" <<'PY'
import json
import sys

version, moving_tag, image = sys.argv[1:4]
with open("release-image.json", "w", encoding="utf-8") as handle:
    json.dump({"image": image, "version": version, "moving_tag": moving_tag}, handle, indent=2)
    handle.write("\n")
PY
