"""Verify private GHCR visibility and safely retain recent image versions."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

_NOT_FOUND = 404
_MAX_VISIBILITY_ATTEMPTS = 5
_SEMVER = re.compile(
    r"^(?:v)?(?P<major>0|[1-9]\d*)\."
    r"(?P<minor>0|[1-9]\d*)\."
    r"(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


class PackageApiError(RuntimeError):
    """A GitHub Packages request failed."""


class RetentionSafetyError(RuntimeError):
    """Retention state was ambiguous, so no deletion may proceed."""


@dataclass(frozen=True)
class PackageVersion:
    """The package-version fields needed for retention decisions.

    ``digest`` is the manifest digest GitHub reports as the version ``name``. For
    a multi-platform image the tagged version is the index and each platform
    manifest is an untagged version of its own, so the digest is what ties a
    child back to the index that references it.
    """

    version_id: int
    created_at: str
    tags: tuple[str, ...]
    digest: str = ""


def _request(path: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> tuple[Any, dict[str, str]]:
    """Call the configured GitHub REST API and decode its JSON response."""
    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        msg = "GITHUB_TOKEN is required"
        raise PackageApiError(msg)
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(
        f"{api_url}{path}",
        data=body,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "prowl-release",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read()
            result = json.loads(raw) if raw else None
            return result, {key.lower(): value for key, value in response.headers.items()}
    except HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:1000]
        msg = f"GitHub Packages API {method} {path} failed with HTTP {exc.code}: {detail}"
        error = PackageApiError(msg)
        error.status = exc.code  # type: ignore[attr-defined]
        raise error from exc


def _owner_path(owner: str, owner_type: str) -> str:
    """Return the REST path prefix for a user or organization package owner."""
    encoded_owner = quote(owner, safe="")
    return f"/users/{encoded_owner}" if owner_type == "user" else f"/orgs/{encoded_owner}"


def verify_private(owner: str, package: str, owner_type: str, *, allow_missing: bool = False) -> None:
    """Require the package to be private, optionally tolerating a missing package.

    A missing package is only tolerable because the image is published with a
    personal access token, which creates an unlinked package that is private.
    """
    package_name = quote(package, safe="")
    get_path = f"{_owner_path(owner, owner_type)}/packages/container/{package_name}"
    try:
        details, _ = _request(get_path)
    except PackageApiError as exc:
        if allow_missing and getattr(exc, "status", None) == _NOT_FOUND:
            print("GHCR package does not exist yet; a token push creates it unlinked and private")
            return
        raise
    if details.get("visibility") != "private":
        msg = f"GHCR package visibility is {details.get('visibility')!r}, not 'private'; refusing publication"
        raise PackageApiError(msg)
    print("Verified GHCR package visibility: private")


def _next_link(link_header: str | None) -> str | None:
    """Extract the path and query for the next GitHub API page."""
    if not link_header:
        return None
    for item in link_header.split(","):
        target, *parameters = item.split(";")
        if any(parameter.strip() == 'rel="next"' for parameter in parameters):
            url = target.strip().strip("<>")
            api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
            if not url.startswith(f"{api_url}/"):
                msg = f"Refusing an unexpected pagination URL: {url}"
                raise PackageApiError(msg)
            return url[len(api_url) :]
    return None


def list_versions(owner: str, package: str, owner_type: str) -> list[PackageVersion]:
    """List all container package versions, following GitHub pagination."""
    package_name = quote(package, safe="")
    path = f"{_owner_path(owner, owner_type)}/packages/container/{package_name}/versions?per_page=100"
    versions: list[PackageVersion] = []
    while path:
        page, headers = _request(path)
        if not isinstance(page, list):
            msg = "GitHub Packages versions response was not a list"
            raise PackageApiError(msg)
        for item in page:
            tags = item.get("metadata", {}).get("container", {}).get("tags", [])
            versions.append(
                PackageVersion(
                    version_id=int(item["id"]),
                    created_at=str(item["created_at"]),
                    tags=tuple(str(tag) for tag in tags),
                    digest=str(item.get("name") or ""),
                )
            )
        path = _next_link(headers.get("link"))
    return versions


def classify_version(version: PackageVersion) -> str | None:
    """Classify one package version once, regardless of its number of tags."""
    kinds = {
        "prerelease" if match.group("prerelease") else "stable"
        for tag in version.tags
        if (match := _SEMVER.fullmatch(tag))
    }
    if len(kinds) > 1:
        msg = f"Package version {version.version_id} mixes stable and prerelease semantic-version tags"
        raise RetentionSafetyError(msg)
    return next(iter(kinds), None)


def retention_plan(
    versions: list[PackageVersion],
    current_tag: str,
    *,
    keep_prerelease: int,
    keep_stable: int,
) -> tuple[set[int], list[PackageVersion]]:
    """Return protected IDs and oldest-first versions safe to delete."""
    if keep_prerelease < 1 or keep_stable < 1:
        msg = "Retention counts must each be at least one"
        raise RetentionSafetyError(msg)
    current = [version for version in versions if current_tag in version.tags]
    if len(current) != 1:
        msg = f"Expected exactly one current package version tagged {current_tag!r}; found {len(current)}"
        raise RetentionSafetyError(msg)
    current_id = current[0].version_id

    buckets: dict[str, list[PackageVersion]] = {"prerelease": [], "stable": []}
    for version in versions:
        kind = classify_version(version)
        if kind is not None:
            buckets[kind].append(version)

    current_kind = classify_version(current[0])
    if current_kind is None:
        msg = "Current manifest does not have a semantic-version tag"
        raise RetentionSafetyError(msg)
    current_keep_count = keep_prerelease if current_kind == "prerelease" else keep_stable
    current_bucket = sorted(
        buckets[current_kind],
        key=lambda version: (version.created_at, version.version_id),
        reverse=True,
    )
    if current_id not in {version.version_id for version in current_bucket[:current_keep_count]}:
        msg = "Current manifest is unexpectedly outside its retention window; refusing all deletion"
        raise RetentionSafetyError(msg)

    protected: set[int] = {current_id}
    deletions: list[PackageVersion] = []
    for kind, keep_count in (("prerelease", keep_prerelease), ("stable", keep_stable)):
        newest_first = sorted(
            buckets[kind],
            key=lambda version: (version.created_at, version.version_id),
            reverse=True,
        )
        retained = newest_first[:keep_count]
        protected.update(version.version_id for version in retained)
        deletions.extend(newest_first[keep_count:])

    deletions = [version for version in deletions if version.version_id not in protected]
    deletions.sort(key=lambda version: (version.created_at, version.version_id))
    return protected, deletions


def retained_semver_tags(
    versions: list[PackageVersion],
    current_tag: str,
    *,
    keep_prerelease: int,
    keep_stable: int,
) -> list[str]:
    """Return the semantic-version tags of every manifest retention keeps.

    The caller resolves these against the registry to learn which platform
    manifests a retained image still needs.
    """
    protected, _ = retention_plan(
        versions,
        current_tag,
        keep_prerelease=keep_prerelease,
        keep_stable=keep_stable,
    )
    tags: set[str] = set()
    for version in versions:
        if version.version_id in protected:
            tags.update(tag for tag in version.tags if _SEMVER.fullmatch(tag))
    return sorted(tags)


def orphaned_platform_manifests(
    versions: list[PackageVersion],
    protected_digests: frozenset[str],
) -> list[PackageVersion]:
    """Return untagged platform manifests that no retained image references.

    A version is only considered orphaned when it has no tags, reports a digest,
    and that digest is absent from every digest a retained manifest references.
    An unknown digest is never treated as orphaned.
    """
    orphans = [
        version
        for version in versions
        if not version.tags and version.digest and version.digest not in protected_digests
    ]
    orphans.sort(key=lambda version: (version.created_at, version.version_id))
    return orphans


def _delete_version(  # noqa: PLR0913
    owner: str,
    owner_type: str,
    package: str,
    version: PackageVersion,
    description: str,
    *,
    execute: bool,
    mode: str,
) -> None:
    """Report one planned deletion and perform it when executing."""
    print(f"{mode} {description} {version.version_id}")
    if not execute:
        return
    package_name = quote(package, safe="")
    path = f"{_owner_path(owner, owner_type)}/packages/container/{package_name}/versions/{version.version_id}"
    _request(path, method="DELETE")


def _await_current_version(owner: str, package: str, owner_type: str, current_tag: str) -> list[PackageVersion]:
    """List versions, retrying until the manifest this release just pushed appears."""
    for attempt in range(_MAX_VISIBILITY_ATTEMPTS):
        versions = list_versions(owner, package, owner_type)
        if any(current_tag in version.tags for version in versions):
            return versions
        if attempt == _MAX_VISIBILITY_ATTEMPTS - 1:
            msg = f"Current package version tagged {current_tag!r} did not become visible"
            raise RetentionSafetyError(msg)
        time.sleep(2**attempt)
    return []


def retain(  # noqa: PLR0913
    owner: str,
    package: str,
    owner_type: str,
    current_tag: str,
    *,
    keep_prerelease: int,
    keep_stable: int,
    execute: bool,
    prune_untagged: bool = False,
    protected_digests: frozenset[str] = frozenset(),
) -> None:
    """Plan and optionally execute bounded package-version retention."""
    versions = _await_current_version(owner, package, owner_type, current_tag)
    protected, deletions = retention_plan(
        versions,
        current_tag,
        keep_prerelease=keep_prerelease,
        keep_stable=keep_stable,
    )
    mode = "Deleting" if execute else "Would delete"
    print(f"Retention protects package version IDs: {sorted(protected)}")

    # Tagged manifests go first. Removing an index before its children means a
    # retained index can never be left pointing at a manifest that is gone.
    if not deletions:
        print("No package versions are outside the retention windows")
    for version in deletions:
        _delete_version(
            owner,
            owner_type,
            package,
            version,
            f"package version with tags {list(version.tags)}",
            execute=execute,
            mode=mode,
        )

    if not prune_untagged:
        print("Untagged platform manifests were left in place because their parents were not resolved")
        return

    orphans = orphaned_platform_manifests(versions, protected_digests)
    if not orphans:
        print("No orphaned platform manifests to remove")
        return
    for version in orphans:
        _delete_version(
            owner,
            owner_type,
            package,
            version,
            f"orphaned platform manifest {version.digest}",
            execute=execute,
            mode=mode,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--owner-type", choices=("user", "org"), default="user")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("verify-private").add_argument("--allow-missing", action="store_true")

    retention = subparsers.add_parser("retain")
    retention.add_argument("--current-tag", required=True)
    retention.add_argument("--keep-prerelease", type=int, default=3)
    retention.add_argument("--keep-stable", type=int, default=2)
    retention.add_argument("--execute", action="store_true")
    retention.add_argument(
        "--prune-untagged",
        action="store_true",
        help="also remove untagged platform manifests that no retained image references",
    )
    retention.add_argument(
        "--protect-digest",
        action="append",
        default=[],
        help="a digest a retained image references; repeat for each child",
    )

    kept = subparsers.add_parser("kept-tags")
    kept.add_argument("--current-tag", required=True)
    kept.add_argument("--keep-prerelease", type=int, default=3)
    kept.add_argument("--keep-stable", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    """Run a visibility or retention operation."""
    args = _parse_args()
    try:
        if args.command == "verify-private":
            verify_private(args.owner, args.package, args.owner_type, allow_missing=args.allow_missing)
        elif args.command == "kept-tags":
            for tag in retained_semver_tags(
                list_versions(args.owner, args.package, args.owner_type),
                args.current_tag,
                keep_prerelease=args.keep_prerelease,
                keep_stable=args.keep_stable,
            ):
                print(tag)
        else:
            retain(
                args.owner,
                args.package,
                args.owner_type,
                args.current_tag,
                keep_prerelease=args.keep_prerelease,
                keep_stable=args.keep_stable,
                execute=args.execute,
                prune_untagged=args.prune_untagged,
                protected_digests=frozenset(args.protect_digest),
            )
    except (PackageApiError, RetentionSafetyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
