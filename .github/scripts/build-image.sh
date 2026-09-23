#!/usr/bin/env bash
# Build Prowl with an optional font archive kept outside the source context.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: build-image.sh [--font-archive PATH] [docker buildx build options]

The build context is the repository root. A local archive can be supplied with
--font-archive (or PROWL_FONT_ARCHIVE). When PROWL_FONT_GITHUB_TOKEN is set,
the helper otherwise fetches PROWL_FONT_ARCHIVE_PATH from the Git LFS repository
named by PROWL_FONT_REPOSITORY and PROWL_FONT_REF.
EOF
}

font_archive="${PROWL_FONT_ARCHIVE:-}"
docker_args=()
while (($#)); do
    case "$1" in
        --font-archive)
            [[ $# -ge 2 ]] || { echo "--font-archive requires a path" >&2; exit 2; }
            font_archive="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            docker_args+=("$1")
            shift
            ;;
    esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
temporary_root=""
cleanup() {
    if [[ -n "$temporary_root" ]]; then
        rm -rf "$temporary_root"
    fi
}
trap cleanup EXIT

make_temporary_root() {
    if [[ -z "$temporary_root" ]]; then
        temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/prowl-image-build.XXXXXXXX")"
        chmod 700 "$temporary_root"
    fi
}

if [[ -z "$font_archive" && -n "${PROWL_FONT_GITHUB_TOKEN:-}" ]]; then
    command -v git >/dev/null || { echo "git is required to fetch the optional font archive" >&2; exit 1; }
    git lfs version >/dev/null 2>&1 || { echo "git-lfs is required to fetch the optional font archive" >&2; exit 1; }

    make_temporary_root
    font_repository="${PROWL_FONT_REPOSITORY:-IMXEren/extra-fonts}"
    font_ref="${PROWL_FONT_REF:-main}"
    remote_archive="${PROWL_FONT_ARCHIVE_PATH:-fonts.zip}"
    if [[ "$remote_archive" == /* || "/$remote_archive/" == *"/../"* ]]; then
        echo "PROWL_FONT_ARCHIVE_PATH must be a relative path inside the font repository" >&2
        exit 2
    fi
    askpass="$temporary_root/git-askpass.sh"
    cat >"$askpass" <<'EOF'
#!/usr/bin/env bash
case "$1" in
    *Username*) printf '%s\n' 'x-access-token' ;;
    *Password*) printf '%s\n' "$PROWL_FONT_GITHUB_TOKEN" ;;
    *) exit 1 ;;
esac
EOF
    chmod 700 "$askpass"

    echo "Fetching optional font archive from ${font_repository}@${font_ref}"
    mkdir "$temporary_root/font-repository"
    (
        cd "$temporary_root/font-repository"
        git init --quiet
        git remote add origin "https://github.com/${font_repository}.git"
        GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 GIT_LFS_SKIP_SMUDGE=1 \
            git fetch --quiet --depth 1 origin "$font_ref"
        GIT_LFS_SKIP_SMUDGE=1 git checkout --quiet --detach FETCH_HEAD
        GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
            git lfs pull --include="$remote_archive" --exclude=""
    )
    font_archive="$temporary_root/font-repository/$remote_archive"
fi

build_context_args=()
if [[ -n "$font_archive" ]]; then
    [[ -f "$font_archive" ]] || { echo "Font archive not found: $font_archive" >&2; exit 1; }
    if grep -q '^version https://git-lfs.github.com/spec/v1' "$font_archive"; then
        echo "Font archive is still a Git LFS pointer: $font_archive" >&2
        exit 1
    fi
    make_temporary_root
    mkdir -p "$temporary_root/windows-fonts"
    cp "$font_archive" "$temporary_root/windows-fonts/fonts.zip"
    build_context_args=(--build-context "windows_fonts=$temporary_root/windows-fonts")
    echo "Building with the optional fingerprint font archive"
else
    echo "Building without an optional fingerprint font archive"
fi

cd "$repo_root"
docker buildx build "${build_context_args[@]}" "${docker_args[@]}" .
