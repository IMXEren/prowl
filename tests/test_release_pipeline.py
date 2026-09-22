"""Release pipeline contracts: semantic-release target, version wiring, artifact audit."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github" / "scripts"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit = _load("audit_artifacts")
set_version = _load("set_version")


def _plugin(name: str) -> dict:
    config = json.loads((ROOT / ".releaserc").read_text(encoding="utf-8"))
    for plugin in config["plugins"]:
        if isinstance(plugin, list) and plugin[0] == name:
            return plugin[1]
    msg = f"{name} is not configured"
    raise AssertionError(msg)


def _write_wheel(path: Path, version: str, extra: dict[str, str] | None = None) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("prowl/__init__.py", "")
        archive.writestr(
            f"prowl-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: prowl\nVersion: {version}\n",
        )
        for name, content in (extra or {}).items():
            archive.writestr(name, content)


def _write_sdist(path: Path, version: str, extra: dict[str, str] | None = None) -> None:
    prefix = f"prowl-{version}"
    entries = {
        f"{prefix}/PKG-INFO": f"Name: prowl\nVersion: {version}\n",
        f"{prefix}/src/prowl/__init__.py": "",
    }
    entries.update(extra or {})
    with tarfile.open(path, "w:gz") as archive:
        for name, content in entries.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_release_config_targets_main_and_dev_only() -> None:
    config = json.loads((ROOT / ".releaserc").read_text(encoding="utf-8"))
    assert config["branches"] == ["main", {"name": "dev", "prerelease": True}]
    names = [plugin if isinstance(plugin, str) else plugin[0] for plugin in config["plugins"]]
    for forbidden in ("docker", "npm", "pypi", "gitlab", "gpr", "publish"):
        assert not any(forbidden in name for name in names), forbidden


def test_release_config_attaches_only_wheel_and_sdist() -> None:
    github = _plugin("@semantic-release/github")
    assert github["assets"] == ["dist/*.whl", "dist/*.tar.gz"]
    assert github["successComment"] is False
    assert github["failComment"] is False


def test_release_config_commits_version_and_changelog() -> None:
    exec_config = _plugin("@semantic-release/exec")
    assert "${nextRelease.version}" in exec_config["prepareCmd"]
    assert "prepare-release.sh" in exec_config["prepareCmd"]
    git_assets = _plugin("@semantic-release/git")["assets"]
    assert "CHANGELOG.md" in git_assets
    assert "src/prowl/__about__.py" in git_assets


def test_package_json_is_private_and_publish_free() -> None:
    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    assert package["private"] is True
    assert "publishConfig" not in package
    assert "publish" not in package.get("scripts", {})
    dev_dependencies = package["devDependencies"]
    assert not any(name.startswith("@semantic-release/npm") for name in dev_dependencies)
    assert "@semantic-release/github" in dev_dependencies


def test_release_workflow_has_full_gates_without_registry_publication() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "fetch-depth: 0" in workflow
    assert "cancel-in-progress: true" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert workflow.count("contents: write") == 1
    assert "needs: verify" in workflow
    assert "if: github.event_name != 'pull_request'" in workflow
    assert "packages: write" not in workflow
    assert "id-token" not in workflow
    assert "docker/build-push-action" in workflow
    assert workflow.count("push: false") == 2
    assert "docker/login-action" not in workflow
    assert workflow.index("Fetch private fingerprint fonts") < workflow.index(
        "Build local image with fingerprint fonts"
    )
    assert "windows_fonts=.private-fonts" in workflow


def test_dev_push_opens_a_draft_pull_request_to_main() -> None:
    workflow = (ROOT / ".github" / "workflows" / "open_pull_request.yml").read_text(encoding="utf-8")
    assert "branches:\n      - dev" in workflow
    assert "destination_branch: main" in workflow
    assert "pr_draft: true" in workflow
    assert "pull-requests: write" in workflow


def test_set_version_rewrites_single_source(tmp_path: Path) -> None:
    about = tmp_path / "__about__.py"
    about.write_text('"""Version."""\n\n__version__ = "0.0.0"\n', encoding="utf-8")
    set_version.set_version("1.2.3", about)
    assert about.read_text(encoding="utf-8").count('__version__ = "1.2.3"') == 1


def test_set_version_rejects_missing_assignment(tmp_path: Path) -> None:
    about = tmp_path / "__about__.py"
    about.write_text('"""Version."""\n', encoding="utf-8")
    try:
        set_version.set_version("1.2.3", about)
    except SystemExit:
        return
    msg = "set_version accepted a file without an __version__ assignment"
    raise AssertionError(msg)


def test_audit_accepts_clean_artifacts(tmp_path: Path) -> None:
    _write_wheel(tmp_path / "prowl-1.2.3-py3-none-any.whl", "1.2.3")
    _write_sdist(tmp_path / "prowl-1.2.3.tar.gz", "1.2.3")
    assert audit.audit_dist(tmp_path, "1.2.3") == []


def test_audit_rejects_forbidden_payloads(tmp_path: Path) -> None:
    _write_wheel(
        tmp_path / "prowl-1.2.3-py3-none-any.whl",
        "1.2.3",
        {"prowl/vendor/cloakbrowser": "binary"},
    )
    _write_sdist(
        tmp_path / "prowl-1.2.3.tar.gz",
        "1.2.3",
        {
            "prowl-1.2.3/geoip/GeoLite2-City.mmdb": "db",
            "prowl-1.2.3/node_modules/example/index.js": "module",
        },
    )
    failures = audit.audit_dist(tmp_path, "1.2.3")
    assert any("cloakbrowser" in failure for failure in failures)
    assert any("mmdb" in failure.lower() for failure in failures)
    assert any("node_modules" in failure for failure in failures)


def test_audit_rejects_version_drift(tmp_path: Path) -> None:
    _write_wheel(tmp_path / "prowl-9.9.9-py3-none-any.whl", "9.9.9")
    _write_sdist(tmp_path / "prowl-9.9.9.tar.gz", "9.9.9")
    failures = audit.audit_dist(tmp_path, "1.2.3")
    assert len(failures) == 2
    assert all("9.9.9" in failure for failure in failures)


def test_audit_requires_one_wheel_and_one_sdist(tmp_path: Path) -> None:
    _write_wheel(tmp_path / "prowl-1.2.3-py3-none-any.whl", "1.2.3")
    failures = audit.audit_dist(tmp_path, "1.2.3")
    assert len(failures) == 1
    assert "exactly one wheel and one sdist" in failures[0]
