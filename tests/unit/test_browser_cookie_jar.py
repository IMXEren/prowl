"""Browser cookie conversion preserves the previous CookieJar defaults."""

from http.cookiejar import Cookie as JarCookie
from pathlib import Path
from urllib.request import Request

import pytest

from prowl.browser.cookies import Cookie, Cookies, create_cookie


@pytest.mark.parametrize(
    "attributes",
    [
        ("example.com", "/", None, False, False),
        (".example.com", "/private", 2000000000.75, True, True),
        ("", "", 0, False, True),
    ],
)
def test_cookie_attributes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attributes: tuple[str, str, float | None, bool, bool],
) -> None:
    domain, path, expires, secure, http_only = attributes
    monkeypatch.chdir(tmp_path)
    cookies = Cookies()
    cookies.cookies = [
        Cookie(
            name="sid", value="fixture", domain=domain, path=path, expires=expires, secure=secure, httpOnly=http_only
        )
    ]
    jar = cookies.load_to_cookie_jar()
    cookie = next(iter(jar))
    assert isinstance(cookie, JarCookie)
    assert cookie.name == "sid"
    assert cookie.value == "fixture"
    assert cookie.version == 0
    assert cookie.port is None
    assert cookie.port_specified is False
    assert cookie.domain == domain
    assert cookie.domain_specified == bool(domain)
    assert cookie.domain_initial_dot == domain.startswith(".")
    assert cookie.path == path
    assert cookie.path_specified == bool(path)
    assert cookie.expires == (int(expires) if expires is not None else None)
    assert cookie.discard is True
    assert cookie.secure == secure
    assert cookie.get_nonstandard_attr("HttpOnly") == http_only
    assert cookie.comment is None
    assert cookie.comment_url is None
    assert cookie.rfc2109 is False


def test_cookie_jar_retains_scope_and_secure_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cookies = Cookies()
    cookies.cookies = [
        Cookie(
            name="sid", value="fixture", domain="example.com", path="/private", expires=None, secure=True, httpOnly=True
        )
    ]
    jar = cookies.load_to_cookie_jar()
    for url, expected in (
        ("https://example.com/private/page", "sid=fixture"),
        ("https://other.invalid/private/page", None),
        ("https://example.com/public", None),
        ("http://example.com/private/page", None),
    ):
        request = Request(url)
        jar.add_cookie_header(request)
        assert request.get_header("Cookie") == expected
    assert not Cookies().load_to_cookie_jar()


def test_create_cookie_defaults_and_overrides() -> None:
    cookie = create_cookie("sid", "fixture")
    assert isinstance(cookie, JarCookie)
    assert (cookie.domain, cookie.path, cookie.version) == ("", "/", 0)
    assert cookie.port is None
    assert not cookie.port_specified
    assert not cookie.domain_specified
    assert not cookie.domain_initial_dot
    assert cookie.path_specified
    assert cookie.expires is None
    assert cookie.discard
    assert not cookie.secure
    assert cookie.has_nonstandard_attr("HttpOnly")
    assert cookie.get_nonstandard_attr("HttpOnly") is None
    assert not cookie.rfc2109
    overridden = create_cookie(
        "sid",
        "fixture",
        domain=".example.com",
        path="",
        port="443",
        expires=2000000000.75,
        discard=False,
        secure=True,
        version=1,
        rest={"HttpOnly": False, "SameSite": "Lax"},
        comment="comment",
        comment_url="https://example.com/",
        rfc2109=True,
    )
    assert overridden.port == "443"
    assert overridden.port_specified
    assert overridden.domain_specified
    assert overridden.domain_initial_dot
    assert not overridden.path_specified
    assert overridden.expires == 2000000000
    assert not overridden.discard
    assert overridden.secure
    assert overridden.version == 1
    assert overridden.comment == "comment"
    assert overridden.comment_url == "https://example.com/"
    assert overridden.rfc2109
    assert overridden.get_nonstandard_attr("HttpOnly") is False
    assert overridden.get_nonstandard_attr("SameSite") == "Lax"


def test_create_cookie_derives_flags_and_rejects_unknown_keywords() -> None:
    for name in ("domain_specified", "port_specified", "path_specified", "unknown"):
        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            create_cookie("sid", "fixture", **{name: True})
