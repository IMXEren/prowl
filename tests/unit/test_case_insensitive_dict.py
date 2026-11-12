"""Single-value, case-insensitive mapping compatibility."""

from collections import UserDict

import pytest

from prowl.browser.page_handler import PageResponse, _generate_headers
from prowl.structures import CaseInsensitiveDict


def test_last_spelling_and_value_preserve_first_insertion_order() -> None:
    headers = CaseInsensitiveDict([("Accept", "one"), ("Other", "two"), ("ACCEPT", "three")])
    assert list(headers.items()) == [("ACCEPT", "three"), ("Other", "two")]
    assert list(headers.lower_items()) == [("accept", "three"), ("other", "two")]
    assert headers["aCcEpT"] == "three"
    assert "accept" in headers
    assert len(headers) == 2
    assert dict(headers) == {"ACCEPT": "three", "Other": "two"}


def test_mapping_iterable_kwargs_and_update() -> None:
    headers = CaseInsensitiveDict(UserDict({"Header": 1}), HEADER=2)
    headers.update([("header", 3), ("Other", 4)], OTHER=5)
    assert list(headers.items()) == [("header", 3), ("OTHER", 5)]
    assert CaseInsensitiveDict(None, TEST=1) == {"test": 1}
    assert not CaseInsensitiveDict()


def test_equality_normalizes_mapping_keys_not_values() -> None:
    headers = CaseInsensitiveDict({"Name": "VALUE"})
    assert headers == CaseInsensitiveDict({"NAME": "VALUE"})
    assert headers == UserDict({"name": "VALUE"})
    mapping = {"name": "VALUE"}
    assert mapping == headers
    assert headers != {"name": "value"}
    assert headers != {"Other": "VALUE"}
    assert headers != [("Name", "VALUE")]
    assert headers.__eq__(object()) is NotImplemented


def test_mutable_mapping_methods_are_case_insensitive() -> None:
    headers = CaseInsensitiveDict({"Name": 1, "Other": 2})
    assert headers.get("NAME") == 1
    assert headers.get("missing", 3) == 3
    assert headers.setdefault("NAME", 4) == 1
    assert headers.setdefault("New", 5) == 5
    assert headers.pop("OTHER") == 2
    del headers["nAmE"]
    assert list(headers.items()) == [("New", 5)]
    assert headers.pop("missing", 6) == 6
    assert headers.popitem() == ("New", 5)
    headers.update({"A": 1, "B": 2})
    headers.clear()
    assert not headers


def test_copy_is_shallow_and_independent() -> None:
    value = []
    headers = CaseInsensitiveDict({"Name": value})
    copied = headers.copy()
    assert copied is not headers
    assert list(copied) == ["Name"]
    assert copied["name"] is value
    copied["NAME"] = [1]
    assert headers["name"] == []
    assert list(headers) == ["Name"]


def test_repr_and_missing_keys_match_mapping_behavior() -> None:
    headers = CaseInsensitiveDict({"Name": "value"})
    assert repr(headers) == "{'Name': 'value'}"
    for operation in (lambda: headers["MISSING"], lambda: headers.pop("MISSING")):
        with pytest.raises(KeyError) as exc:
            operation()
        assert exc.value.args == ("missing",)


def test_keys_use_lower_not_unicode_casefold() -> None:
    headers = CaseInsensitiveDict({"Straße": 1, "STRASSE": 2})
    assert len(headers) == 2
    assert headers["straße"] == 1
    assert headers["strasse"] == 2


def test_response_and_intercepted_headers_use_local_mapping() -> None:
    headers = _generate_headers([{"name": "Content-Type", "value": "text/html"}])
    response = PageResponse("body", headers=headers)
    assert isinstance(response.headers, CaseInsensitiveDict)
    assert response.headers["CONTENT-TYPE"] == "text/html"
    response.headers["content-type"] = "text/plain"
    assert headers["Content-Type"] == "text/html"
    assert PageResponse("body", headers={"Content-Type": "text/plain"}).headers["CONTENT-type"] == "text/plain"
