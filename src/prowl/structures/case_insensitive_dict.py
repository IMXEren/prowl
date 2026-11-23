"""Small mapping utilities."""

from collections.abc import Iterable, Iterator, Mapping, MutableMapping


class CaseInsensitiveDict[V](MutableMapping[str, V]):
    """A single-value string-key mapping retaining each key's last spelling.

    Lookups use ``str.lower()``; overwriting a key preserves its insertion order.
    """

    __hash__ = None

    def __init__(
        self,
        data: Mapping[str, V] | Iterable[tuple[str, V]] | None = None,
        **kwargs: V,
    ) -> None:
        self._store: dict[str, tuple[str, V]] = {}
        if data is not None:
            self.update(data)
        self.update(kwargs)

    def __getitem__(self, key: str) -> V:
        """Return the value for any casing of *key*."""
        return self._store[key.lower()][1]

    def __setitem__(self, key: str, value: V) -> None:
        """Replace the value and remembered spelling of *key*."""
        self._store[key.lower()] = (key, value)

    def __delitem__(self, key: str) -> None:
        """Delete *key* regardless of casing."""
        del self._store[key.lower()]

    def __iter__(self) -> Iterator[str]:
        """Iterate over remembered spellings in insertion order."""
        return (key for key, _ in self._store.values())

    def __len__(self) -> int:
        """Count distinct lowercase keys."""
        return len(self._store)

    def lower_items(self) -> Iterator[tuple[str, V]]:
        """Iterate over lowercase keys and their values."""
        return ((key, item[1]) for key, item in self._store.items())

    def __eq__(self, other: object) -> bool:
        """Compare mappings without regard to key casing."""
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self.lower_items()) == dict(CaseInsensitiveDict(other).lower_items())

    def copy(self) -> "CaseInsensitiveDict[V]":
        """Return an independent shallow copy."""
        return CaseInsensitiveDict(self._store.values())

    def __repr__(self) -> str:
        """Represent the stored spellings and values as a dict."""
        return repr(dict(self.items()))
