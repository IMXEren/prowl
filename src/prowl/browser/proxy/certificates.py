"""Bounded TLS contexts signed by an explicitly supplied CA; client trust is unchanged."""

from __future__ import annotations

import ipaddress
import ssl
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from cryptography import x509
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID

from prowl.service.errors import CallerSafeError

__all__ = ["MAX_CERTIFICATE_CONTEXTS", "ProxyCertificateError", "ProxyCertificates"]

#: Largest number of host certificate contexts kept. Fixed so the cache can never grow an unbounded
#: set of contexts; the least recently used entry is dropped past the cap.
MAX_CERTIFICATE_CONTEXTS: Final[int] = 64

#: The one generic failure message. It never carries a path, hostname, key material or cause.
_GENERIC_MESSAGE: Final[str] = "the proxy certificate authority is not usable"

#: Host certificates start slightly before "now" so a small clock difference cannot mint a
#: certificate that looks not-yet-valid to the peer.
_CLOCK_SKEW: Final[timedelta] = timedelta(minutes=5)

#: The single ALPN protocol the interceptor advertises.
_ALPN_PROTOCOLS: Final[tuple[str, ...]] = ("http/1.1",)

type _SupportedKey = rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey


class ProxyCertificateError(CallerSafeError):
    """Raised when the operator certificate authority cannot be used, with a fixed safe message."""


@dataclass(frozen=True, slots=True)
class _OperatorAuthority:
    """A loaded, validated signing authority and the original PEM chain that accompanied it."""

    certificate: x509.Certificate
    key: _SupportedKey
    chain_pem: bytes
    not_before: datetime
    not_after: datetime


class ProxyCertificates:
    """Mint per-host TLS server contexts signed by one supplied operator authority."""

    def __init__(self, ca_cert: Path, ca_key: Path) -> None:
        """Load and validate the operator authority and generate the one shared leaf key.

        :raises ProxyCertificateError: when the pair is missing, unreadable, encrypted,
            unsupported, not a current signing authority, or mismatched.
        """
        self._authority = _load_authority(Path(ca_cert), Path(ca_key))
        self._leaf_key = ec.generate_private_key(ec.SECP256R1())
        self._contexts: OrderedDict[str, ssl.SSLContext] = OrderedDict()
        self._lock = threading.Lock()

    def context_for(self, hostname: str) -> ssl.SSLContext:
        """Return a cached TLS server context for an already validated CONNECT *hostname*.

        The hostname is matched case-insensitively: a repeat lookup for the same host returns the
        very same context object. Concurrent callers are serialized by one lock, so a context is
        created and cached exactly once.
        """
        name = hostname.lower()
        with self._lock:
            cached = self._contexts.get(name)
            if cached is not None:
                self._contexts.move_to_end(name)
                return cached
            try:
                context = _build_context(self._authority, self._leaf_key, name)
            except (OSError, ValueError, TypeError, UnsupportedAlgorithm):
                raise ProxyCertificateError(_GENERIC_MESSAGE) from None
            self._contexts[name] = context
            while len(self._contexts) > MAX_CERTIFICATE_CONTEXTS:
                self._contexts.popitem(last=False)
            return context


def _load_authority(ca_cert: Path, ca_key: Path) -> _OperatorAuthority:
    """Read and validate the operator pair, raising the one generic error on any failure."""
    try:
        chain_pem = ca_cert.read_bytes()
        key_pem = ca_key.read_bytes()
        certificate = x509.load_pem_x509_certificate(chain_pem)
        key = serialization.load_pem_private_key(key_pem, password=None)
        if not isinstance(key, (rsa.RSAPrivateKey, ec.EllipticCurvePrivateKey)):
            raise ProxyCertificateError(_GENERIC_MESSAGE) from None
        if not _key_matches(certificate, key) or not _is_current_signing_authority(certificate):
            raise ProxyCertificateError(_GENERIC_MESSAGE) from None
    except (OSError, ValueError, TypeError, UnsupportedAlgorithm, x509.DuplicateExtension):
        raise ProxyCertificateError(_GENERIC_MESSAGE) from None
    return _OperatorAuthority(
        certificate=certificate,
        key=key,
        chain_pem=chain_pem,
        not_before=certificate.not_valid_before_utc,
        not_after=certificate.not_valid_after_utc,
    )


def _key_matches(certificate: x509.Certificate, key: _SupportedKey) -> bool:
    """Return whether *certificate*'s public key is the public half of *key*."""
    encoding = serialization.Encoding.DER
    public_format = serialization.PublicFormat.SubjectPublicKeyInfo
    return certificate.public_key().public_bytes(encoding, public_format) == key.public_key().public_bytes(
        encoding,
        public_format,
    )


def _is_current_signing_authority(certificate: x509.Certificate) -> bool:
    """Return whether *certificate* may sign right now and is a certificate authority."""
    try:
        constraints = certificate.extensions.get_extension_for_class(x509.BasicConstraints).value
    except x509.ExtensionNotFound:
        return False
    if not constraints.ca:
        return False
    try:
        usage = certificate.extensions.get_extension_for_class(x509.KeyUsage).value
    except x509.ExtensionNotFound:
        pass
    else:
        if not usage.key_cert_sign:
            return False
    now = datetime.now(UTC)
    return certificate.not_valid_before_utc <= now <= certificate.not_valid_after_utc


def _build_context(
    authority: _OperatorAuthority,
    leaf_key: ec.EllipticCurvePrivateKey,
    hostname: str,
) -> ssl.SSLContext:
    """Mint a leaf for *hostname*, present it under the operator chain, and return its context."""
    leaf_pem = _sign_host_certificate(authority, leaf_key, hostname)
    leaf_key_pem = leaf_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return _load_server_context(leaf_pem + authority.chain_pem, leaf_key_pem)


def _sign_host_certificate(
    authority: _OperatorAuthority,
    leaf_key: ec.EllipticCurvePrivateKey,
    hostname: str,
) -> bytes:
    """Sign one SAN leaf certificate for *hostname* with the authority; return its PEM.

    The subject is left empty, so the SAN is marked critical. The validity window is bounded by the
    authority: it starts at "now" minus a small skew (never before the authority itself) and ends
    when the authority expires, so no renewal daemon is needed.
    """
    now = datetime.now(UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([]))
        .issuer_name(authority.certificate.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(max(now - _CLOCK_SKEW, authority.not_before))
        .not_valid_after(authority.not_after)
        .add_extension(x509.SubjectAlternativeName([_subject_alt_name(hostname)]), critical=True)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(authority.key, hashes.SHA256())
    )
    return certificate.public_bytes(serialization.Encoding.PEM)


def _subject_alt_name(hostname: str) -> x509.GeneralName:
    """Return an IP SAN for a literal address, otherwise a DNS SAN for the IDNA form of *hostname*."""
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return x509.DNSName(hostname.encode("idna").decode("ascii"))
    return x509.IPAddress(address)


def _load_server_context(chain_pem: bytes, leaf_key_pem: bytes) -> ssl.SSLContext:
    """Build a secure server context, deleting every temporary file on success and on failure."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.set_alpn_protocols(_ALPN_PROTOCOLS)
    with tempfile.TemporaryDirectory() as directory:
        certificate_path = _write_private_file(directory, ".pem", chain_pem)
        key_path = _write_private_file(directory, ".key", leaf_key_pem)
        context.load_cert_chain(certfile=certificate_path, keyfile=key_path)
    return context


def _write_private_file(directory: str, suffix: str, data: bytes) -> str:
    """Write *data* to a new private file in *directory* and return its path, closed before return."""
    with tempfile.NamedTemporaryFile(dir=directory, suffix=suffix, delete=False) as handle:
        handle.write(data)
        handle.flush()
    return handle.name
