"""Focused tests for the per-host proxy certificate helper.

Every test runs against a throwaway authority generated under a ``TemporaryDirectory`` outside the
repository: nothing is installed, no machine trust store is touched, and no real certificate is
tracked. The loopback tests use real :mod:`ssl` contexts and real asyncio sockets only.
"""

from __future__ import annotations

import asyncio
import contextlib
import ssl
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import IsolatedAsyncioTestCase, TestCase, mock

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import NameOID

from prowl.browser.proxy import certificates as proxy_certificates
from prowl.browser.proxy.certificates import ProxyCertificates

if TYPE_CHECKING:
    from collections.abc import Mapping


def _generate_authority(
    directory: Path,
    *,
    name: str = "ca",
    ca: bool = True,
    valid: bool = True,
    key_type: str = "ec",
) -> tuple[Path, Path]:
    """Write a throwaway self-signed authority pair and return its certificate and key paths."""
    if key_type == "rsa":
        key: ec.EllipticCurvePrivateKey | rsa.RSAPrivateKey = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
    else:
        key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"prowl-test-{name}")])
    now = datetime.now(UTC)
    if valid:
        not_before, not_after = now - timedelta(minutes=5), now + timedelta(hours=1)
    else:
        not_before, not_after = now - timedelta(hours=2), now - timedelta(hours=1)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=ca,
                crl_sign=ca,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path = directory / f"{name}-cert.pem"
    key_path = directory / f"{name}-key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
    )
    return cert_path, key_path


async def _serve(context: ssl.SSLContext) -> tuple[asyncio.Server, int]:
    """Start a loopback TLS server with *context* and return it with its bound port."""

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await asyncio.wait_for(reader.read(1), 5)
        except (OSError, TimeoutError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()

    server = await asyncio.start_server(_handle, "127.0.0.1", 0, ssl=context)
    sockets = server.sockets
    assert sockets
    return server, int(sockets[0].getsockname()[1])


async def _close_server(server: asyncio.Server) -> None:
    """Stop a loopback server and wait for it to close, bounded."""
    server.close()
    async with asyncio.timeout(5):
        await server.wait_closed()


async def _handshake(port: int, ca_file: Path, server_hostname: str) -> tuple[str | None, Mapping[str, object]]:
    """Complete a verified TLS handshake and return the negotiated ALPN and peer certificate."""
    context = ssl.create_default_context(cafile=str(ca_file))
    context.set_alpn_protocols(["http/1.1"])
    _reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=context, server_hostname=server_hostname)
    try:
        ssl_object = writer.get_extra_info("ssl_object")
        assert isinstance(ssl_object, ssl.SSLObject)
        peer = ssl_object.getpeercert()
        assert peer is not None
        return ssl_object.selected_alpn_protocol(), peer
    finally:
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()


class TrustedCertificateTests(IsolatedAsyncioTestCase):
    """A verified loopback client accepts the minted leaf for the requested hostname."""

    async def test_example_host_serves_a_trusted_san_certificate_over_http1_alpn(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ca_cert, ca_key = _generate_authority(Path(raw))
            server, port = await _serve(ProxyCertificates(ca_cert, ca_key).context_for("example.test"))
            try:
                alpn, peer = await _handshake(port, ca_cert, "example.test")
            finally:
                await _close_server(server)

        self.assertEqual(alpn, "http/1.1")
        self.assertEqual(peer["subjectAltName"], (("DNS", "example.test"),))


class RsaAuthorityTests(IsolatedAsyncioTestCase):
    """An RSA operator authority mints a leaf that a verified loopback client accepts."""

    async def test_rsa_authority_serves_a_trusted_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ca_cert, ca_key = _generate_authority(Path(raw), key_type="rsa")
            server, port = await _serve(ProxyCertificates(ca_cert, ca_key).context_for("example.test"))
            try:
                alpn, peer = await _handshake(port, ca_cert, "example.test")
            finally:
                await _close_server(server)

        self.assertEqual(alpn, "http/1.1")
        self.assertEqual(peer["subjectAltName"], (("DNS", "example.test"),))


class RejectedHandshakeTests(IsolatedAsyncioTestCase):
    """A mismatched hostname or a foreign authority must fail verification, never be bypassed."""

    async def test_wrong_hostname_and_foreign_authority_fail_verification(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            ca_cert, ca_key = _generate_authority(directory, name="trusted")
            foreign_cert, _ = _generate_authority(directory, name="foreign")
            server, port = await _serve(ProxyCertificates(ca_cert, ca_key).context_for("example.test"))
            try:
                with self.assertRaises(ssl.SSLCertVerificationError):
                    await _handshake(port, ca_cert, "wrong.test")
                with self.assertRaises(ssl.SSLCertVerificationError):
                    await _handshake(port, foreign_cert, "example.test")
            finally:
                await _close_server(server)


class AddressCertificateTests(IsolatedAsyncioTestCase):
    """A literal address becomes an IP SAN, verified over loopback for IPv4 and inspected for IPv6."""

    async def test_ipv4_host_serves_an_verified_ip_san(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ca_cert, ca_key = _generate_authority(Path(raw))
            server, port = await _serve(ProxyCertificates(ca_cert, ca_key).context_for("127.0.0.1"))
            try:
                _, peer = await _handshake(port, ca_cert, "127.0.0.1")
            finally:
                await _close_server(server)

        self.assertEqual(peer["subjectAltName"], (("IP Address", "127.0.0.1"),))

    def test_ipv6_host_produces_an_ip_san_in_the_signed_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ca_cert, ca_key = _generate_authority(Path(raw))
            authority = proxy_certificates._load_authority(ca_cert, ca_key)
            leaf_key = ec.generate_private_key(ec.SECP256R1())
            leaf = x509.load_pem_x509_certificate(
                proxy_certificates._sign_host_certificate(authority, leaf_key, "::1"),
            )
            san = leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).value

        self.assertEqual({str(address) for address in san.get_values_for_type(x509.IPAddress)}, {"::1"})


class RejectedAuthorityTests(TestCase):
    """Every unusable operator pair reports the same fixed, secret-free error."""

    def test_unusable_pairs_report_one_generic_message(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            ca_cert, ca_key = _generate_authority(directory, name="good")
            _, foreign_key = _generate_authority(directory, name="foreign")
            expired_cert, expired_key = _generate_authority(directory, name="expired", valid=False)
            not_a_ca_cert, not_a_ca_key = _generate_authority(directory, name="leaf", ca=False)

            invalid_cert = directory / "invalid-cert.pem"
            invalid_cert.write_bytes(b"-----BEGIN CERTIFICATE-----\nnot-a-certificate\n")

            encrypted_key = directory / "encrypted-key.pem"
            encrypted_key.write_bytes(
                ec.generate_private_key(ec.SECP256R1()).private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.BestAvailableEncryption(b"test-passphrase"),
                ),
            )

            cases = {
                "missing-certificate": (directory / "absent.pem", ca_key),
                "missing-key": (ca_cert, directory / "absent.key"),
                "invalid-certificate": (invalid_cert, ca_key),
                "encrypted-key": (ca_cert, encrypted_key),
                "key-mismatch": (ca_cert, foreign_key),
                "expired": (expired_cert, expired_key),
                "not-a-ca": (not_a_ca_cert, not_a_ca_key),
            }
            for name, (cert, key) in cases.items():
                with self.subTest(name=name):
                    with self.assertRaises(proxy_certificates.ProxyCertificateError) as caught:
                        ProxyCertificates(cert, key)
                    error = caught.exception
                    self.assertEqual(str(error), proxy_certificates._GENERIC_MESSAGE)
                    self.assertNotIn(str(directory), str(error))
                    self.assertIsNone(error.__cause__)


class ContextCacheTests(TestCase):
    """The cache matches hostnames case-insensitively and evicts the least recently used context."""

    def test_repeat_lookup_is_the_same_object_and_the_cache_stays_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ca_cert, ca_key = _generate_authority(Path(raw))
            certificates = ProxyCertificates(ca_cert, ca_key)

            self.assertIs(certificates.context_for("Example.Test"), certificates.context_for("example.test"))
            self.assertEqual(list(certificates._contexts), ["example.test"])

            with mock.patch.object(proxy_certificates, "MAX_CERTIFICATE_CONTEXTS", 2):
                certificates.context_for("one.test")
                two = certificates.context_for("two.test")
                three = certificates.context_for("three.test")

                self.assertEqual(len(certificates._contexts), 2)
                self.assertNotIn("example.test", certificates._contexts)
                self.assertNotIn("one.test", certificates._contexts)
                self.assertIs(certificates.context_for("two.test"), two)
                self.assertIs(certificates.context_for("three.test"), three)


class TemporaryFileCleanupTests(TestCase):
    """The leaf directory is deleted after the context is built, even when the build fails."""

    def test_temporary_leaf_directory_is_removed_on_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ca_cert, ca_key = _generate_authority(root)
            certificates = ProxyCertificates(ca_cert, ca_key)

            real_temporary_directory = tempfile.TemporaryDirectory
            observed: list[str] = []

            def factory() -> tempfile.TemporaryDirectory[str]:
                created = real_temporary_directory()
                observed.append(created.name)
                return created

            with mock.patch.object(proxy_certificates.tempfile, "TemporaryDirectory", factory):
                certificates.context_for("example.test")

            self.assertEqual(len(observed), 1)
            self.assertFalse(Path(observed[0]).exists())
            self.assertTrue(root.exists())

            with (
                mock.patch.object(proxy_certificates.tempfile, "TemporaryDirectory", factory),
                mock.patch.object(ssl.SSLContext, "load_cert_chain", side_effect=ssl.SSLError("boom")),
                self.assertRaises(proxy_certificates.ProxyCertificateError) as caught,
            ):
                certificates.context_for("failure.test")

            self.assertIsNone(caught.exception.__cause__)
            self.assertNotIn("boom", str(caught.exception))
            self.assertEqual(len(observed), 2)
            self.assertFalse(Path(observed[1]).exists())
            self.assertTrue(root.exists())


class SigningPermissionTests(TestCase):
    def test_ca_without_certificate_signing_permission_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            cert_path, key_path = _generate_authority(Path(raw))
            original = x509.load_pem_x509_certificate(cert_path.read_bytes())
            key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
            assert isinstance(key, ec.EllipticCurvePrivateKey)
            certificate = (
                x509.CertificateBuilder()
                .subject_name(original.subject)
                .issuer_name(original.issuer)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(original.not_valid_before_utc)
                .not_valid_after(original.not_valid_after_utc)
                .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
                .add_extension(
                    x509.KeyUsage(
                        digital_signature=True,
                        content_commitment=False,
                        key_encipherment=False,
                        data_encipherment=False,
                        key_agreement=False,
                        key_cert_sign=False,
                        crl_sign=False,
                        encipher_only=False,
                        decipher_only=False,
                    ),
                    critical=True,
                )
                .sign(key, hashes.SHA256())
            )
            cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
            with self.assertRaises(proxy_certificates.ProxyCertificateError) as caught:
                ProxyCertificates(cert_path, key_path)
            self.assertIsNone(caught.exception.__cause__)
