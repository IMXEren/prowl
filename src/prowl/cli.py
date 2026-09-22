"""Command-line interface for one-shot browser fetches and the HTTP service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from prowl.browser.site import source as fetch_source
from prowl.browser.utils import run_coroutine_sync

EXIT_OK = 0
EXIT_FETCH_FAILED = 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prowl",
        description="Fetch pages through a real browser, or run the FlareSolverr-compatible service.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="load one URL and print its HTML source")
    fetch.add_argument("url", help="http(s) URL to load")
    fetch.add_argument("--timeout", type=int, default=60, help="load timeout in seconds (default: 60)")
    fetch.add_argument("-o", "--output", type=Path, default=None, help="write the source to this file")

    serve = subparsers.add_parser("serve", help="run the FlareSolverr-compatible HTTP service")
    serve.add_argument("--host", default=None, help="bind address (default: 0.0.0.0)")
    serve.add_argument("--port", type=int, default=None, help="bind port (default: 8191)")

    return parser


def _run_fetch(args: argparse.Namespace) -> int:
    try:
        page = run_coroutine_sync(fetch_source(args.url, args.timeout))
    except Exception as exc:  # noqa: BLE001
        logger.error(f"failed to load {args.url}: {exc}")
        return EXIT_FETCH_FAILED
    source = page.text
    if args.output is not None:
        args.output.write_text(source, encoding="utf-8")
    else:
        sys.stdout.write(source)
        if not source.endswith("\n"):
            sys.stdout.write("\n")
    return EXIT_OK


def _run_serve(args: argparse.Namespace) -> int:
    from prowl.service.app import ServiceConfig, run_server  # noqa: PLC0415

    config = ServiceConfig.from_env()
    if args.host is not None:
        config.host = args.host
    if args.port is not None:
        config.port = args.port
    run_server(config)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit status."""
    args = _build_parser().parse_args(argv)
    if args.command == "fetch":
        return _run_fetch(args)
    return _run_serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
