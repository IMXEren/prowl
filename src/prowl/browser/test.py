"""Run a manual browser-backed page fetch against a caller-supplied URL."""

import argparse
import asyncio
import time

import turbohtml
from loguru import logger

from prowl.browser.site import source as page_source


async def main(url: str, request_timeout: float) -> None:
    """Fetch *url* in the browser and print its document title."""
    try:
        response = await page_source(url, request_timeout)
        document = turbohtml.parse(response.text)
        title = document.select_one("title")
        if title:
            print(title.text)  # noqa: T201
    except BaseException as error:
        logger.error(error)
        raise


def parse_args() -> argparse.Namespace:
    """Parse manual-test command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", default="https://example.com/")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    start = time.perf_counter()
    asyncio.run(main(arguments.url, arguments.timeout))
    elapsed = time.perf_counter() - start
    print(f"Elapsed time: {elapsed:.3f} seconds")  # noqa: T201
