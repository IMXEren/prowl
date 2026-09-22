"""Allow ``python -m prowl.service`` to run the HTTP service."""

from prowl.service.app import run_server

if __name__ == "__main__":
    run_server()
