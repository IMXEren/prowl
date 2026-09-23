"""Single source of the Prowl distribution version.

The release pipeline rewrites the assignment below from semantic-release's
``nextRelease.version`` before building the wheel and sdist, so the built
artifacts and the committed metadata always agree with the released tag.
"""

__version__ = "1.1.0-dev.2"
