# syntax=docker/dockerfile:1

# Empty fallback context. BuildKit's `--build-context windows_fonts=<dir>` can
# override this stage with a directory containing fonts.zip.
FROM scratch AS windows_fonts

FROM python:3.12-slim-bookworm

# Chromium runtime libraries and base fonts proven with CloakBrowser on Debian.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libasound2 libatk-bridge2.0-0 libatk1.0-0 libatspi2.0-0 \
    libcairo2 libcups2 libdbus-1-3 libdrm2 libgbm1 libglib2.0-0 \
    libnspr4 libnss3 libpango-1.0-0 libx11-6 libxcb1 \
    libxcomposite1 libxdamage1 libxext6 libxfixes3 \
    libxkbcommon0 libxrandr2 libx11-xcb1 \
    unzip curl ca-certificates git \
    fonts-liberation fonts-noto-color-emoji fonts-unifont \
    fonts-freefont-ttf fonts-ipafont-gothic fonts-wqy-zenhei \
    fonts-tlwg-loma-otf libfontconfig1 libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Optional Windows font metrics improve fingerprint fidelity. BuildKit secrets
# are limited to 500 KiB, so the font directory is supplied as an additional
# read-only build context instead of being copied into the source context:
#   docker build --build-context windows_fonts=./private-fonts -t prowl:local .
RUN --mount=type=bind,from=windows_fonts,source=.,target=/tmp/windows-fonts,ro \
    if [ -s /tmp/windows-fonts/fonts.zip ]; then \
        mkdir -p /usr/share/fonts/windows && \
        unzip -q /tmp/windows-fonts/fonts.zip -d /usr/share/fonts/windows/ && \
        fc-cache -f; \
    fi

ENV CLOAKBROWSER_AUTO_UPDATE="false" \
    PROWL_PROFILE_DIR=/state/profile \
    PROWL_PROFILE_ARCHIVE=/state/browser-profile.zip \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip && pip install .

# Run as a non-root user. Chromium keeps --no-sandbox because containers do not
# grant the user namespaces its sandbox needs; the browser only reaches sites
# the caller asks it to.
RUN useradd --create-home --uid 10001 browser \
    && mkdir -p /state && chown -R browser:browser /state

# Install the CloakBrowser binary and GeoIP database as the runtime user.
USER browser
ENV HOME=/home/browser
RUN python -c "import httpx; from cloakbrowser import download, ensure_binary; download.DOWNLOAD_TIMEOUT = httpx.Timeout(300.0, connect=120.0); ensure_binary()"
RUN mkdir -p /home/browser/.cloakbrowser/geoip \
    && curl --fail --location --retry 5 --retry-all-errors --connect-timeout 30 --max-time 300 \
        --output /home/browser/.cloakbrowser/geoip/GeoLite2-City.mmdb \
        "https://github.com/P3TERX/GeoLite.mmdb/raw/download/GeoLite2-City.mmdb"

USER root
COPY entrypoint /entrypoint
RUN sed -i 's/\r$//g' /entrypoint && chmod +x /entrypoint
USER browser

VOLUME ["/state"]
EXPOSE 8191

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8191/healthz || exit 1

CMD ["bash", "/entrypoint"]
