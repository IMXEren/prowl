## [2.0.0-dev.9](https://github.com/IMXEren/prowl/compare/v2.0.0-dev.8...v2.0.0-dev.9) (2026-10-29)

### Bug Fixes

* **browser:** migrate challenge handling to PyDoll 3 ([82fff0f](https://github.com/IMXEren/prowl/commit/82fff0f1bf4c6f3e5da65bcef83103b2fe8e0bfa))

## [2.0.0-dev.8](https://github.com/IMXEren/prowl/compare/v2.0.0-dev.7...v2.0.0-dev.8) (2026-10-28)

### New Features

* **captcha:** solve challenges on reserved CDP tabs ([b98edbf](https://github.com/IMXEren/prowl/commit/b98edbf6378c94dc8b32294e552aa540d80aca79))

## [2.0.0-dev.7](https://github.com/IMXEren/prowl/compare/v2.0.0-dev.6...v2.0.0-dev.7) (2026-10-22)

### Bug Fixes

* **captcha:** wait for stable GeeTest v4 layout ([05609e6](https://github.com/IMXEren/prowl/commit/05609e669861135dc0e5df82594ab7529e9279e8))

### New Features

* **service:** expose authenticated CDP reservation tunnel ([fd9e439](https://github.com/IMXEren/prowl/commit/fd9e439b045d4d6693040614fd21cbf389d7ee01))
* **service:** reserve browser tabs with protected lifecycle ([43296a2](https://github.com/IMXEren/prowl/commit/43296a2e919b264cb160a288d974834041189e47))

## [2.0.0-dev.6](https://github.com/IMXEren/prowl/compare/v2.0.0-dev.5...v2.0.0-dev.6) (2026-10-18)

### Bug Fixes

* **captcha:** recognize Friendly Captcha agent frames ([4986750](https://github.com/IMXEren/prowl/commit/4986750372d881b20db9fa74468ba7f2fbb0332b))

### New Features

* **captcha:** add opt-in visual challenge adapters ([8caace8](https://github.com/IMXEren/prowl/commit/8caace85b2da846c7aba35b59b5d6d534650eab5))

### Dependencies

* **deps:** bump Pillow to 13.0.0 ([e947492](https://github.com/IMXEren/prowl/commit/e947492c8e4ba047b513f81c5e8c9716d73caa56))
* **deps:** bump pycryptodome to 4.0.1 ([9afc7a3](https://github.com/IMXEren/prowl/commit/9afc7a3f2d020d17eca2a0316945f8e6a1f351c5))
* **deps:** bump ruff to 0.17.1 ([6291344](https://github.com/IMXEren/prowl/commit/62913448d0fb38fa4ef4ae0d9da3b24edf745d00))
* **deps:** bump turbohtml to 1.15.2 ([8f6d80d](https://github.com/IMXEren/prowl/commit/8f6d80de6c373a4fb4722dae1169cbb43dface2a))

## [2.0.0-dev.5](https://github.com/IMXEren/prowl/compare/v2.0.0-dev.4...v2.0.0-dev.5) (2026-10-14)

### Dependencies

* **deps:** bump cloakbrowser to 0.6.3 ([fe3fe47](https://github.com/IMXEren/prowl/commit/fe3fe47ae77d4a3b0830f7a0fa1f2df7ad2b1d20))
* **deps:** bump cryptography to 50.0.2 ([9ed8180](https://github.com/IMXEren/prowl/commit/9ed8180ddb13758487f05441733ca38f26737051))
* **deps:** bump curl_cffi to 0.16.3 ([f5815b2](https://github.com/IMXEren/prowl/commit/f5815b260cfeba2d8fce6167cc0b825008bc2618))
* **deps:** bump pycryptodome to 4.0.0 ([a053b66](https://github.com/IMXEren/prowl/commit/a053b663aa055149515863a5b875421f9b50d11f))
* **deps:** bump ruff to 0.17.0 ([920c5c4](https://github.com/IMXEren/prowl/commit/920c5c4386fe9006461c9a8c3d2c2e8e8f1c8b27))
* **deps:** bump transitive npm to 11.21.0 ([d4520ce](https://github.com/IMXEren/prowl/commit/d4520ceebd4ad773ece52b6637be4ff84e0c7dd2))
* **deps:** bump turbohtml to 1.15.1 ([b6d973f](https://github.com/IMXEren/prowl/commit/b6d973f4721442a5eb1e0ab124ce7bce79e13faa))
* **deps:** update transitive handlebars ([3153e8b](https://github.com/IMXEren/prowl/commit/3153e8bc74479d72d2b679f5d79829f718d446d5))

## [2.0.0-dev.4](https://github.com/IMXEren/prowl/compare/v2.0.0-dev.3...v2.0.0-dev.4) (2026-10-14)

### ⚠ BREAKING CHANGES

* **routing:** Fetch requests without mode now use auto routing instead of browser-only execution. Set mode to browser to retain JavaScript rendering on every fetch.

### New Features

* **routing:** default fetch execution mode to auto ([6ceec3d](https://github.com/IMXEren/prowl/commit/6ceec3d80322c2778d513a9f87d78143c51dca7d))

## [2.0.0-dev.3](https://github.com/IMXEren/prowl/compare/v2.0.0-dev.2...v2.0.0-dev.3) (2026-10-14)

### Bug Fixes

* **typing:** use generator annotations for context managers ([f305bc8](https://github.com/IMXEren/prowl/commit/f305bc864312c6d97c35210a05a8cf9c900d9ee8))

### Refactoring

* replace direct requests helper dependencies ([fafdf0d](https://github.com/IMXEren/prowl/commit/fafdf0d464fdf08a0792a49ae63e19d6dad4a8b5))
* **structures:** relocate and clean up RWLock ([d2e6abf](https://github.com/IMXEren/prowl/commit/d2e6abfdf4bac4bdab4dcfa9ae39fa79734d6fcd))

## [2.0.0-dev.2](https://github.com/IMXEren/prowl/compare/v2.0.0-dev.1...v2.0.0-dev.2) (2026-10-14)

### ⚠ BREAKING CHANGES

* **captcha:** remove tabs_till_verify and turnstile_token. Use solveCaptcha (solve_captcha in Python) and captcha_provider/captcha_token instead.

### New Features

* **captcha:** unify widget solving behind solveCaptcha ([4b4c595](https://github.com/IMXEren/prowl/commit/4b4c595cce6bb6eb3eec2b315a9c33a25000184a))

## [2.0.0-dev.1](https://github.com/IMXEren/prowl/compare/v1.2.0-dev.1...v2.0.0-dev.1) (2026-10-09)

### ⚠ BREAKING CHANGES

* **service:** PageHandler and PageResponse replace the Site and Source Python exports. The old browser.site and browser.egress import paths were removed.

### New Features

* **browser:** add isolated contexts, page handling, and challenge solvers ([56f425b](https://github.com/IMXEren/prowl/commit/56f425b80413ad80327be596f544eec8e5787079))
* **proxy:** add authenticated bridge and forward-proxy primitives ([ace9d95](https://github.com/IMXEren/prowl/commit/ace9d956f89d6ab30498891701e4849e696a03ce))
* **service:** add session-aware routing, persistence, and metrics ([9df9e30](https://github.com/IMXEren/prowl/commit/9df9e30337e6e4a2a57f803890c13bde2e5b6971))

## [1.2.0-dev.1](https://github.com/IMXEren/prowl/compare/v1.1.0...v1.2.0-dev.1) (2026-09-26)

### New Features

* **browser:** isolate named egresses and configure launches ([bfb83d5](https://github.com/IMXEren/prowl/commit/bfb83d58b81791ca3992b79f1b83b9da138a3a7b))
* **service:** support named egresses and interactive browser sessions ([886c203](https://github.com/IMXEren/prowl/commit/886c203e40f6685b8aa3946708f26e1f168ab3d4))

## [1.1.0](https://github.com/IMXEren/prowl/compare/v1.0.0...v1.1.0) (2026-09-23)

### Bug Fixes

* **browser:** isolate custom request headers ([681eadb](https://github.com/IMXEren/prowl/commit/681eadb195314236cb4268cdc1f926cf840fbe88))
* **release:** keep platform manifests a retained image needs ([e98bf35](https://github.com/IMXEren/prowl/commit/e98bf35bf9ef3bde315026c1f9738919f075214c))
* **release:** make the arm64 merge run and push by digest ([f30d258](https://github.com/IMXEren/prowl/commit/f30d258872caa1e0e02a1525821d12459420f3a9))
* **release:** publish the image with a personal access token ([975c118](https://github.com/IMXEren/prowl/commit/975c1186aa347fed1f8db45c2f2ead0498bb3a7e))
* **service:** validate and scope browser headers ([ef553d7](https://github.com/IMXEren/prowl/commit/ef553d7560d78c02240e63290810d48a76344d82))

### New Features

* **release:** publish a native arm64 image ([933bd88](https://github.com/IMXEren/prowl/commit/933bd883aa12b656f6fdf945ad6bf9ae6413747a))
* **release:** publish the private GHCR image ([c161259](https://github.com/IMXEren/prowl/commit/c161259650ca89a09959f401faf5151a86de4781))

## [1.1.0-dev.2](https://github.com/IMXEren/prowl/compare/v1.1.0-dev.1...v1.1.0-dev.2) (2026-09-23)

### Bug Fixes

* **release:** publish the image with a personal access token ([975c118](https://github.com/IMXEren/prowl/commit/975c1186aa347fed1f8db45c2f2ead0498bb3a7e))

## [1.1.0-dev.1](https://github.com/IMXEren/prowl/compare/v1.0.1-dev.1...v1.1.0-dev.1) (2026-09-23)

### Bug Fixes

* **release:** keep platform manifests a retained image needs ([e98bf35](https://github.com/IMXEren/prowl/commit/e98bf35bf9ef3bde315026c1f9738919f075214c))
* **release:** make the arm64 merge run and push by digest ([f30d258](https://github.com/IMXEren/prowl/commit/f30d258872caa1e0e02a1525821d12459420f3a9))

### New Features

* **release:** publish a native arm64 image ([933bd88](https://github.com/IMXEren/prowl/commit/933bd883aa12b656f6fdf945ad6bf9ae6413747a))
* **release:** publish the private GHCR image ([c161259](https://github.com/IMXEren/prowl/commit/c161259650ca89a09959f401faf5151a86de4781))

## [1.0.1-dev.1](https://github.com/IMXEren/prowl/compare/v1.0.0...v1.0.1-dev.1) (2026-09-22)

### Bug Fixes

* **browser:** isolate custom request headers ([681eadb](https://github.com/IMXEren/prowl/commit/681eadb195314236cb4268cdc1f926cf840fbe88))
* **service:** validate and scope browser headers ([ef553d7](https://github.com/IMXEren/prowl/commit/ef553d7560d78c02240e63290810d48a76344d82))

## 1.0.0 (2026-09-22)

### Bug Fixes

* **ci:** lint release scripts on Linux ([32653a2](https://github.com/IMXEren/prowl/commit/32653a2b065b93c36f271b813ad318d2fb3cf761))

### New Features

* add reusable browser automation service ([377de65](https://github.com/IMXEren/prowl/commit/377de651039346c2fa6b0bb5e8641713d99b731c))
* **deploy:** add the local container stack ([06645fb](https://github.com/IMXEren/prowl/commit/06645fbf2818ceb40655c7754ccb958b271c1c35))

## 1.0.0-dev.1 (2026-09-22)

### Bug Fixes

* **ci:** lint release scripts on Linux ([32653a2](https://github.com/IMXEren/prowl/commit/32653a2b065b93c36f271b813ad318d2fb3cf761))

### New Features

* add reusable browser automation service ([377de65](https://github.com/IMXEren/prowl/commit/377de651039346c2fa6b0bb5e8641713d99b731c))
* **deploy:** add the local container stack ([06645fb](https://github.com/IMXEren/prowl/commit/06645fbf2818ceb40655c7754ccb958b271c1c35))

# Changelog

All notable changes to Prowl are documented in this file. It is generated by
[semantic-release](https://github.com/semantic-release/semantic-release) from
[Conventional Commits](https://www.conventionalcommits.org/); edit it only
through release commits.
