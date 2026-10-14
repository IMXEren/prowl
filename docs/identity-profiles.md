# Browser profiles

Each egress has its own browser process, persistent profile, and archive. Sessions on the same egress share that browser identity; an isolated session has separate browsing state, not a different proxy or fingerprint.

## Storage layout

`PROWL_PROFILE_DIR` sets the default profile directory. `PROWL_PROFILE_ARCHIVE` sets its ZIP archive. Named egresses use sibling paths derived from those settings:

| Identity | Profile directory | Archive |
| --- | --- | --- |
| Default | `/state/profile` | `/state/browser-profile.zip` |
| Named `work` | `/state/profile-work` | `/state/browser-profile-work.zip` |

These are the Docker paths. Outside Docker, the defaults are `/tmp/browser-profile` and `browser-profile.zip`.

Profiles must not overlap or contain one another. Archives must be separate from all profile trees and from one another. Prowl rejects conflicting paths when loading the egress configuration.

On startup, Prowl restores an archive if the profile directory is empty. Otherwise it uses the existing profile. On shutdown, it packs the profile, excluding caches and journals.

Treat profiles and archives as private data: they contain cookies, login state, and browser storage. Keep each profile paired with its egress, and stop Prowl before moving or restoring it.

## Isolated session state

Shared sessions use the persistent profile. Isolated sessions keep separate cookies and storage inside the same browser process. Their state is discarded when the context closes unless `PROWL_SESSION_STATE_DIR` is configured.

With that setting, Prowl saves isolated state on eviction or orderly shutdown and restores it when the same session ID is recreated on the same egress. Explicitly destroying a session deletes its saved state. This directory needs the same privacy protections as the profiles.

## Legacy development profiles

An earlier development layout put named profiles inside the default profile directory. Only installations using that layout need migration; fresh installs and upgrades from stable v1 do not need this step.

Stop Prowl and ensure no other process is using the profiles. From the repository root, preview the migration first:

```sh
python -m scripts.migrate_identity_profiles
```

After reviewing the plan:

```sh
python -m scripts.migrate_identity_profiles --apply
```

The tool reads the configured egress names. Use `--names` to include retired names, and `--profile-dir` or `--profile-archive` if the paths differ from the environment settings. It does not discover unselected identities.

Migration moves the selected directories to sibling paths and removes their entries from the default archive. Separate named archives are left unchanged. Before rewriting the default archive, it makes a backup; keep that backup until the migrated profiles have been checked.

The tool rejects ambiguous or conflicting layouts. It attempts rollback on failure, but it is not a crash-atomic transaction and does not check for running browser processes.
