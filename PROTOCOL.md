# PROTOCOL — wavefront

Decision log for the finwave dataset client. Governance: [`../CLAUDE.md`](../CLAUDE.md).

## Mission

### One-line — the official, public Python client for fetching finwave datasets
Status: Committed — 2026-06-21
finwave already exposes a dataset handshake API (`/api/datasets-api/{id}`); every
consumer was re-implementing the manifest → handshake → SAS-download → unzip
dance by hand. `wavefront` is the single, supported way to do it. It is
OpEco's **first public** project (PyPI + public GitHub) and a deliberate
finwave × Operational Ecology partnership artifact, both authored by AB.

## Scope

### v1 fetches frozen versions in declared export formats; it does not create them
Status: Committed — 2026-06-21
The client is read/download only. Generating an export (e.g. producing the YOLO
bundle for a frozen version) is an admin action on the Hub and is out of scope —
hence `FormatNotAvailableError` carries `.available` rather than trying to
trigger generation. The dataset-API key is download-scoped only.

## Design

### Caching is keyed by the server's content fingerprint, not by id alone
Status: Committed — 2026-06-21
Versions are frozen, so a fingerprint pins exact bytes. A completed extraction
writes a `.wavefront-complete` marker containing the fingerprint; a later fetch
with a matching marker is a no-op. This makes `fetch` idempotent and cheap to
call in a training loop without a manual "already downloaded?" guard.

### `Dataset.fingerprint` is surfaced as first-class provenance
Status: Committed — 2026-06-21
Every fetched dataset exposes the version fingerprint so a downstream training
run can record exactly which data produced a model (aligns with OpEco Rule 2,
source-code/data consistency). The client does not recompute the hash
client-side (the algorithm is server-owned); it carries the declared one.

### API-key precedence: explicit arg → `FW_API_TOKEN` → `WAVEFRONT_API_KEY` → `FINWAVE_DATASET_API_KEY` → `DATASET_API_KEY`
Status: Committed — 2026-06-21
`FW_API_TOKEN` is the canonical name; the rest are accepted so existing
finwave/finprint workspace environments keep working unchanged.

## Deferred

### Streaming-to-disk integrity check beyond fingerprint consistency
Status: Open — 2026-06-21
The handshake and manifest fingerprints are checked for consistency, but the
downloaded bytes are not independently re-hashed against a per-file digest (the
export bundle does not yet ship one). Revisit if the Hub starts emitting
per-artifact `sha256` so `IntegrityError` can be raised on a real mismatch.

### Additional export formats (COCO, PascalVoc)
Status: Parked — 2026-06-21
Format aliases are mapped, but only YOLO is produced by the Hub in v1. The
client already accepts any format string and lets the server decide.
