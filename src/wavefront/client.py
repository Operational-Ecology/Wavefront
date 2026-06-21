"""The finwave dataset client.

The flow mirrors the finwave dataset API exactly:

1. ``GET /api/datasets-api/{id}/manifest``  → cheap metadata + available formats
2. ``GET /api/datasets-api/{id}?format=...`` → a *handshake* that mints a short-
   lived signed download URL (no bytes yet)
3. download the signed URL → a zip → extract → a :class:`~wavefront.models.Dataset`

Authentication is the ``X-API-KEY`` header; the key needs the dataset-download
scope. Downloads are cached by content fingerprint, so a repeated fetch of the
same frozen version is a no-op.

Every step emits an ``INFO`` log line on the ``wavefront`` logger so a caller
can see exactly what happened; the library installs a ``NullHandler`` and never
configures logging itself.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from .exceptions import (
    APIError,
    AuthError,
    DatasetNotFoundError,
    FormatNotAvailableError,
)
from .models import Dataset, Manifest

log = logging.getLogger("wavefront")

DEFAULT_BASE_URL = "https://finwave.io"
#: Environment variables consulted for the API key, in order. ``FW_API_TOKEN``
#: is the canonical name; the rest are accepted for compatibility.
API_KEY_ENV = ("FW_API_TOKEN", "WAVEFRONT_API_KEY", "FINWAVE_DATASET_API_KEY", "DATASET_API_KEY")
_FORMAT_ALIASES = {"yolo": "Yolo", "coco": "Coco", "pascalvoc": "PascalVoc", "voc": "PascalVoc"}
_COMPLETE_MARKER = ".wavefront-complete"
#: Thread-pool size for concurrent per-image downloads in the manifest path.
_MANIFEST_DOWNLOAD_WORKERS = 16


def _mask(key: str) -> str:
    """A safe-to-log fingerprint of a secret: never the secret itself."""
    return f"{key[:3]}…{key[-2:]} ({len(key)} chars)" if len(key) >= 6 else "set"


def _resolve_key(api_key: Optional[str]) -> tuple[str, str]:
    """Return (key, source) — source is 'argument' or the env var name."""
    if api_key:
        return api_key, "argument"
    for name in API_KEY_ENV:
        v = os.environ.get(name)
        if v:
            return v, name
    raise AuthError(
        "No API key provided. Pass api_key=... or set the FW_API_TOKEN "
        "environment variable (also accepted: " + ", ".join(API_KEY_ENV[1:]) + ")."
    )


def _canonical_format(fmt: str) -> str:
    return _FORMAT_ALIASES.get(fmt.lower(), fmt)


def _default_cache_root() -> Path:
    root = os.environ.get("WAVEFRONT_CACHE")
    if root:
        return Path(root)
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "wavefront"


def _human_bytes(n: Optional[int]) -> str:
    if not n:
        return "?"
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} TB"


class Client:
    """A reusable finwave dataset client.

    Parameters
    ----------
    api_key:
        Dataset-download-scoped key. If omitted, the ``FW_API_TOKEN`` environment
        variable is used (also accepted: ``WAVEFRONT_API_KEY``,
        ``FINWAVE_DATASET_API_KEY``, ``DATASET_API_KEY``).
    base_url:
        finwave base URL (default ``https://finwave.io``).
    timeout:
        Per-request timeout in seconds for the API calls (the large artifact
        download uses a longer, separate timeout).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self.api_key, source = _resolve_key(api_key)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        log.info("wavefront client ready: base_url=%s, key from %s [%s]",
                 self.base_url, source, _mask(self.api_key))

    # ── low-level ────────────────────────────────────────────────────────────
    def _get(self, path: str, **kwargs) -> httpx.Response:
        url = f"{self.base_url}/api/datasets-api/{path}"
        log.debug("GET %s %s", url, kwargs.get("params", ""))
        try:
            resp = httpx.get(url, headers={"X-API-KEY": self.api_key},
                             timeout=self.timeout, **kwargs)
        except httpx.HTTPError as e:  # network-level
            log.error("request to %s failed: %s", url, e)
            raise APIError(f"request to {url} failed: {e}") from e
        log.debug("→ HTTP %d (%s)", resp.status_code, _human_bytes(len(resp.content)))
        if resp.status_code in (401, 403):
            raise AuthError(
                "API key rejected (HTTP %d) — check the key and that it has the "
                "dataset-download scope." % resp.status_code
            )
        return resp

    @staticmethod
    def _error_payload(resp: httpx.Response) -> dict:
        try:
            return resp.json()
        except Exception:
            return {}

    # ── public API ───────────────────────────────────────────────────────────
    def manifest(self, dataset_version_id: str) -> Manifest:
        """Return version metadata + available export formats (no download)."""
        log.info("manifest: requesting %s", dataset_version_id)
        resp = self._get(f"{dataset_version_id}/manifest")
        if resp.status_code == 404:
            raise DatasetNotFoundError(
                f"dataset version {dataset_version_id!r} not found (or not visible to this key)"
            )
        if resp.status_code != 200:
            raise APIError("manifest request failed", status_code=resp.status_code,
                           payload=self._error_payload(resp))
        m = Manifest.from_response(resp.json())
        log.info("manifest: '%s' v%d — %d samples, %d annotations, formats=%s",
                 m.name, m.version_number, m.sample_count, m.annotation_count,
                 m.available_formats or "none yet")
        return m

    def fetch(
        self,
        dataset_version_id: str,
        *,
        format: str = "yolo",
        dest: Optional[os.PathLike] = None,
        cache: bool = True,
        force: bool = False,
        progress: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> Dataset:
        """Fetch + extract a dataset version, returning a :class:`Dataset`.

        Parameters
        ----------
        format:
            Export format, case-insensitive (``"yolo"`` by default).
        dest:
            Directory to extract into. Defaults to the fingerprint-keyed cache.
        cache:
            Reuse a previously-completed download of the same frozen fingerprint.
        force:
            Re-download even if a cached copy exists.
        progress:
            Optional callback ``(bytes_downloaded, total_or_None)`` for the
            artifact download.
        """
        fmt = _canonical_format(format)
        log.info("fetch: %s (format=%s)", dataset_version_id, fmt)
        m = self.manifest(dataset_version_id)
        if not m.has_format(fmt):
            raise FormatNotAvailableError(
                f"format {fmt!r} is not available for '{m.name}'. "
                f"Available: {m.available_formats or 'none yet — an export must be generated'}.",
                available=m.available_formats,
            )

        if dest is not None:
            out = Path(dest)
        else:
            out = _default_cache_root() / f"{dataset_version_id}" / f"{fmt}-{m.fingerprint[:12]}"

        marker = out / _COMPLETE_MARKER
        if cache and not force and marker.exists() and marker.read_text().strip() == m.fingerprint:
            ds = Dataset.from_extracted(root=out, manifest=m, fmt=fmt)
            log.info("fetch: cache hit (fingerprint %s) → %s [%d images]",
                     m.fingerprint[:12], out, ds.num_images)
            return ds

        log.info("fetch: requesting download handshake…")
        handshake = self._handshake(dataset_version_id, fmt)

        manifest_json = handshake.get("manifestJson")
        if manifest_json:
            return self._fetch_manifest(
                manifest_json, out=out, marker=marker, manifest=m, fmt=fmt,
                progress=progress,
            )

        download_url = handshake.get("downloadUrl")
        if not download_url:
            raise APIError("handshake response had neither manifestJson nor downloadUrl",
                           payload=handshake)
        out.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            self._download(download_url, tmp_path, progress=progress)
            log.info("fetch: extracting to %s", out)
            for child in out.iterdir():
                if child.name == _COMPLETE_MARKER:
                    continue
                shutil.rmtree(child) if child.is_dir() else child.unlink()
            with zipfile.ZipFile(tmp_path) as zf:
                zf.extractall(out)
        finally:
            tmp_path.unlink(missing_ok=True)
        marker.write_text(m.fingerprint)
        ds = Dataset.from_extracted(root=out, manifest=m, fmt=fmt)
        log.info("fetch: ready → %s [%d images, %d labels, classes=%s]",
                 out, ds.num_images, ds.num_labels, ds.classes)
        return ds

    # ── internals ────────────────────────────────────────────────────────────
    def _handshake(self, dataset_version_id: str, fmt: str) -> dict:
        """Mint a download handshake and return the full parsed response.

        The response is either the *zip* style (a signed ``downloadUrl``) or the
        *manifest* style (a ``manifestJson`` string enumerating per-image blob
        URLs). The caller inspects which keys are present to pick the path.
        """
        resp = self._get(dataset_version_id, params={"format": fmt})
        if resp.status_code == 404:
            payload = self._error_payload(resp)
            detail = (payload.get("detail") or "").lower()
            if "format" in detail:
                raise FormatNotAvailableError(
                    f"format {fmt!r} has not been produced for this version yet "
                    "(exports are generated separately from freezing)."
                )
            raise DatasetNotFoundError(f"dataset version {dataset_version_id!r} not found")
        if resp.status_code != 200:
            raise APIError("handshake failed", status_code=resp.status_code,
                           payload=self._error_payload(resp))
        body = resp.json()
        if body.get("manifestJson"):
            log.info("handshake: manifest export received")
        elif body.get("downloadUrl"):
            log.info("handshake: signed URL minted (expires %s)",
                     body.get("sasExpiresAt", "soon"))
        return body

    def _fetch_manifest(
        self,
        manifest_json: str,
        *,
        out: Path,
        marker: Path,
        manifest: Manifest,
        fmt: str,
        progress: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> Dataset:
        """Materialise a *manifest*-style export: per-image downloads + labels.

        Each ``Item.Url`` is fetched concurrently into ``{out}/{RelativePath}``,
        then a YOLO label file, ``classes.txt`` and ``data.yaml`` are written and
        the completion marker is stamped with the fingerprint.
        """
        try:
            data: dict[str, Any] = json.loads(manifest_json)
        except (TypeError, ValueError) as e:
            raise APIError(f"handshake manifestJson was not valid JSON: {e}") from e

        items: list[dict] = list(data.get("Items", []) or [])
        classes: list[str] = list(data.get("Classes", []) or [])
        total = len(items)
        log.info("manifest: %d samples, downloading images…", total)

        out.mkdir(parents=True, exist_ok=True)
        for child in out.iterdir():
            if child.name == _COMPLETE_MARKER:
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()

        labels_dir = out / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Download every image concurrently; write its YOLO label alongside.
        done = 0
        if progress is not None:
            progress(done, total)
        with ThreadPoolExecutor(max_workers=_MANIFEST_DOWNLOAD_WORKERS) as pool:
            futures = {pool.submit(self._fetch_item, item, out): item for item in items}
            for future in as_completed(futures):
                future.result()  # re-raise any APIError from the worker
                done += 1
                if progress is not None:
                    progress(done, total)

        (out / "classes.txt").write_text("".join(f"{c}\n" for c in classes))
        self._write_data_yaml(out / "data.yaml", out, classes)
        marker.write_text(manifest.fingerprint)
        ds = Dataset.from_extracted(root=out, manifest=manifest, fmt=fmt)
        log.info("manifest: ready → %s [%d images, %d labels, classes=%s]",
                 out, ds.num_images, ds.num_labels, ds.classes)
        return ds

    def _fetch_item(self, item: dict, out: Path) -> None:
        """Download one manifest item's image and write its YOLO label file."""
        rel = item.get("RelativePath") or ""
        url = item.get("Url")
        if not rel or not url:
            raise APIError(f"manifest item missing RelativePath/Url: {item.get('SampleId')!r}")

        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            resp = httpx.get(url, timeout=httpx.Timeout(None, connect=30.0),
                             follow_redirects=True)
        except httpx.HTTPError as e:
            raise APIError(f"image download failed for {rel}: {e}") from e
        if resp.status_code != 200:
            raise APIError(f"image download failed for {rel} (HTTP {resp.status_code})",
                           status_code=resp.status_code)
        dest.write_bytes(resp.content)

        stem = Path(rel).stem
        lines = [
            f"{b.get('ClassIndex', 0)} {b['X']} {b['Y']} {b['Width']} {b['Height']}"
            for b in (item.get("Boxes") or [])
        ]
        (out / "labels" / f"{stem}.txt").write_text(
            "".join(f"{ln}\n" for ln in lines))

    @staticmethod
    def _write_data_yaml(path: Path, root: Path, classes: list[str]) -> None:
        """Write a minimal Ultralytics-style ``data.yaml`` for the dataset."""
        lines = [
            f"path: {root}",
            "train: images",
            "val: images",
            "names:",
        ]
        lines += [f"  {i}: {c}" for i, c in enumerate(classes)]
        path.write_text("".join(f"{ln}\n" for ln in lines))

    def _download(self, url: str, dest: Path, *,
                  progress: Optional[Callable[[int, Optional[int]], None]] = None) -> None:
        # The download URL is a pre-signed object URL — no API key, long timeout.
        t0 = time.monotonic()
        with httpx.stream("GET", url, timeout=httpx.Timeout(None, connect=30.0),
                          follow_redirects=True) as resp:
            if resp.status_code != 200:
                raise APIError(f"artifact download failed (HTTP {resp.status_code})",
                               status_code=resp.status_code)
            total = int(resp.headers.get("Content-Length", 0)) or None
            log.info("download: %s …", _human_bytes(total))
            got = 0
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(1 << 20):
                    f.write(chunk)
                    got += len(chunk)
                    if progress is not None:
                        progress(got, total)
        dt = time.monotonic() - t0
        rate = got / dt / (1 << 20) if dt > 0 else 0
        log.info("download: %s in %.1fs (%.0f MB/s)", _human_bytes(got), dt, rate)
