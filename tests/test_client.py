"""Mocked unit tests for the wavefront client (no network)."""
from __future__ import annotations

import io
import zipfile

import httpx
import pytest
import respx

import wavefront
from wavefront import AuthError, Client, FormatNotAvailableError
from wavefront.client import _resolve_key

BASE = "https://finwave.io"
DVID = "11111111-2222-3333-4444-555555555555"
FP = "abc123def456"


def _manifest_body(formats=("Yolo",)):
    return {
        "datasetVersionId": DVID, "parentDatasetId": "p", "name": "Test v1",
        "versionNumber": 1, "fingerprint": FP, "sampleCount": 2,
        "annotationCount": 2, "includesNegatives": False,
        "availableFormats": list(formats), "frozenAt": "2026-06-21T00:00:00+00:00",
    }


def _zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("classes.txt", "head\n")
        zf.writestr("images/a.jpg", b"\xff\xd8\xff")
        zf.writestr("images/b.jpg", b"\xff\xd8\xff")
        zf.writestr("labels/a.txt", "0 0.5 0.5 0.2 0.2\n")
        zf.writestr("labels/b.txt", "0 0.4 0.4 0.1 0.1\n")
    return buf.getvalue()


# ── key resolution ────────────────────────────────────────────────────────────
def test_key_from_argument():
    assert _resolve_key("explicit") == ("explicit", "argument")


def test_key_from_fw_api_token(monkeypatch):
    for v in wavefront.API_KEY_ENV:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("FW_API_TOKEN", "from-env")
    assert _resolve_key(None) == ("from-env", "FW_API_TOKEN")


def test_missing_key_raises(monkeypatch):
    for v in wavefront.API_KEY_ENV:
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(AuthError):
        _resolve_key(None)


# ── manifest + fetch ──────────────────────────────────────────────────────────
@respx.mock
def test_manifest():
    respx.get(f"{BASE}/api/datasets-api/{DVID}/manifest").mock(
        return_value=httpx.Response(200, json=_manifest_body()))
    m = Client("k").manifest(DVID)
    assert m.name == "Test v1" and m.has_format("yolo") and m.fingerprint == FP


@respx.mock
def test_fetch_and_cache(tmp_path):
    respx.get(f"{BASE}/api/datasets-api/{DVID}/manifest").mock(
        return_value=httpx.Response(200, json=_manifest_body()))
    handshake = respx.get(f"{BASE}/api/datasets-api/{DVID}").mock(
        return_value=httpx.Response(200, json={"downloadUrl": "https://blob/x.zip"}))
    dl = respx.get("https://blob/x.zip").mock(
        return_value=httpx.Response(200, content=_zip_bytes()))

    out = tmp_path / "ds"
    ds = Client("k").fetch(DVID, format="yolo", dest=out)
    assert ds.num_images == 2 and ds.num_labels == 2 and ds.classes == ["head"]
    assert ds.fingerprint == FP and (out / "classes.txt").exists()

    # second fetch is served from the completed cache — no second download
    ds2 = Client("k").fetch(DVID, format="yolo", dest=out)
    assert ds2.num_images == 2
    assert dl.call_count == 1 and handshake.call_count == 1


@respx.mock
def test_format_not_available():
    respx.get(f"{BASE}/api/datasets-api/{DVID}/manifest").mock(
        return_value=httpx.Response(200, json=_manifest_body(formats=[])))
    with pytest.raises(FormatNotAvailableError) as ei:
        Client("k").fetch(DVID, format="yolo")
    assert ei.value.available == []


@respx.mock
def test_auth_error():
    respx.get(f"{BASE}/api/datasets-api/{DVID}/manifest").mock(
        return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        Client("k").manifest(DVID)
