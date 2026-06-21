"""Mocked unit tests for the wavefront client (no network)."""
from __future__ import annotations

import io
import json
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


# ── manifest-export path ──────────────────────────────────────────────────────
def _manifest_json():
    return json.dumps({
        "Container": "finwave-detector-datasets",
        "Classes": ["dorsal_fin"],
        "SampleCount": 2,
        "Items": [
            {
                "SampleId": "s1", "BlobKey": "k/s1.jpg",
                "RelativePath": "images/img1.jpg", "Width": 2047, "Height": 1365,
                "Url": "https://blob/img1.jpg",
                "Boxes": [{"Class": "dorsal_fin", "ClassIndex": 0,
                           "X": 0.09, "Y": 0.12, "Width": 0.59, "Height": 0.87}],
            },
            {
                "SampleId": "s2", "BlobKey": "k/s2.jpg",
                "RelativePath": "images/img2.jpg", "Width": 2048, "Height": 1366,
                "Url": "https://blob/img2.jpg",
                "Boxes": [
                    {"Class": "dorsal_fin", "ClassIndex": 0,
                     "X": 0.46, "Y": 0.47, "Width": 0.03, "Height": 0.04},
                    {"Class": "dorsal_fin", "ClassIndex": 0,
                     "X": 0.80, "Y": 0.46, "Width": 0.06, "Height": 0.09},
                ],
            },
        ],
    })


@respx.mock
def test_fetch_manifest_path(tmp_path):
    respx.get(f"{BASE}/api/datasets-api/{DVID}/manifest").mock(
        return_value=httpx.Response(200, json=_manifest_body()))
    respx.get(f"{BASE}/api/datasets-api/{DVID}").mock(
        return_value=httpx.Response(200, json={"manifestJson": _manifest_json()}))
    img1 = respx.get("https://blob/img1.jpg").mock(
        return_value=httpx.Response(200, content=b"\xff\xd8\xff"))
    img2 = respx.get("https://blob/img2.jpg").mock(
        return_value=httpx.Response(200, content=b"\xff\xd8\xff"))

    out = tmp_path / "ds"
    seen: list[tuple[int, int]] = []
    ds = Client("k").fetch(DVID, format="yolo", dest=out,
                           progress=lambda n, t: seen.append((n, t)))

    # images downloaded to images/<RelativePath>
    assert (out / "images" / "img1.jpg").read_bytes() == b"\xff\xd8\xff"
    assert (out / "images" / "img2.jpg").read_bytes() == b"\xff\xd8\xff"
    assert img1.call_count == 1 and img2.call_count == 1

    # one label file per image, one line per box, correct contents
    assert (out / "labels" / "img1.txt").read_text() == "0 0.09 0.12 0.59 0.87\n"
    assert (out / "labels" / "img2.txt").read_text() == (
        "0 0.46 0.47 0.03 0.04\n0 0.8 0.46 0.06 0.09\n")

    # classes.txt + data.yaml
    assert (out / "classes.txt").read_text() == "dorsal_fin\n"
    yaml = (out / "data.yaml").read_text()
    assert "train: images" in yaml and "val: images" in yaml
    assert "names:" in yaml and "0: dorsal_fin" in yaml
    assert f"path: {out}" in yaml

    assert ds.num_images == 2 and ds.num_labels == 2 and ds.classes == ["dorsal_fin"]
    assert ds.fingerprint == FP
    assert seen and seen[-1] == (2, 2)

    # completed cache short-circuits a second fetch (no further image GETs)
    ds2 = Client("k").fetch(DVID, format="yolo", dest=out)
    assert ds2.num_images == 2
    assert img1.call_count == 1 and img2.call_count == 1
