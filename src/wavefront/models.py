"""Typed views over the finwave dataset-API responses and the local result.

These are thin, read-only dataclasses; the wire shapes they mirror are the
``/manifest`` and handshake responses of ``/api/datasets-api/{id}``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(frozen=True)
class Manifest:
    """Cheap pre-flight metadata for a dataset version (no download minted)."""

    dataset_version_id: str
    parent_dataset_id: str
    name: str
    version_number: int
    fingerprint: str
    sample_count: int
    annotation_count: int
    includes_negatives: bool
    available_formats: list[str]
    frozen_at: Optional[datetime] = None
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_response(cls, d: dict) -> "Manifest":
        return cls(
            dataset_version_id=d.get("datasetVersionId", ""),
            parent_dataset_id=d.get("parentDatasetId", ""),
            name=d.get("name", ""),
            version_number=int(d.get("versionNumber", 0) or 0),
            fingerprint=d.get("fingerprint", ""),
            sample_count=int(d.get("sampleCount", 0) or 0),
            annotation_count=int(d.get("annotationCount", 0) or 0),
            includes_negatives=bool(d.get("includesNegatives", False)),
            available_formats=list(d.get("availableFormats", []) or []),
            frozen_at=_parse_dt(d.get("frozenAt")),
            raw=d,
        )

    def has_format(self, fmt: str) -> bool:
        return fmt.lower() in {f.lower() for f in self.available_formats}


@dataclass(frozen=True)
class Dataset:
    """A fetched, extracted dataset on local disk.

    ``fingerprint`` is the server's content hash for this exact version — record
    it alongside any model you train so the data is traceable.
    """

    id: str
    name: str
    version: int
    fmt: str
    fingerprint: str
    path: Path
    classes: list[str] = field(default_factory=list)
    num_images: int = 0
    num_labels: int = 0

    def __fspath__(self) -> str:        # usable directly as a path
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)

    @property
    def images_dir(self) -> Path:
        return self.path / "images"

    @property
    def labels_dir(self) -> Path:
        return self.path / "labels"

    @classmethod
    def from_extracted(cls, *, root: Path, manifest: Manifest, fmt: str) -> "Dataset":
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
        imgs = [p for p in root.rglob("*") if p.suffix.lower() in exts]
        labels = [p for p in root.rglob("*.txt") if "label" in str(p.parent).lower()]
        classes: list[str] = []
        cf = next((p for p in root.rglob("classes.txt")), None)
        if cf is not None:
            classes = [ln.strip() for ln in cf.read_text().splitlines() if ln.strip()]
        return cls(
            id=manifest.dataset_version_id,
            name=manifest.name,
            version=manifest.version_number,
            fmt=fmt,
            fingerprint=manifest.fingerprint,
            path=root,
            classes=classes,
            num_images=len(imgs),
            num_labels=len(labels),
        )
