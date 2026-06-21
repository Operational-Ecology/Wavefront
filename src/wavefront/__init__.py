"""wavefront — the official Python client for finwave datasets.

finwave (https://finwave.io) serves frozen, versioned wildlife photo-ID and
detector datasets behind a small handshake API. ``wavefront`` turns that into
one call:

    >>> import wavefront
    >>> ds = wavefront.fetch("a7673931-9810-4c52-9654-1c9b1fafb63d", format="yolo")
    >>> ds.path, ds.classes, ds.num_images
    (PosixPath('.../Yolo-81f97dec8667'), ['fluke'], 497)

The key is read from the ``FW_API_TOKEN`` environment variable (or passed
explicitly as ``api_key=``); ``WAVEFRONT_API_KEY``, ``FINWAVE_DATASET_API_KEY``
and ``DATASET_API_KEY`` are also accepted for compatibility. For repeated or
configured use, construct a :class:`Client`.

Every step logs on the ``wavefront`` logger. The library attaches a
``NullHandler`` and never configures logging itself — enable output with
``logging.basicConfig(level=logging.INFO)`` in your application.

Built by Operational Ecology (https://operationalecology.io).
"""
from __future__ import annotations

import logging
from typing import Optional

from .client import API_KEY_ENV, DEFAULT_BASE_URL, Client

logging.getLogger("wavefront").addHandler(logging.NullHandler())
from .exceptions import (
    APIError,
    AuthError,
    DatasetNotFoundError,
    FormatNotAvailableError,
    IntegrityError,
    WavefrontError,
)
from .models import Dataset, Manifest

__version__ = "0.1.0"
__all__ = [
    "fetch",
    "manifest",
    "Client",
    "Dataset",
    "Manifest",
    "WavefrontError",
    "AuthError",
    "DatasetNotFoundError",
    "FormatNotAvailableError",
    "IntegrityError",
    "APIError",
    "DEFAULT_BASE_URL",
    "API_KEY_ENV",
    "__version__",
]


def fetch(dataset_version_id: str, *, format: str = "yolo",
          api_key: Optional[str] = None, base_url: str = DEFAULT_BASE_URL, **kwargs) -> Dataset:
    """Fetch + extract a dataset version with a one-off client. See :meth:`Client.fetch`."""
    return Client(api_key, base_url=base_url).fetch(dataset_version_id, format=format, **kwargs)


def manifest(dataset_version_id: str, *,
             api_key: Optional[str] = None, base_url: str = DEFAULT_BASE_URL) -> Manifest:
    """Return a dataset version's manifest with a one-off client. See :meth:`Client.manifest`."""
    return Client(api_key, base_url=base_url).manifest(dataset_version_id)
