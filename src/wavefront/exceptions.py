"""Exceptions raised by wavefront.

All inherit from :class:`WavefrontError`, so callers can catch the whole family
with a single ``except WavefrontError``.
"""
from __future__ import annotations


class WavefrontError(Exception):
    """Base class for all wavefront errors."""


class AuthError(WavefrontError):
    """The API key was missing, malformed, or rejected (HTTP 401/403)."""


class DatasetNotFoundError(WavefrontError):
    """No dataset version with the given id is visible to this key (HTTP 404)."""


class FormatNotAvailableError(WavefrontError):
    """The requested export format has not been produced for this version yet.

    The dataset exists but ``availableFormats`` does not include the requested
    format — an admin must generate the export first (it is not produced on
    freeze). :attr:`available` lists what *is* ready.
    """

    def __init__(self, message: str, *, available: list[str] | None = None) -> None:
        super().__init__(message)
        self.available = available or []


class IntegrityError(WavefrontError):
    """A downloaded artifact did not match the fingerprint the server declared."""


class APIError(WavefrontError):
    """An unexpected, non-success response from the finwave API.

    :attr:`status_code` is the HTTP status; :attr:`payload` is the parsed error
    body when the server returned one.
    """

    def __init__(self, message: str, *, status_code: int | None = None,
                 payload: dict | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}
