"""A minimal JSON-over-HTTP client on ``urllib``, so the daemon needs no HTTP dependency.

Blocking by design: callers run it in a worker thread (``asyncio.to_thread``), never on the loop.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

__all__ = ["HttpError", "request_json", "probe"]


class HttpError(RuntimeError):
    """A request failed: connection refused, timeout, non-2xx status or a non-JSON body."""

    def __init__(self, message: str, status: int | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


def request_json(method: str, url: str, payload: Any = None, *, timeout: float = 10.0) -> Any:
    """Send ``payload`` as JSON (if given) and return the decoded JSON response."""
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            parsed: Any = json.loads(raw)
        except ValueError:
            parsed = raw.decode(errors="replace")
        message = parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            message = parsed["error"].get("message", parsed)
        raise HttpError(f"{method} {url}: HTTP {exc.code}: {message}", exc.code, parsed) from None
    except (urllib.error.URLError, OSError) as exc:
        raise HttpError(f"{method} {url}: {getattr(exc, 'reason', exc)}") from None
    if not body:
        return None
    try:
        return json.loads(body)
    except ValueError:
        raise HttpError(f"{method} {url}: response is not JSON") from None


def probe(url: str, *, timeout: float = 2.0) -> bool:
    """``True`` if ``GET url`` answers 2xx."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False
