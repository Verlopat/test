"""Provider failure categories and bounded Retry-After parsing."""
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re


class CandidateError(Exception):
    def __init__(self, message: str, *, kind: str = "other",
                 retry_after: float | None = None):
        super().__init__(message)
        self.kind = kind
        self.retry_after = retry_after


def retry_after_seconds(value: str | None, *, now: datetime | None = None) -> float | None:
    if not value:
        return None
    value = str(value).strip()
    try:
        return max(0, min(86400, float(value)))
    except (TypeError, ValueError):
        pass
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        base = now or datetime.now(timezone.utc)
        return max(0, min(86400, (date - base).total_seconds()))
    except (TypeError, ValueError, OverflowError, IndexError):
        return None


CONTEXT_RE = re.compile(
    r"context (?:window|length|limit)|too many (?:input )?tokens|"
    r"(?:input|prompt|request).{0,45}too long|"
    r"maximum (?:context|input) tokens|exceeds? (?:the )?(?:model )?context|"
    r"token limit exceeded|context_length_exceeded|"
    r"input_length_exceeded",
    flags=re.I,
)
MODEL_RE = re.compile(
    r"model.{0,35}(?:not found|not available|does not exist|is unavailable|unknown|invalid)|"
    r"(?:unknown|invalid|unsupported) model|model_not_found",
    flags=re.I,
)


def from_http(status_code: int, response_text: str = "",
              retry_after: str | None = None) -> CandidateError:
    """Inspect error category only; do not include raw remote text in log output."""
    body = response_text[:2400]
    if status_code in (401, 498):
        return CandidateError(f"HTTP {status_code}: invalid API credentials", kind="auth")
    if status_code == 403:
        return CandidateError("HTTP 403: API access denied", kind="auth")
    if status_code == 429:
        return CandidateError("HTTP 429: rate limit/quota", kind="rate_limit",
                              retry_after=retry_after_seconds(retry_after))
    if status_code in (400, 413, 422) and CONTEXT_RE.search(body):
        return CandidateError(f"HTTP {status_code}: context length exceeded",
                              kind="context_length")
    if status_code in (400, 404, 422) and MODEL_RE.search(body):
        return CandidateError(f"HTTP {status_code}: model unavailable",
                              kind="model_unavailable")
    if status_code == 404:
        return CandidateError("HTTP 404: model or endpoint unavailable",
                              kind="model_unavailable")
    if status_code in (408, 504):
        return CandidateError(f"HTTP {status_code}: request timeout", kind="timeout")
    if status_code >= 500:
        return CandidateError(f"HTTP {status_code}: provider/server error", kind="server")
    if status_code in (400, 413, 422):
        return CandidateError(f"HTTP {status_code}: request rejected", kind="request_invalid")
    return CandidateError(f"HTTP {status_code}: request rejected", kind="other")
