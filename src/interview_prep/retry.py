"""Retry LLM calls on temporary provider errors (overloaded, rate limited)."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable

log = logging.getLogger(__name__)

_TRANSIENT = ("503", "429", "500 internal", "unavailable", "resource_exhausted", "overloaded",
              "high demand", "rate limit", "try again later", "deadline exceeded", "timed out")

DEFAULT_DELAYS = (5, 15, 45, 90)


class QuotaExceeded(RuntimeError):
    """A daily / plan quota is used up: retrying within minutes cannot help."""


_RETRY_IN = re.compile(r"retry in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", re.IGNORECASE)


def quota_error(exc: BaseException) -> QuotaExceeded | None:
    """Turn a provider's daily-quota error into a short, readable QuotaExceeded (else None)."""
    text = str(exc)
    low = text.lower()
    if "quota" not in low:
        return None
    when = _RETRY_IN.search(text)
    h = m = 0
    s = 0.0
    if when:
        h, m, s = int(when.group(1) or 0), int(when.group(2) or 0), float(when.group(3))
    wait = h * 3600 + m * 60 + s
    # Per-minute limits (short waits) are worth retrying; daily limits are not.
    if not ("per day" in low or "perday" in low or wait > 300):
        return None
    reset = f" It resets in about {h}h {m}m." if when else ""
    limit = re.search(r"limit: (\d+), model: ([\w.-]+)", text)
    what = f" ({limit.group(1)} requests/day for {limit.group(2)})" if limit else ""
    return QuotaExceeded(
        f"Your API plan's daily request limit is used up{what}.{reset} Options: wait, enable billing "
        "for your key's project (Google AI Studio -> Billing), or set LLM_MODEL in .env to another model."
    )


def is_transient(exc: BaseException) -> bool:
    if quota_error(exc):
        return False
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT)


def with_retry[T](fn: Callable[[], T], what: str, progress: Callable[[str], None] = lambda m: None,
               delays: tuple[float, ...] = DEFAULT_DELAYS, sleep: Callable[[float], None] = time.sleep) -> T:
    """Call ``fn``; on a transient error wait and try again, up to len(delays) more times."""
    for attempt in range(len(delays) + 1):
        try:
            return fn()
        except Exception as e:
            if quota := quota_error(e):
                raise quota from e
            if attempt == len(delays) or not is_transient(e):
                raise
            wait = delays[attempt]
            log.warning("%s failed (%s); retrying in %ss", what, e, wait)
            progress(f"The AI service is busy ({_short(e)}). Retrying {what} in {wait:g}s...")
            sleep(wait)
    raise AssertionError("unreachable")


def _short(e: BaseException) -> str:
    s = str(e).split("{", 1)[0].strip() or type(e).__name__
    return s[:80]
