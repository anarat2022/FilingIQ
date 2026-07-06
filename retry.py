"""
Small retry helper for hosted APIs with tight rate limits (Gemini's free
tier is 5 requests/minute on some models as of mid-2026, which a pipeline
that fires off a request per chunk or per eval question can blow through in
seconds). Exponential backoff turns a hard failure into a short wait.
"""

import time


def with_retry(fn, max_retries: int = 4, base_delay: float = 8.0):
    """Call fn() with exponential backoff on failure. Re-raises the last
    exception if all attempts are exhausted."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, we retry any transient failure
            last_exc = exc
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                print(f"  (rate limited or transient error, retrying in {delay:.0f}s... [{attempt + 1}/{max_retries}])")
                time.sleep(delay)
    raise last_exc
