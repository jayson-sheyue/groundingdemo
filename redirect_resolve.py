"""Resolve Vertex grounding redirect URLs to final destinations (best effort)."""

from __future__ import annotations

import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Optional

import certifi

REDIRECT_HOST_MARKERS = (
    "vertexaisearch.cloud.google.com",
    "grounding-api-redirect",
)

DEFAULT_TIMEOUT_S = float(os.getenv("REDIRECT_RESOLVE_TIMEOUT_S", "8"))
DEFAULT_MAX_HOPS = int(os.getenv("REDIRECT_RESOLVE_MAX_HOPS", "10"))
DEFAULT_MAX_WORKERS = int(os.getenv("REDIRECT_RESOLVE_MAX_WORKERS", "6"))


def env_resolve_redirects_default() -> bool:
    raw = (os.getenv("RESOLVE_GROUNDING_REDIRECTS") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def should_resolve_url(url: Optional[str]) -> bool:
    if not url:
        return False
    lower = url.lower()
    return any(marker in lower for marker in REDIRECT_HOST_MARKERS)


@lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext:
    """Use certifi CA bundle so redirect resolve works on fresh macOS/Windows/Linux."""
    return ssl.create_default_context(cafile=certifi.where())


@lru_cache(maxsize=1)
def _url_opener() -> urllib.request.OpenerDirector:
    class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
            return None

    https_handler = urllib.request.HTTPSHandler(context=_ssl_context())
    return urllib.request.build_opener(NoRedirectHandler(), https_handler)


def resolve_redirect_url(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> dict[str, Any]:
    """Follow redirect chain; return {original, final, ok, error}."""
    original = (url or "").strip()
    if not original:
        return {"original": url, "final": None, "ok": False, "error": "empty url"}

    if not should_resolve_url(original):
        return {"original": original, "final": original, "ok": True, "error": None}

    opener = _url_opener()
    current = original
    last_error: Optional[str] = None

    for _ in range(max_hops):
        for method in ("HEAD", "GET"):
            req = urllib.request.Request(
                current,
                method=method,
                headers={"User-Agent": "groundingdemo/1.0 (redirect-resolve)"},
            )
            try:
                with opener.open(req, timeout=timeout) as resp:
                    final = resp.geturl() or current
                    return {
                        "original": original,
                        "final": final,
                        "ok": True,
                        "error": None,
                    }
            except urllib.error.HTTPError as exc:
                if exc.code in (301, 302, 303, 307, 308):
                    location = exc.headers.get("Location")
                    if not location:
                        last_error = f"redirect {exc.code} without Location"
                        break
                    current = urllib.parse.urljoin(current, location)
                    last_error = None
                    break
                if exc.code == 405 and method == "HEAD":
                    last_error = None
                    continue
                last_error = f"HTTP {exc.code}"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                break
        else:
            continue
        if last_error is None:
            continue
        break

    if current != original:
        return {
            "original": original,
            "final": current,
            "ok": True,
            "error": None,
        }

    return {
        "original": original,
        "final": None,
        "ok": False,
        "error": last_error or "could not resolve redirect",
    }


def resolve_many(urls: list[str]) -> dict[str, dict[str, Any]]:
    unique = []
    seen: set[str] = set()
    for url in urls:
        if url and url not in seen and should_resolve_url(url):
            seen.add(url)
            unique.append(url)

    if not unique:
        return {}

    results: dict[str, dict[str, Any]] = {}
    workers = min(DEFAULT_MAX_WORKERS, max(1, len(unique)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(resolve_redirect_url, u): u for u in unique}
        for fut in as_completed(futures):
            url = futures[fut]
            try:
                results[url] = fut.result()
            except Exception as exc:  # noqa: BLE001
                results[url] = {
                    "original": url,
                    "final": None,
                    "ok": False,
                    "error": str(exc),
                }
    return results


def enrich_sources(
    sources: list[dict[str, Any]],
    *,
    enabled: bool,
) -> list[dict[str, Any]]:
    """Attach resolved.{uri|source_uri|image_uri} to each source dict."""
    if not enabled or not sources:
        return sources

    urls: list[str] = []
    for source in sources:
        for key in ("uri", "source_uri", "image_uri"):
            value = source.get(key)
            if should_resolve_url(value):
                urls.append(str(value))

    resolved_map = resolve_many(urls)
    enriched: list[dict[str, Any]] = []
    for source in sources:
        item = dict(source)
        resolved_fields: dict[str, Any] = {}
        for key in ("uri", "source_uri", "image_uri"):
            value = source.get(key)
            if value and str(value) in resolved_map:
                resolved_fields[key] = resolved_map[str(value)]
        if resolved_fields:
            item["resolved"] = resolved_fields
        enriched.append(item)
    return enriched
