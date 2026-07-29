#!/usr/bin/env python3
"""Simple web UI for Grounding with Google Search.

Renders:
- model answer with inline citation highlights
- weak-confidence explanations + confidence bars
- geographic bias controls (manual / browser geolocation / presets)
- Google Search Suggestions HTML exactly as returned (ToS requirement)
- run history box + submit-button loading feedback
"""

from __future__ import annotations

import html
import os
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from threading import Lock
from typing import Any, Optional

from flask import Flask, render_template, request
from markupsafe import Markup

from demo_grounded_search import (
    GroundedResult,
    GroundedSpan,
    _env,
    build_client,
    grounded_ask,
    make_lat_lng,
    verify_claims,
)

app = Flask(__name__)

_HISTORY_LOCK = Lock()
_RUN_HISTORY: deque[dict[str, Any]] = deque(maxlen=30)

CITY_PRESETS = [
    {"name": "不使用偏置", "lat": "", "lng": ""},
    {"name": "上海", "lat": "31.2304", "lng": "121.4737"},
    {"name": "北京", "lat": "39.9042", "lng": "116.4074"},
    {"name": "旧金山", "lat": "37.7749", "lng": "-122.4194"},
    {"name": "纽约", "lat": "40.7128", "lng": "-74.0060"},
    {"name": "伦敦", "lat": "51.5074", "lng": "-0.1278"},
]


@lru_cache(maxsize=1)
def get_client():
    return build_client()


def annotate_answer_html(text: str, supports: list[GroundedSpan], min_confidence: float) -> str:
    """Escape answer text and wrap grounded spans with citation markers."""
    if not text:
        return ""
    if not supports:
        return html.escape(text).replace("\n", "<br>\n")

    intervals: list[tuple[int, int, GroundedSpan]] = []
    for span in supports:
        if span.start_index is None or span.end_index is None:
            continue
        start, end = int(span.start_index), int(span.end_index)
        if start < 0 or end <= start or end > len(text):
            if span.text and span.text in text:
                start = text.index(span.text)
                end = start + len(span.text)
            else:
                continue
        intervals.append((start, end, span))

    chosen: list[tuple[int, int, GroundedSpan]] = []
    occupied_until = -1
    for start, end, span in sorted(intervals, key=lambda x: x[0]):
        if start < occupied_until:
            continue
        chosen.append((start, end, span))
        occupied_until = end

    parts: list[str] = []
    cursor = 0
    for start, end, span in chosen:
        if cursor < start:
            parts.append(html.escape(text[cursor:start]))
        conf = span.max_confidence
        if span.score_missing:
            conf_s = "无分"
            weak_class = " missing"
            label = "无分"
        else:
            conf_s = f"{conf:.2f}" if conf is not None else "?"
            weak_class = " weak" if span.weak else ""
            label = "弱" if span.weak else "强"
        refs = ",".join(str(i) for i in span.chunk_indices) or "?"
        chunk = html.escape(text[start:end])
        parts.append(
            f'<mark class="cite{weak_class}" '
            f'title="置信度 {conf_s}（阈值 {min_confidence:.2f}）· 来源 [{refs}] · {label}">'
            f"{chunk}"
            f'<sup class="cite-ref">[{refs}·{conf_s}]</sup></mark>'
        )
        cursor = end
    if cursor < len(text):
        parts.append(html.escape(text[cursor:]))
    return "".join(parts).replace("\n", "<br>\n")


def result_to_view(
    result: GroundedResult,
    *,
    question: str,
    min_confidence: float,
    verify_text: Optional[str] = None,
    elapsed_ms: Optional[int] = None,
    geo: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    answer_html = annotate_answer_html(result.text, result.supports, min_confidence)
    suggestions = None
    if result.search_suggestions_html:
        suggestions = Markup(result.search_suggestions_html)

    support_rows = []
    missing_count = 0
    for s in result.supports:
        if s.score_missing:
            missing_count += 1
        support_rows.append(
            {
                **asdict(s),
                "bar_pct": int(round((s.max_confidence or 0) * 100)),
            }
        )

    scored_weak = max(0, len(result.weak_spans) - missing_count)
    strong_count = max(0, len(result.supports) - len(result.weak_spans))

    return {
        "question": question,
        "status": result.status,
        "model": result.model,
        "warning": result.warning,
        "answer_text": result.text,
        "answer_html": Markup(answer_html),
        "queries": result.web_search_queries,
        "sources": result.sources,
        "supports": support_rows,
        "weak_count": scored_weak,
        "missing_score_count": missing_count,
        "strong_count": strong_count,
        "search_suggestions_html": suggestions,
        "verify_text": verify_text,
        "usage": result.raw_usage,
        "min_confidence": min_confidence,
        "elapsed_ms": elapsed_ms,
        "geo": geo or {"enabled": False},
    }


def _preview(text: str, limit: int = 120) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def push_history(entry: dict[str, Any]) -> None:
    with _HISTORY_LOCK:
        _RUN_HISTORY.appendleft(entry)


def list_history() -> list[dict[str, Any]]:
    with _HISTORY_LOCK:
        return list(_RUN_HISTORY)


def clear_history() -> None:
    with _HISTORY_LOCK:
        _RUN_HISTORY.clear()


def _parse_optional_float(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return float(text)


def _default_form() -> dict[str, Any]:
    env_lat_lng = _env("USER_LAT_LNG")
    lat, lng = "", ""
    if env_lat_lng and "," in env_lat_lng:
        a, b = env_lat_lng.split(",", 1)
        lat, lng = a.strip(), b.strip()
    return {
        "model": _env("GEMINI_MODEL", "gemini-2.5-flash"),
        "min_confidence": _env("MIN_CONFIDENCE", "0.65"),
        "question": "上海明天天气怎么样？",
        "retry": True,
        "verify": False,
        "lat": lat,
        "lng": lng,
        "geo_label": "",
    }


def _render(*, view=None, defaults=None, error=None):
    return render_template(
        "index.html",
        view=view,
        defaults=defaults or _default_form(),
        error=error,
        history=list_history(),
        city_presets=CITY_PRESETS,
    )


@app.get("/")
def index():
    return _render()


@app.post("/history/clear")
def history_clear():
    clear_history()
    return _render()


@app.post("/ask")
def ask():
    question = (request.form.get("question") or "").strip()
    model = (request.form.get("model") or _env("GEMINI_MODEL", "gemini-2.5-flash")).strip()
    try:
        min_confidence = float(request.form.get("min_confidence") or _env("MIN_CONFIDENCE", "0.65"))
    except ValueError:
        min_confidence = 0.65
    min_confidence = max(0.0, min(1.0, min_confidence))
    do_verify = request.form.get("verify") == "on"
    do_retry = request.form.get("retry") == "on"
    lat_raw = (request.form.get("lat") or "").strip()
    lng_raw = (request.form.get("lng") or "").strip()
    geo_label = (request.form.get("geo_label") or "").strip()

    defaults = {
        "model": model,
        "min_confidence": f"{min_confidence:.2f}",
        "question": question,
        "verify": do_verify,
        "retry": do_retry,
        "lat": lat_raw,
        "lng": lng_raw,
        "geo_label": geo_label,
    }

    if not question:
        return _render(defaults=defaults, error="请输入问题。")

    lat_lng = None
    geo_meta: dict[str, Any] = {"enabled": False, "label": geo_label or None}
    try:
        lat_v = _parse_optional_float(lat_raw)
        lng_v = _parse_optional_float(lng_raw)
    except ValueError:
        return _render(defaults=defaults, error="纬度/经度必须是数字，例如 31.2304 / 121.4737。")

    if (lat_v is None) ^ (lng_v is None):
        return _render(defaults=defaults, error="地理偏置需要同时填写纬度与经度，或都留空。")
    if lat_v is not None and lng_v is not None:
        if not (-90.0 <= lat_v <= 90.0 and -180.0 <= lng_v <= 180.0):
            return _render(defaults=defaults, error="纬度范围 [-90,90]，经度范围 [-180,180]。")
        lat_lng = make_lat_lng(lat_v, lng_v)
        geo_meta = {
            "enabled": True,
            "lat": lat_v,
            "lng": lng_v,
            "label": geo_label or f"{lat_v:.4f}, {lng_v:.4f}",
        }

    started = time.perf_counter()
    try:
        client = get_client()
        result = grounded_ask(
            client,
            question,
            model=model,
            min_confidence=min_confidence,
            max_retries=1 if do_retry else 0,
            lat_lng=lat_lng,
        )
        verify_text = None
        if do_verify and result.status == "ok" and result.sources:
            verify_text = verify_claims(
                client,
                model=model,
                answer_text=result.text,
                sources=result.sources,
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        view = result_to_view(
            result,
            question=question,
            min_confidence=min_confidence,
            verify_text=verify_text,
            elapsed_ms=elapsed_ms,
            geo=geo_meta,
        )
        push_history(
            {
                "id": f"run-{int(time.time() * 1000)}",
                "time": datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S"),
                "question": question,
                "preview": _preview(result.text),
                "status": result.status,
                "model": result.model,
                "queries": len(result.web_search_queries),
                "sources": len(result.sources),
                "weak_count": len(result.weak_spans),
                "elapsed_ms": elapsed_ms,
                "has_suggestions": bool(result.search_suggestions_html),
                "verified": bool(verify_text),
                "min_confidence": min_confidence,
                "geo": geo_meta.get("label") if geo_meta.get("enabled") else "无偏置",
                "error": None,
            }
        )
        return _render(view=view, defaults=defaults)
    except Exception as exc:  # noqa: BLE001 - show in UI for local demo
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        push_history(
            {
                "id": f"err-{int(time.time() * 1000)}",
                "time": datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S"),
                "question": question,
                "preview": str(exc),
                "status": "error",
                "model": model,
                "queries": 0,
                "sources": 0,
                "weak_count": 0,
                "elapsed_ms": elapsed_ms,
                "has_suggestions": False,
                "verified": False,
                "min_confidence": min_confidence,
                "geo": geo_meta.get("label") if geo_meta.get("enabled") else "无偏置",
                "error": str(exc),
            }
        )
        return _render(defaults=defaults, error=str(exc))


def main() -> None:
    host = os.getenv("WEB_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_PORT", "8080"))
    print(f"Grounding demo UI → http://{host}:{port}")
    print("Tip: Search Suggestions HTML is rendered exactly as returned by Google.")
    app.run(host=host, port=port, debug=True)


if __name__ == "__main__":
    main()
