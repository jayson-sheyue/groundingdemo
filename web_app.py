#!/usr/bin/env python3
"""Simple web UI for Grounding with Google Search.

Renders:
- model answer with inline citation highlights
- source list
- Google Search Suggestions HTML exactly as returned (ToS requirement)

Run:
  python web_app.py
  open http://127.0.0.1:8080
"""

from __future__ import annotations

import html
import os
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Optional

from flask import Flask, render_template, request
from markupsafe import Markup

from demo_grounded_search import (
    GroundedResult,
    GroundedSpan,
    _env,
    build_client,
    grounded_ask,
    verify_claims,
)

app = Flask(__name__)


@lru_cache(maxsize=1)
def get_client():
    return build_client()


def annotate_answer_html(text: str, supports: list[GroundedSpan], min_confidence: float) -> str:
    """Escape answer text and wrap grounded spans with citation markers."""
    if not text:
        return ""
    if not supports:
        return html.escape(text).replace("\n", "<br>\n")

    # Build non-overlapping intervals (prefer higher confidence on overlap)
    intervals: list[tuple[int, int, GroundedSpan]] = []
    for span in supports:
        if span.start_index is None or span.end_index is None:
            continue
        start, end = int(span.start_index), int(span.end_index)
        if start < 0 or end <= start or end > len(text):
            # Fall back: locate span.text once
            if span.text and span.text in text:
                start = text.index(span.text)
                end = start + len(span.text)
            else:
                continue
        intervals.append((start, end, span))

    intervals.sort(key=lambda x: (x[0], -(x[1] - x[0])))
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
        conf_s = f"{conf:.2f}" if conf is not None else "?"
        weak_class = " weak" if span.weak else ""
        refs = ",".join(str(i) for i in span.chunk_indices) or "?"
        chunk = html.escape(text[start:end])
        parts.append(
            f'<mark class="cite{weak_class}" title="confidence={conf_s}; sources=[{refs}]">'
            f"{chunk}"
            f'<sup class="cite-ref">[{refs}]</sup></mark>'
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
) -> dict[str, Any]:
    answer_html = annotate_answer_html(result.text, result.supports, min_confidence)
    suggestions = None
    if result.search_suggestions_html:
        # Must render Google-provided HTML/CSS without modification.
        suggestions = Markup(result.search_suggestions_html)

    return {
        "question": question,
        "status": result.status,
        "model": result.model,
        "warning": result.warning,
        "answer_text": result.text,
        "answer_html": Markup(answer_html),
        "queries": result.web_search_queries,
        "sources": result.sources,
        "supports": [asdict(s) for s in result.supports],
        "weak_count": len(result.weak_spans),
        "search_suggestions_html": suggestions,
        "verify_text": verify_text,
        "usage": result.raw_usage,
        "min_confidence": min_confidence,
    }


@app.get("/")
def index():
    return render_template(
        "index.html",
        view=None,
        defaults={
            "model": _env("GEMINI_MODEL", "gemini-2.5-flash"),
            "min_confidence": _env("MIN_CONFIDENCE", "0.65"),
            "question": "美国下一次可见的日全食是什么时候？",
            "retry": True,
            "verify": False,
        },
        error=None,
    )


@app.post("/ask")
def ask():
    question = (request.form.get("question") or "").strip()
    model = (request.form.get("model") or _env("GEMINI_MODEL", "gemini-2.5-flash")).strip()
    try:
        min_confidence = float(request.form.get("min_confidence") or _env("MIN_CONFIDENCE", "0.65"))
    except ValueError:
        min_confidence = 0.65
    do_verify = request.form.get("verify") == "on"
    do_retry = request.form.get("retry") == "on"

    defaults = {
        "model": model,
        "min_confidence": str(min_confidence),
        "question": question,
        "verify": do_verify,
        "retry": do_retry,
    }

    if not question:
        return render_template(
            "index.html",
            view=None,
            defaults=defaults,
            error="请输入问题。",
        )

    try:
        client = get_client()
        result = grounded_ask(
            client,
            question,
            model=model,
            min_confidence=min_confidence,
            max_retries=1 if do_retry else 0,
        )
        verify_text = None
        if do_verify and result.status == "ok" and result.sources:
            verify_text = verify_claims(
                client,
                model=model,
                answer_text=result.text,
                sources=result.sources,
            )
        view = result_to_view(
            result,
            question=question,
            min_confidence=min_confidence,
            verify_text=verify_text,
        )
        return render_template("index.html", view=view, defaults=defaults, error=None)
    except Exception as exc:  # noqa: BLE001 - show in UI for local demo
        return render_template(
            "index.html",
            view=None,
            defaults=defaults,
            error=str(exc),
        )


def main() -> None:
    host = os.getenv("WEB_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_PORT", "8080"))
    print(f"Grounding demo UI → http://{host}:{port}")
    print("Tip: Search Suggestions HTML is rendered exactly as returned by Google.")
    app.run(host=host, port=port, debug=True)


if __name__ == "__main__":
    main()
