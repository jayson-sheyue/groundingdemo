#!/usr/bin/env python3
"""Web UI for modern Grounding (Search / Image Search / Maps) on Gemini 2.5+."""

from __future__ import annotations

import html
import math
import os
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from threading import Lock
from typing import Any, Optional

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
import google.genai
from google.genai import types as genai_types
from markupsafe import Markup

from demo_grounded_search import (
    GroundedResult,
    GroundedSpan,
    GroundingMode,
    _env,
    build_client,
    cost_from_usage,
    count_tokens_for_request,
    default_model_for_mode,
    default_question_for_mode,
    format_sse,
    grounded_ask,
    grounded_ask_stream,
    is_image_generation_mode,
    make_lat_lng,
    parse_exclude_domains_csv,
    parse_gcs_paths_csv,
    verify_claims,
)
from gcs_upload import upload_bytes_to_gcs
from redirect_resolve import enrich_sources, env_resolve_redirects_default

app = Flask(__name__)
# Jinja's tojson uses Flask's JSON provider. Preserve readable Unicode while
# retaining its HTML-safe escaping for data embedded in the page.
app.json.ensure_ascii = False

_HISTORY_LOCK = Lock()
_RUN_HISTORY: deque[dict[str, Any]] = deque(maxlen=30)

CITY_PRESETS = [
    {"name": "不传坐标", "lat": "", "lng": ""},
    {"name": "上海", "lat": "31.2304", "lng": "121.4737"},
    {"name": "北京", "lat": "39.9042", "lng": "116.4074"},
    {"name": "旧金山", "lat": "37.7749", "lng": "-122.4194"},
    {"name": "纽约", "lat": "40.7128", "lng": "-74.0060"},
]


@lru_cache(maxsize=1)
def get_client():
    return build_client()


def annotate_answer_html(text: str, supports: list[GroundedSpan]) -> str:
    if not text:
        return ""
    if not supports:
        return html.escape(text).replace("\n", "<br>\n")

    intervals: list[tuple[int, int, GroundedSpan]] = []
    encoded = text.encode("utf-8")
    for span in supports:
        start = end = -1
        try:
            # Grounding segment indices are UTF-8 byte offsets, not Python
            # character offsets. A missing start_index means zero.
            byte_start = int(span.start_index or 0)
            byte_end = int(span.end_index) if span.end_index is not None else -1
            if 0 <= byte_start < byte_end <= len(encoded):
                start = len(encoded[:byte_start].decode("utf-8"))
                end = start + len(encoded[byte_start:byte_end].decode("utf-8"))
        except (ValueError, TypeError, UnicodeDecodeError):
            pass
        if start < 0 or end <= start or (span.text and text[start:end] != span.text):
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
        refs = ",".join(str(i) for i in span.chunk_indices) or "?"
        chunk = html.escape(text[start:end])
        parts.append(
            f'<mark class="cite" title="来源 [{refs}]">{chunk}'
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
    verify_text: Optional[str] = None,
    elapsed_ms: Optional[int] = None,
    geo: Optional[dict[str, Any]] = None,
    exclude_domains: Optional[list[str]] = None,
    gcs_paths: Optional[list[str]] = None,
    resolve_redirects: bool = False,
) -> dict[str, Any]:
    suggestions = None
    if result.search_suggestions_html:
        suggestions = Markup(result.search_suggestions_html)
    sources = enrich_sources(result.sources, enabled=resolve_redirects)
    return {
        "question": question,
        "status": result.status,
        "mode": result.mode,
        "model": result.model,
        "warning": result.warning,
        "answer_html": Markup(annotate_answer_html(result.text, result.supports)),
        "queries": result.web_search_queries,
        "image_queries": result.image_search_queries,
        "retrieval_queries": result.retrieval_queries,
        "sources": sources,
        "supports": [asdict(s) for s in result.supports],
        "search_suggestions_html": suggestions,
        "search_entry_point": result.search_entry_point,
        "retrieval_metadata": result.retrieval_metadata,
        "google_maps_widget_context_token": result.google_maps_widget_context_token,
        "source_flagging_uris": result.source_flagging_uris,
        "grounding_field_presence": result.grounding_field_presence,
        "grounding_metadata_raw": result.grounding_metadata_raw,
        "grounding_metadata_api": result.grounding_metadata_api,
        "response_sdk": result.response_sdk,
        "verify_text": verify_text,
        "usage": result.raw_usage,
        "elapsed_ms": elapsed_ms,
        "geo": geo or {"enabled": False},
        "exclude_domains": exclude_domains or [],
        "gcs_paths": gcs_paths or [],
        "generated_images": result.generated_images,
        "demo_kind": result.demo_kind,
        "image_reference_sources": [
            s for s in sources if s.get("kind") == "image"
        ],
        "resolve_redirects": resolve_redirects,
        "usage_cost": cost_from_usage(result.raw_usage),
    }


def _preview(text: str, limit: int = 120) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


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
        "mode": "search",
        "question": "美国下一次可见的日全食是什么时候？",
        "retry": True,
        "verify": False,
        "lat": lat,
        "lng": lng,
        "geo_label": "",
        "exclude_domains": _env("EXCLUDE_DOMAINS"),
        "gcs_paths": "",
        "agent_search_datastore": _env("AGENT_SEARCH_DATASTORE"),
        "resolve_redirects": env_resolve_redirects_default(),
    }


def _render(*, view=None, defaults=None, error=None):
    return render_template(
        "index.html",
        view=view,
        defaults=defaults or _default_form(),
        error=error,
        history=list_history(),
        city_presets=CITY_PRESETS,
        gcs_bucket=_env("GCS_BUCKET"),
    )


def _parse_ask_form() -> tuple[dict[str, Any], Optional[str]]:
    """Parse shared form fields; return (defaults, error)."""
    question = (request.form.get("question") or "").strip()
    model = (request.form.get("model") or _env("GEMINI_MODEL", "gemini-2.5-flash")).strip()
    mode_raw = (request.form.get("mode") or "search").strip()
    # Keep this list in lockstep with the ten visible demo cards/radio buttons.
    _valid_modes = frozenset({"search", "image_search", "maps", "url_context", "agent_search", "rag_engine", "external_search", "elasticsearch", "parallel_web", "enterprise_web"})
    mode: GroundingMode = mode_raw if mode_raw in _valid_modes else "search"  # type: ignore[assignment]
    do_verify = request.form.get("verify") == "on"
    do_retry = request.form.get("retry") == "on"
    lat_raw = (request.form.get("lat") or "").strip()
    lng_raw = (request.form.get("lng") or "").strip()
    geo_label = (request.form.get("geo_label") or "").strip()
    exclude_raw = (request.form.get("exclude_domains") or "").strip()
    exclude_domains = parse_exclude_domains_csv(exclude_raw)
    gcs_raw = (request.form.get("gcs_paths") or "").strip()
    gcs_paths = parse_gcs_paths_csv(gcs_raw)
    agent_search_datastore = (request.form.get("agent_search_datastore") or "").strip()
    grounding_options = {key: (request.form.get(key) or "").strip() or None for key in (
        "rag_corpus", "external_endpoint", "external_api_key", "elastic_index",
        "elastic_template", "partner_api_key", "partner_provider", "rag_top_k",
        "rag_distance", "rag_similarity", "rag_metadata_filter", "rag_file_ids",
        "rag_hybrid_alpha", "rag_ranker_type", "rag_ranker_model", "agent_filter",
        "agent_max_results", "agent_engine", "elastic_num_hits", "search_blocking_confidence",
        "retrieval_language", "temperature", "max_output_tokens",
    )}
    grounding_options["maps_enable_widget"] = True if mode == "maps" and request.form.get("maps_enable_widget") == "on" else None
    resolve_redirects = request.form.get("resolve_redirects") == "on"

    # Never put credentials back into HTML after a POST.  ``_grounding_options``
    # remains request-local for the call itself; only non-secret fields can be
    # used to refill the form on validation/API errors.
    public_grounding_options = {
        key: value for key, value in grounding_options.items()
        if key not in {"external_api_key", "partner_api_key"}
    }
    defaults = {
        "model": model,
        "mode": mode,
        "question": question,
        "verify": do_verify,
        "retry": do_retry,
        "lat": lat_raw,
        "lng": lng_raw,
        "geo_label": geo_label,
        "exclude_domains": exclude_raw,
        "gcs_paths": gcs_raw,
        "agent_search_datastore": agent_search_datastore,
        **public_grounding_options,
        "resolve_redirects": resolve_redirects,
    }

    if not question:
        return defaults, "请输入问题。"

    # Only validate/pass controls relevant to this mode. Hidden values must not
    # unexpectedly change a different tool when the user switches modes.
    numeric_fields = {
        "temperature": (float, 0, 2), "max_output_tokens": (int, 1, None),
    }
    if mode == "rag_engine":
        numeric_fields.update({"rag_top_k": (int, 1, None), "rag_distance": (float, None, None),
                               "rag_similarity": (float, None, None), "rag_hybrid_alpha": (float, 0, 1)})
    if mode == "agent_search":
        numeric_fields["agent_max_results"] = (int, 1, 10)
    if mode == "elasticsearch":
        numeric_fields["elastic_num_hits"] = (int, 1, None)
    for key, (converter, minimum, maximum) in numeric_fields.items():
        raw = grounding_options.get(key)
        if raw is None:
            continue
        try:
            value = converter(raw)
            if not math.isfinite(value) or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
                raise ValueError
        except (ValueError, TypeError):
            return defaults, f"参数 {key} 的数值无效，请检查输入和取值范围。"
        grounding_options[key] = value
    if mode == "rag_engine":
        if grounding_options["rag_distance"] is not None and grounding_options["rag_similarity"] is not None:
            return defaults, "距离阈值和相似度阈值只能填写一个。"
        if bool(grounding_options["rag_ranker_type"]) != bool(grounding_options["rag_ranker_model"]):
            return defaults, "重排类型和重排模型名必须同时填写。"
    else:
        for key in ("rag_top_k", "rag_distance", "rag_similarity", "rag_metadata_filter", "rag_file_ids", "rag_hybrid_alpha", "rag_ranker_type", "rag_ranker_model"):
            grounding_options[key] = None
    if mode != "agent_search":
        grounding_options["agent_filter"] = grounding_options["agent_max_results"] = grounding_options["agent_engine"] = None
    if mode != "elasticsearch":
        grounding_options["elastic_num_hits"] = None
    if mode != "search":
        grounding_options["search_blocking_confidence"] = None

    lat_lng = None
    geo_meta: dict[str, Any] = {"enabled": False, "label": geo_label or None}
    try:
        lat_v = _parse_optional_float(lat_raw)
        lng_v = _parse_optional_float(lng_raw)
    except ValueError:
        return defaults, "纬度/经度必须是数字。"
    if (lat_v is None) ^ (lng_v is None):
        return defaults, "坐标需成对填写，或都留空。"
    if lat_v is not None and lng_v is not None:
        if not (-90.0 <= lat_v <= 90.0 and -180.0 <= lng_v <= 180.0):
            return defaults, "纬度 [-90,90]，经度 [-180,180]。"
        lat_lng = make_lat_lng(lat_v, lng_v)
        geo_meta = {
            "enabled": True,
            "lat": lat_v,
            "lng": lng_v,
            "label": geo_label or f"{lat_v:.4f}, {lng_v:.4f}",
        }

    defaults["_lat_lng"] = lat_lng
    defaults["_geo_meta"] = geo_meta
    defaults["_exclude_domains"] = exclude_domains
    defaults["_gcs_paths"] = gcs_paths
    defaults["_agent_search_datastore"] = agent_search_datastore
    defaults["_grounding_options"] = grounding_options
    return defaults, None


def _mode_uses_exclude_domains(mode: GroundingMode) -> bool:
    return mode == "search"


def _mode_uses_redirect_resolution(mode: GroundingMode) -> bool:
    """Only Google Search / Image Search commonly return Google redirect URIs."""
    return mode in ("search", "image_search")


def _has_verifiable_retrieved_context(sources: list[dict[str, Any]]) -> bool:
    """The second pass is only evidence-based when retrieval text is returned.

    A title/URI alone (typical for web, Maps, and partner search) is not the
    source content and must not be presented as a fact-checking input.
    """
    return any(
        source.get("kind") == "retrievedContext" and bool(source.get("text"))
        for source in sources
    )


@app.get("/")
def index():
    return _render()


@app.get("/api-reference")
def api_reference():
    """Reference reflects installed SDK types; not a promise of service support."""
    names = ["GenerateContentConfig", "Tool", "GoogleSearch", "SearchTypes", "GoogleMaps",
             "GoogleMapsGroundingTypes", "UrlContext", "Retrieval", "RetrievalConfig", "ToolConfig",
             "VertexAISearch", "VertexAISearchDataStoreSpec", "VertexRagStore", "VertexRagStoreRagResource",
             "RagRetrievalConfig", "RagRetrievalConfigFilter", "RagRetrievalConfigHybridSearch",
             "RagRetrievalConfigRanking", "RagRetrievalConfigRankingRankService", "RagRetrievalConfigRankingLlmRanker",
             "ExternalApi", "ExternalApiElasticSearchParams", "ExternalApiSimpleSearchParams", "EnterpriseWebSearch", "ToolParallelAiSearch", "ToolExaAiSearch",
             "GenerateContentResponse", "Candidate", "GenerateContentResponseUsageMetadata", "GroundingMetadata",
             "GroundingChunk", "GroundingChunkWeb", "GroundingChunkImage", "GroundingChunkMaps",
             "GroundingChunkRetrievedContext", "GroundingChunkMapsPlaceAnswerSources", "GroundingChunkMapsRoute",
             "RagChunk", "RagChunkPageSpan", "Segment", "GroundingSupport", "RetrievalMetadata", "SearchEntryPoint", "UrlContextMetadata", "UrlMetadata"]
    entries = []
    for name in names:
        cls = getattr(genai_types, name, None)
        if cls is not None and hasattr(cls, "model_fields"):
            entries.append({"name": name, "fields": [
                {"name": field_name, "alias": f.alias, "description": f.description or "SDK 未提供字段说明。"}
                for field_name, f in cls.model_fields.items()
            ]})
    return render_template("api_reference.html", entries=entries, sdk_version=google.genai.__version__)


@app.post("/history/clear")
def history_clear():
    clear_history()
    return _render()


@app.post("/upload")
def upload_files():
    """Browser upload → GCS bucket (GCS_BUCKET in .env)."""
    files = request.files.getlist("files")
    if not files or all(not (f.filename or "").strip() for f in files):
        return jsonify({"error": "请选择至少一个文件。"}), 400

    uploaded: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for f in files:
        name = (f.filename or "").strip()
        if not name:
            continue
        try:
            data = f.read()
            item = upload_bytes_to_gcs(
                data,
                name,
                content_type=f.content_type or None,
            )
            uploaded.append(item)
        except Exception as exc:  # noqa: BLE001
            errors.append({"filename": name, "error": str(exc)})

    if not uploaded:
        msg = errors[0]["error"] if errors else "上传失败。"
        return jsonify({"error": msg, "errors": errors}), 400

    return jsonify({"uploaded": uploaded, "errors": errors})


@app.post("/ask")
def ask():
    defaults, error = _parse_ask_form()
    if error:
        return _render(defaults=defaults, error=error)

    question = defaults["question"]
    model = defaults["model"]
    mode = defaults["mode"]
    do_verify = defaults["verify"]
    do_retry = defaults["retry"]
    lat_lng = defaults.get("_lat_lng")
    geo_meta = defaults.get("_geo_meta") or {"enabled": False}
    exclude_domains = defaults.get("_exclude_domains") or []
    gcs_paths = defaults.get("_gcs_paths") or []
    agent_search_datastore = defaults.get("_agent_search_datastore") or None
    grounding_options = defaults.get("_grounding_options") or {}
    resolve_redirects = defaults.get("resolve_redirects", False)

    started = time.perf_counter()
    try:
        client = get_client()
        result = grounded_ask(
            client,
            question,
            model=model,
            mode=mode,
            max_retries=1 if do_retry else 0,
            lat_lng=lat_lng,
            exclude_domains=exclude_domains if _mode_uses_exclude_domains(mode) else None,
            gcs_paths=gcs_paths,
            agent_search_datastore=agent_search_datastore,
            **grounding_options,
        )
        verify_text = None
        if (
            do_verify
            and result.status == "ok"
            and _has_verifiable_retrieved_context(result.sources)
            and not is_image_generation_mode(mode)
        ):
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
            verify_text=verify_text,
            elapsed_ms=elapsed_ms,
            geo=geo_meta,
            exclude_domains=exclude_domains if _mode_uses_exclude_domains(mode) else [],
            gcs_paths=gcs_paths,
            resolve_redirects=resolve_redirects and _mode_uses_redirect_resolution(mode),
        )
        push_history(
            {
                "time": datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S"),
                "question": question,
                "preview": _preview(result.text),
                "status": result.status,
                "mode": mode,
                "model": result.model,
                "queries": len(result.web_search_queries),
                "sources": len(result.sources),
                "images": len(result.generated_images),
                "elapsed_ms": elapsed_ms,
                "has_suggestions": bool(result.search_suggestions_html),
                "verified": bool(verify_text),
                "geo": geo_meta.get("label") if geo_meta.get("enabled") else "无坐标",
                "exclude_n": len(exclude_domains) if _mode_uses_exclude_domains(mode) else 0,
                "gcs_n": len(gcs_paths),
                "error": None,
            }
        )
        return _render(view=view, defaults=defaults)
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        push_history(
            {
                "time": datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S"),
                "question": question,
                "preview": str(exc),
                "status": "error",
                "mode": mode,
                "model": model,
                "queries": 0,
                "sources": 0,
                "elapsed_ms": elapsed_ms,
                "has_suggestions": False,
                "verified": False,
                "geo": geo_meta.get("label") if geo_meta.get("enabled") else "无坐标",
                "exclude_n": len(exclude_domains) if _mode_uses_exclude_domains(mode) else 0,
                "gcs_n": len(gcs_paths),
                "error": str(exc),
            }
        )
        return _render(defaults=defaults, error=str(exc))


@app.post("/ask/stream")
def ask_stream():
    defaults, error = _parse_ask_form()
    if error:
        return jsonify({"error": error}), 400

    if is_image_generation_mode(defaults["mode"]):
        return jsonify({
            "error": "Image Search 生图模式不支持流式，请使用「发送」。",
        }), 400

    question = defaults["question"]
    model = defaults["model"]
    mode = defaults["mode"]
    lat_lng = defaults.get("_lat_lng")
    exclude_domains = defaults.get("_exclude_domains") or []
    gcs_paths = defaults.get("_gcs_paths") or []
    agent_search_datastore = defaults.get("_agent_search_datastore") or None
    grounding_options = defaults.get("_grounding_options") or {}
    resolve_redirects = defaults.get("resolve_redirects", False)

    def generate():
        started = time.perf_counter()
        try:
            client = get_client()
            for ev in grounded_ask_stream(
                client,
                question,
                model=model,
                mode=mode,
                lat_lng=lat_lng,
                exclude_domains=exclude_domains if _mode_uses_exclude_domains(mode) else None,
                gcs_paths=gcs_paths,
                agent_search_datastore=agent_search_datastore,
                **grounding_options,
            ):
                event = ev.get("event")
                data = dict(ev.get("data") or {})
                if resolve_redirects and _mode_uses_redirect_resolution(mode) and event in ("grounding", "done"):
                    grounding = data.get("grounding") if event == "done" else data
                    if grounding:
                        g = dict(grounding)
                        g["sources"] = enrich_sources(
                            g.get("sources") or [], enabled=True
                        )
                        if event == "done":
                            data["grounding"] = g
                        else:
                            data = g
                yield format_sse(event, data)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            yield format_sse("elapsed", {"elapsed_ms": elapsed_ms})
        except Exception as exc:  # noqa: BLE001
            yield format_sse("error", {"message": str(exc)})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/count-tokens")
def count_tokens():
    defaults, error = _parse_ask_form()
    if error:
        return jsonify({"error": error}), 400

    try:
        client = get_client()
        payload = count_tokens_for_request(
            client,
            model=defaults["model"],
            question=defaults["question"],
            mode=defaults["mode"],
            lat_lng=defaults.get("_lat_lng"),
            exclude_domains=defaults.get("_exclude_domains") if _mode_uses_exclude_domains(defaults["mode"]) else None,
            gcs_paths=defaults.get("_gcs_paths") or [],
            agent_search_datastore=defaults.get("_agent_search_datastore") or None,
            **(defaults.get("_grounding_options") or {}),
        )
        return jsonify(payload)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


def main() -> None:
    # Cloud Run sets PORT; bind all interfaces there. Local default stays loopback.
    port = int(os.getenv("PORT") or os.getenv("WEB_PORT", "8080"))
    host = os.getenv("WEB_HOST") or ("0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
    debug = os.getenv("FLASK_DEBUG", "0" if os.getenv("PORT") else "1") == "1"
    print(f"Grounding demo UI → http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
