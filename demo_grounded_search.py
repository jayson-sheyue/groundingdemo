#!/usr/bin/env python3
"""Grounding with Google Search demo (Vertex AI + google-genai).

Goal: call Gemini with Search grounding and apply application-level
guards that reduce hallucinations as much as practical.

Official references:
- https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-search
- https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/overview
- https://ai.google.dev/gemini-api/docs/google-search
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

SYSTEM_INSTRUCTION = """你是一个注重事实准确性的助手。必须遵守：

1. 优先且仅依据本次 Google 搜索检索到的网页信息回答事实性问题。
2. 关键数字、日期、人名、金额、法规条款必须能对应到来源；无法核对时明确写「未找到可靠来源」，禁止用训练记忆补全。
3. 多来源冲突时并列呈现差异，不要擅自选定一边并装作确定。
4. 区分「事实」与「推断」；推断必须标明。
5. 若搜索结果不足以回答，宁可不答也不要编造。
"""

VERIFY_INSTRUCTION = """你是事实核对员。只能根据给定的「模型断言」与「来源列表」判断支撑情况。
对每条断言输出一行 JSON（不要 markdown）：
{"claim":"...","verdict":"Supported|Unsupported|Partial","reason":"..."}
禁止引入来源之外的知识。若来源不足，判 Unsupported。
"""


@dataclass
class GroundedSpan:
    text: str
    start_index: Optional[int]
    end_index: Optional[int]
    chunk_indices: list[int] = field(default_factory=list)
    confidence_scores: list[float] = field(default_factory=list)
    max_confidence: Optional[float] = None
    weak: bool = False


@dataclass
class GroundedResult:
    status: str  # ok | ungrounded | error
    text: str = ""
    model: str = ""
    web_search_queries: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    supports: list[GroundedSpan] = field(default_factory=list)
    weak_spans: list[GroundedSpan] = field(default_factory=list)
    search_suggestions_html: Optional[str] = None
    warning: Optional[str] = None
    raw_usage: Optional[dict[str, Any]] = None


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def build_client() -> genai.Client:
    project = _env("GOOGLE_CLOUD_PROJECT") or _env("GCP_PROJECT_ID")
    location = _env("GOOGLE_CLOUD_LOCATION") or _env("GCP_LOCATION") or "global"
    if not project:
        raise RuntimeError(
            "Missing GOOGLE_CLOUD_PROJECT. Copy .env.example to .env and set your GCP project."
        )

    # google-genai Vertex mode
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    os.environ["GOOGLE_CLOUD_PROJECT"] = project
    os.environ["GOOGLE_CLOUD_LOCATION"] = location

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=types.HttpOptions(api_version="v1"),
    )


def _parse_exclude_domains() -> list[str]:
    raw = _env("EXCLUDE_DOMAINS")
    if not raw:
        return []
    return [d.strip() for d in raw.split(",") if d.strip()]


def _parse_lat_lng() -> Optional[types.LatLng]:
    raw = _env("USER_LAT_LNG")
    if not raw:
        return None
    try:
        lat_s, lng_s = raw.split(",", 1)
        return types.LatLng(latitude=float(lat_s.strip()), longitude=float(lng_s.strip()))
    except ValueError as exc:
        raise RuntimeError("USER_LAT_LNG must be 'lat,lng', e.g. 31.2304,121.4737") from exc


def _parse_blocking_confidence() -> Optional[types.PhishBlockThreshold]:
    """Vertex-only: block risky sites at/above this phishing confidence."""
    raw = _env("BLOCKING_CONFIDENCE")
    if not raw:
        return None
    try:
        return types.PhishBlockThreshold[raw]
    except KeyError as exc:
        allowed = ", ".join(
            e.name for e in types.PhishBlockThreshold if e.name != "PHISH_BLOCK_THRESHOLD_UNSPECIFIED"
        )
        raise RuntimeError(f"BLOCKING_CONFIDENCE must be one of: {allowed}") from exc


def make_lat_lng(latitude: float, longitude: float) -> types.LatLng:
    return types.LatLng(latitude=latitude, longitude=longitude)


def _build_search_config(
    *,
    temperature: float = 1.0,
    system_instruction: str = SYSTEM_INSTRUCTION,
    lat_lng: Optional[types.LatLng] = None,
) -> types.GenerateContentConfig:
    """Official guidance: temperature 1.0 for ideal Search grounding results.

    lat_lng: optional per-request geographic bias. If omitted, falls back to
    USER_LAT_LNG in env; if still unset, Search runs without location bias.
    """
    exclude = _parse_exclude_domains()
    google_search = types.GoogleSearch(
        exclude_domains=exclude or None,
        blocking_confidence=_parse_blocking_confidence(),
    )
    tool_config = None
    effective_lat_lng = lat_lng if lat_lng is not None else _parse_lat_lng()
    if effective_lat_lng is not None:
        tool_config = types.ToolConfig(
            retrieval_config=types.RetrievalConfig(lat_lng=effective_lat_lng)
        )

    return types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        tools=[types.Tool(google_search=google_search)],
        tool_config=tool_config,
    )


def _chunk_to_dict(chunk: Any) -> Optional[dict[str, Any]]:
    web = getattr(chunk, "web", None)
    if web is None:
        return None
    return {
        "uri": getattr(web, "uri", None),
        "title": getattr(web, "title", None),
        "domain": getattr(web, "domain", None),
    }


def parse_grounding(
    response: types.GenerateContentResponse,
    *,
    model: str,
    min_confidence: float,
) -> GroundedResult:
    text = (response.text or "").strip()
    usage = None
    if response.usage_metadata is not None:
        usage = {
            "prompt_token_count": getattr(response.usage_metadata, "prompt_token_count", None),
            "candidates_token_count": getattr(
                response.usage_metadata, "candidates_token_count", None
            ),
            "total_token_count": getattr(response.usage_metadata, "total_token_count", None),
        }

    if not response.candidates:
        return GroundedResult(
            status="error",
            text=text,
            model=model,
            warning="No candidates in response",
            raw_usage=usage,
        )

    candidate = response.candidates[0]
    gm = getattr(candidate, "grounding_metadata", None)
    if gm is None:
        return GroundedResult(
            status="ungrounded",
            text=text,
            model=model,
            warning=(
                "Missing grounding_metadata. Model may not have used Google Search "
                "(low source relevance / incomplete info / decided not to search)."
            ),
            raw_usage=usage,
        )

    queries = list(getattr(gm, "web_search_queries", None) or [])
    sources: list[dict[str, Any]] = []
    for i, chunk in enumerate(getattr(gm, "grounding_chunks", None) or []):
        item = _chunk_to_dict(chunk)
        if item:
            item["index"] = i
            sources.append(item)

    supports: list[GroundedSpan] = []
    weak: list[GroundedSpan] = []
    for s in getattr(gm, "grounding_supports", None) or []:
        segment = getattr(s, "segment", None)
        seg_text = getattr(segment, "text", None) or ""
        scores = [float(x) for x in (getattr(s, "confidence_scores", None) or [])]
        max_c = max(scores) if scores else None
        span = GroundedSpan(
            text=seg_text,
            start_index=getattr(segment, "start_index", None),
            end_index=getattr(segment, "end_index", None),
            chunk_indices=list(getattr(s, "grounding_chunk_indices", None) or []),
            confidence_scores=scores,
            max_confidence=max_c,
            weak=(max_c is None) or (max_c < min_confidence),
        )
        supports.append(span)
        if span.weak:
            weak.append(span)

    entry = getattr(gm, "search_entry_point", None)
    suggestions = getattr(entry, "rendered_content", None) if entry else None

    return GroundedResult(
        status="ok",
        text=text,
        model=model,
        web_search_queries=queries,
        sources=sources,
        supports=supports,
        weak_spans=weak,
        search_suggestions_html=suggestions,
        warning=(
            f"{len(weak)} span(s) below min_confidence={min_confidence}"
            if weak
            else None
        ),
        raw_usage=usage,
    )


def grounded_ask(
    client: genai.Client,
    question: str,
    *,
    model: str,
    min_confidence: float,
    max_retries: int = 1,
    lat_lng: Optional[types.LatLng] = None,
) -> GroundedResult:
    """Ask with Search grounding; retry once if metadata is missing.

    min_confidence: app-side threshold for labeling weak groundingSupports.
    lat_lng: optional per-request location bias for Search.
    """
    config = _build_search_config(lat_lng=lat_lng)
    last: Optional[GroundedResult] = None

    prompts = [question]
    if max_retries > 0:
        prompts.append(
            question
            + "\n\n[系统补强] 请务必使用 Google 搜索工具检索公开网页后再回答；"
            "若无法检索到可靠来源，请明确说明未找到，不要猜测。"
        )

    for attempt, contents in enumerate(prompts[: max_retries + 1]):
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        result = parse_grounding(response, model=model, min_confidence=min_confidence)
        last = result
        if result.status == "ok":
            if attempt > 0:
                result.warning = (
                    (result.warning + "; " if result.warning else "")
                    + f"succeeded after retry attempt={attempt}"
                )
            return result

    assert last is not None
    return last


def verify_claims(
    client: genai.Client,
    *,
    model: str,
    answer_text: str,
    sources: list[dict[str, Any]],
) -> str:
    """Second-pass check: only use returned sources, no Search tool.

    Note: On Vertex/Enterprise, Search must not be mixed with non-search tools
    in the same request; this is a separate generateContent call without Search.
    """
    source_lines = []
    for s in sources:
        source_lines.append(
            f"- [{s.get('index')}] {s.get('title')} | {s.get('domain')} | {s.get('uri')}"
        )
    prompt = (
        f"模型断言全文：\n{answer_text}\n\n"
        f"来源列表：\n" + ("\n".join(source_lines) or "(空)") + "\n\n"
        "请逐条核对关键事实句。"
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=VERIFY_INSTRUCTION,
            temperature=0.2,
        ),
    )
    return (response.text or "").strip()


def print_report(result: GroundedResult, *, verify_text: Optional[str] = None) -> None:
    print("=" * 72)
    print(f"STATUS : {result.status}")
    print(f"MODEL  : {result.model}")
    if result.warning:
        print(f"WARNING: {result.warning}")
    print("-" * 72)
    print("ANSWER:")
    print(result.text or "(empty)")
    print("-" * 72)
    print("WEB SEARCH QUERIES:")
    if result.web_search_queries:
        for q in result.web_search_queries:
            print(f"  - {q}")
    else:
        print("  (none)")
    print("-" * 72)
    print("SOURCES:")
    if result.sources:
        for s in result.sources:
            print(
                f"  [{s.get('index')}] {s.get('title')} "
                f"({s.get('domain')})\n      {s.get('uri')}"
            )
    else:
        print("  (none)")
    print("-" * 72)
    print("GROUNDING SUPPORTS:")
    if result.supports:
        for span in result.supports:
            flag = "WEAK" if span.weak else "OK"
            conf = (
                f"{span.max_confidence:.3f}" if span.max_confidence is not None else "n/a"
            )
            print(f"  [{flag}] conf={conf} chunks={span.chunk_indices}")
            print(f"         {span.text!r}")
    else:
        print("  (none)")
    if result.search_suggestions_html:
        print("-" * 72)
        print(
            "SEARCH SUGGESTIONS HTML present "
            f"({len(result.search_suggestions_html)} chars). "
            "In production you MUST render searchEntryPoint.renderedContent "
            "exactly as provided (Google ToS)."
        )
    if verify_text:
        print("-" * 72)
        print("SECOND-PASS VERIFICATION:")
        print(verify_text)
    if result.raw_usage:
        print("-" * 72)
        print("USAGE:", json.dumps(result.raw_usage, ensure_ascii=False))
    print("=" * 72)


def to_jsonable(result: GroundedResult) -> dict[str, Any]:
    data = asdict(result)
    # HTML can be huge; keep a flag + length for JSON dumps
    html = data.get("search_suggestions_html")
    if html:
        data["search_suggestions_html"] = {
            "present": True,
            "length": len(html),
            "preview": html[:200],
        }
    return data


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Vertex AI Gemini + Grounding with Google Search demo"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="When is the next total solar eclipse visible from the United States?",
        help="User question",
    )
    parser.add_argument(
        "--model",
        default=_env("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model id",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=float(_env("MIN_CONFIDENCE", "0.65") or "0.65"),
        help="Mark grounding spans below this confidence as weak",
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Do not retry when grounding_metadata is missing",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a second-pass claim verification without Search",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Optional latitude for Search geographic bias",
    )
    parser.add_argument(
        "--lng",
        type=float,
        default=None,
        help="Optional longitude for Search geographic bias",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print machine-readable JSON summary",
    )
    args = parser.parse_args(argv)

    lat_lng = None
    if args.lat is not None or args.lng is not None:
        if args.lat is None or args.lng is None:
            raise SystemExit("Provide both --lat and --lng together.")
        lat_lng = make_lat_lng(args.lat, args.lng)

    client = build_client()
    result = grounded_ask(
        client,
        args.question,
        model=args.model,
        min_confidence=args.min_confidence,
        max_retries=0 if args.no_retry else 1,
        lat_lng=lat_lng,
    )

    verify_text = None
    if args.verify and result.status == "ok" and result.sources:
        verify_text = verify_claims(
            client,
            model=args.model,
            answer_text=result.text,
            sources=result.sources,
        )

    print_report(result, verify_text=verify_text)
    if args.json:
        payload = to_jsonable(result)
        if verify_text:
            payload["verification"] = verify_text
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if result.status == "ok":
        return 0
    if result.status == "ungrounded":
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
