#!/usr/bin/env python3
"""Modern Grounding demo (Vertex AI + google-genai).

Targets Gemini 2.5+ behavior:
- Google Search grounding (public web)
- optional Google Maps grounding (places / local data)
- optional latLng context (NOT the same as Maps grounding)
- no confidenceScores (empty on 2.5+; ignored)

Docs:
- https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-search
- https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-maps
- https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/overview
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

GroundingMode = Literal["search", "maps"]

SYSTEM_SEARCH = """你是注重事实准确性的助手。必须遵守：
1. 优先且仅依据本次 Google 搜索检索到的网页信息回答事实性问题。
2. 关键数字、日期、人名、金额找不到来源就说「未找到可靠来源」，禁止用训练记忆补全。
3. 多来源冲突时并列呈现，不要装作只有一种答案。
4. 区分事实与推断；推断须标明。
"""

SYSTEM_MAPS = """你是本地生活助手。必须遵守：
1. 优先依据本次 Google Maps 检索到的地点 / 商户 / 地理信息作答。
2. 涉及营业时间、地址、评分等，以 Maps 结果为准；找不到就明确说未找到。
3. 不要编造不存在的门店或电话。
"""

VERIFY_INSTRUCTION = """你是事实核对员。只能根据给定的「模型断言」与「来源列表」判断。
对每条断言输出一行 JSON（不要 markdown）：
{"claim":"...","verdict":"Supported|Unsupported|Partial","reason":"..."}
禁止引入来源之外的知识。
"""


@dataclass
class GroundedSpan:
    text: str
    start_index: Optional[int]
    end_index: Optional[int]
    chunk_indices: list[int] = field(default_factory=list)


@dataclass
class GroundedResult:
    status: str  # ok | ungrounded | error
    mode: GroundingMode = "search"
    text: str = ""
    model: str = ""
    web_search_queries: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    supports: list[GroundedSpan] = field(default_factory=list)
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
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    os.environ["GOOGLE_CLOUD_PROJECT"] = project
    os.environ["GOOGLE_CLOUD_LOCATION"] = location
    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=types.HttpOptions(api_version="v1"),
    )


def make_lat_lng(latitude: float, longitude: float) -> types.LatLng:
    return types.LatLng(latitude=latitude, longitude=longitude)


def _parse_exclude_domains() -> list[str]:
    raw = _env("EXCLUDE_DOMAINS")
    return [d.strip() for d in raw.split(",") if d.strip()] if raw else []


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
    raw = _env("BLOCKING_CONFIDENCE")
    if not raw:
        return None
    try:
        return types.PhishBlockThreshold[raw]
    except KeyError as exc:
        allowed = ", ".join(
            e.name
            for e in types.PhishBlockThreshold
            if e.name != "PHISH_BLOCK_THRESHOLD_UNSPECIFIED"
        )
        raise RuntimeError(f"BLOCKING_CONFIDENCE must be one of: {allowed}") from exc


def _build_config(
    *,
    mode: GroundingMode,
    lat_lng: Optional[types.LatLng] = None,
) -> types.GenerateContentConfig:
    """Build generateContent config for modern Search or Maps grounding.

    lat_lng is optional context for either tool. It is NOT the same as enabling
    Google Maps grounding — Maps grounding requires tools=[googleMaps].
    """
    effective = lat_lng if lat_lng is not None else _parse_lat_lng()
    tool_config = None
    if effective is not None:
        tool_config = types.ToolConfig(
            retrieval_config=types.RetrievalConfig(lat_lng=effective)
        )

    if mode == "maps":
        tools = [types.Tool(google_maps=types.GoogleMaps())]
        system = SYSTEM_MAPS
    else:
        tools = [
            types.Tool(
                google_search=types.GoogleSearch(
                    exclude_domains=_parse_exclude_domains() or None,
                    blocking_confidence=_parse_blocking_confidence(),
                )
            )
        ]
        system = SYSTEM_SEARCH

    return types.GenerateContentConfig(
        system_instruction=system,
        temperature=1.0,  # official recommendation for Search grounding
        tools=tools,
        tool_config=tool_config,
    )


def _chunk_to_dict(chunk: Any, index: int) -> Optional[dict[str, Any]]:
    web = getattr(chunk, "web", None)
    if web is not None:
        return {
            "index": index,
            "kind": "web",
            "uri": getattr(web, "uri", None),
            "title": getattr(web, "title", None),
            "domain": getattr(web, "domain", None),
        }
    maps = getattr(chunk, "maps", None)
    if maps is not None:
        return {
            "index": index,
            "kind": "maps",
            "uri": getattr(maps, "uri", None),
            "title": getattr(maps, "title", None),
            "place_id": getattr(maps, "place_id", None),
            "domain": None,
        }
    return None


def parse_grounding(
    response: types.GenerateContentResponse,
    *,
    model: str,
    mode: GroundingMode,
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
            mode=mode,
            text=text,
            model=model,
            warning="No candidates in response",
            raw_usage=usage,
        )

    gm = getattr(response.candidates[0], "grounding_metadata", None)
    if gm is None:
        return GroundedResult(
            status="ungrounded",
            mode=mode,
            text=text,
            model=model,
            warning=(
                "Missing grounding_metadata. Model may not have used the grounding tool."
            ),
            raw_usage=usage,
        )

    queries = list(getattr(gm, "web_search_queries", None) or [])
    sources: list[dict[str, Any]] = []
    for i, chunk in enumerate(getattr(gm, "grounding_chunks", None) or []):
        item = _chunk_to_dict(chunk, i)
        if item:
            sources.append(item)

    supports: list[GroundedSpan] = []
    for s in getattr(gm, "grounding_supports", None) or []:
        segment = getattr(s, "segment", None)
        supports.append(
            GroundedSpan(
                text=getattr(segment, "text", None) or "",
                start_index=getattr(segment, "start_index", None),
                end_index=getattr(segment, "end_index", None),
                chunk_indices=list(getattr(s, "grounding_chunk_indices", None) or []),
            )
        )

    entry = getattr(gm, "search_entry_point", None)
    suggestions = getattr(entry, "rendered_content", None) if entry else None

    return GroundedResult(
        status="ok",
        mode=mode,
        text=text,
        model=model,
        web_search_queries=queries,
        sources=sources,
        supports=supports,
        search_suggestions_html=suggestions,
        warning=None,
        raw_usage=usage,
    )


def grounded_ask(
    client: genai.Client,
    question: str,
    *,
    model: str,
    mode: GroundingMode = "search",
    max_retries: int = 1,
    lat_lng: Optional[types.LatLng] = None,
) -> GroundedResult:
    config = _build_config(mode=mode, lat_lng=lat_lng)
    last: Optional[GroundedResult] = None

    prompts = [question]
    if max_retries > 0:
        tool_name = "Google Maps" if mode == "maps" else "Google 搜索"
        prompts.append(
            question
            + f"\n\n[系统补强] 请务必使用 {tool_name} 工具检索后再回答；"
            "若无法检索到可靠信息，请明确说明未找到，不要猜测。"
        )

    for attempt, contents in enumerate(prompts[: max_retries + 1]):
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        result = parse_grounding(response, model=model, mode=mode)
        last = result
        if result.status == "ok":
            if attempt > 0:
                result.warning = f"succeeded after retry attempt={attempt}"
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
    source_lines = []
    for s in sources:
        source_lines.append(
            f"- [{s.get('index')}] ({s.get('kind')}) {s.get('title')} | {s.get('uri')}"
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
    print(f"MODE   : {result.mode}")
    print(f"MODEL  : {result.model}")
    if result.warning:
        print(f"WARNING: {result.warning}")
    print("-" * 72)
    print("ANSWER:")
    print(result.text or "(empty)")
    print("-" * 72)
    print("QUERIES:")
    for q in result.web_search_queries or ["(none)"]:
        print(f"  - {q}")
    print("-" * 72)
    print("SOURCES:")
    if result.sources:
        for s in result.sources:
            print(
                f"  [{s.get('index')}] ({s.get('kind')}) {s.get('title')}\n"
                f"      {s.get('uri')}"
            )
    else:
        print("  (none)")
    print("-" * 72)
    print("CITATION SPANS:")
    if result.supports:
        for span in result.supports:
            print(f"  chunks={span.chunk_indices} {span.text!r}")
    else:
        print("  (none)")
    if result.search_suggestions_html:
        print("-" * 72)
        print(
            f"SEARCH SUGGESTIONS HTML present ({len(result.search_suggestions_html)} chars)."
        )
    if verify_text:
        print("-" * 72)
        print("VERIFICATION:")
        print(verify_text)
    if result.raw_usage:
        print("-" * 72)
        print("USAGE:", json.dumps(result.raw_usage, ensure_ascii=False))
    print("=" * 72)


def to_jsonable(result: GroundedResult) -> dict[str, Any]:
    data = asdict(result)
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
        description="Modern Vertex Gemini grounding demo (Search or Maps)"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="美国下一次可见的日全食是什么时候？",
        help="User question",
    )
    parser.add_argument(
        "--mode",
        choices=["search", "maps"],
        default="search",
        help="Grounding tool: googleSearch or googleMaps",
    )
    parser.add_argument(
        "--model",
        default=_env("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model id (2.5+ recommended)",
    )
    parser.add_argument("--no-retry", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lng", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.mode == "maps" and args.question == "美国下一次可见的日全食是什么时候？":
        args.question = "这附近有什么口碑好的咖啡馆？"

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
        mode=args.mode,  # type: ignore[arg-type]
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
