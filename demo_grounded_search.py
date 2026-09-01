#!/usr/bin/env python3
"""Modern Grounding demo (Vertex AI + google-genai).

Targets Gemini 2.5+ behavior:
- Google Search grounding (public web)
- Google Image Search grounding (preview; searchTypes.imageSearch)
- optional hybrid web + image search
- optional Google Maps grounding (places / local data)
- optional latLng context (NOT the same as Maps grounding)
- SSE streaming (grounding chunks + usage_metadata)
- Vertex GCS multimodal input (PDF / image via gs://)
- count-tokens + optional cost estimate
- no confidenceScores (empty on 2.5+; ignored)

Docs:
- https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-search
- https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-maps
- https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/overview
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from redirect_resolve import enrich_sources, env_resolve_redirects_default

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

GroundingMode = Literal["search", "image_search", "search_hybrid", "maps"]

DEFAULT_IMAGE_GEN_MODEL = "gemini-3.1-flash-image"
DEFAULT_IMAGE_GEN_QUESTION = (
    "请使用 Google 图片搜索查找真实视觉参考，然后生成一张写实风格照片："
    "墨西哥城阿兹特克体育场（Estadio Azteca）在 2026 世界杯期间的日间外观。"
    "先用 1–2 句说明参考了哪些视觉特征，再输出图片。"
)

MIME_BY_EXT: dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".json": "application/json",
}

SYSTEM_SEARCH = """你是注重事实准确性的助手。必须遵守：
1. 优先且仅依据本次 Google 搜索检索到的网页信息回答事实性问题。
2. 关键数字、日期、人名、金额找不到来源就说「未找到可靠来源」，禁止用训练记忆补全。
3. 多来源冲突时并列呈现，不要装作只有一种答案。
4. 区分事实与推断；推断须标明。
5. 若用户消息中提供了「终端用户位置（lat,lng）」，必须把它当作「这附近 / 本地」的参考位置，
   不要声称无法获取地理位置；搜索与回答应围绕该坐标附近展开。
6. 若用户附带了文档/图片，请结合附件内容与本次 Google 搜索检索结果作答；二者冲突时分别说明。
"""

SYSTEM_MAPS = """你是本地生活助手。必须遵守：
1. 优先依据本次 Google Maps 检索到的地点 / 商户 / 地理信息作答。
2. 涉及营业时间、地址、评分等，以 Maps 结果为准；找不到就明确说未找到。
3. 不要编造不存在的门店或电话。
4. 若用户消息中提供了「终端用户位置（lat,lng）」，必须当作「这附近」的坐标，
   不要声称无法获取地理位置。
5. 若用户附带了文档/图片，请结合附件内容与本次 Google Maps 检索结果作答。
"""

SYSTEM_IMAGE_GEN = """你是 Gemini 3.1 Flash Image 生图助手，已启用 Google 图片搜索接地（imageSearch）。

工作流程（必须遵守）：
1. 收到生图描述后，先通过 Google 图片搜索检索真实世界的视觉参考（地标外观、建筑细节、环境光线等）。
2. 用 1–3 句中文说明参考了哪些视觉元素（不要说「无法搜索」）。
3. 输出一张符合描述的生成图片（IMAGE 输出，不要只给文字列表或 Markdown 假链接）。
4. 检索不到合适参考时如实说明，仍尽量生成；禁止编造虚假 URL。
"""

IMAGE_GEN_RETRY_SUFFIX = (
    "\n\n[系统补强] 必须先调用 Google 图片搜索获取视觉参考，"
    "然后输出一张图片（response 含 IMAGE）。"
    "禁止只输出 Markdown 表格、image_N.png 占位符或假链接。"
)


def is_image_generation_mode(mode: GroundingMode) -> bool:
    return mode == "image_search"


def default_model_for_mode(mode: GroundingMode) -> str:
    if is_image_generation_mode(mode):
        return _env("IMAGE_GEN_MODEL", DEFAULT_IMAGE_GEN_MODEL)
    return _env("GEMINI_MODEL", "gemini-2.5-flash")


def default_question_for_mode(mode: GroundingMode) -> str:
    if mode == "maps":
        return "这附近有什么口碑好的咖啡馆？"
    if is_image_generation_mode(mode):
        return DEFAULT_IMAGE_GEN_QUESTION
    if mode == "search_hybrid":
        return "2026 年世界杯主办城市有哪些？请同时给出相关体育场示意图的来源。"
    return "美国下一次可见的日全食是什么时候？"


def extract_text_from_candidate(candidate: Any) -> str:
    content = getattr(candidate, "content", None)
    if content is None:
        return ""
    texts: list[str] = []
    for part in getattr(content, "parts", None) or []:
        text = getattr(part, "text", None)
        if text:
            texts.append(text)
    return "\n".join(texts).strip()


def extract_generated_images(candidate: Any) -> list[dict[str, Any]]:
    content = getattr(candidate, "content", None)
    if content is None:
        return []
    images: list[dict[str, Any]] = []
    for part in getattr(content, "parts", None) or []:
        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue
        raw = getattr(inline, "data", None)
        if not raw:
            continue
        mime = getattr(inline, "mime_type", None) or "image/png"
        if isinstance(raw, bytes):
            b64 = base64.b64encode(raw).decode("ascii")
        else:
            b64 = str(raw)
        images.append(
            {
                "index": len(images),
                "mime_type": mime,
                "data_uri": f"data:{mime};base64,{b64}",
            }
        )
    return images


def _with_location_hint(question: str, lat_lng: Optional[types.LatLng]) -> str:
    """latLng in toolConfig biases retrieval; models often do NOT see it as chat context.

    Inject an explicit hint so '附近' questions don't get a false 'I have no location' reply.
    """
    if lat_lng is None:
        return question
    lat = getattr(lat_lng, "latitude", None)
    lng = getattr(lat_lng, "longitude", None)
    if lat is None or lng is None:
        return question
    return (
        f"{question}\n\n"
        f"[终端用户位置] latitude={lat}, longitude={lng}。"
        "请将「这附近 / 本地 / near me」解释为该坐标附近，不要说无法获取地理位置。"
    )
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
    confidence_scores: list[float] = field(default_factory=list)


@dataclass
class ParsedGroundingMetadata:
    web_search_queries: list[str] = field(default_factory=list)
    image_search_queries: list[str] = field(default_factory=list)
    retrieval_queries: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    supports: list[GroundedSpan] = field(default_factory=list)
    search_suggestions_html: Optional[str] = None
    search_entry_point: Optional[dict[str, Any]] = None
    retrieval_metadata: Optional[dict[str, Any]] = None
    google_maps_widget_context_token: Optional[str] = None
    source_flagging_uris: list[dict[str, Any]] = field(default_factory=list)
    field_presence: dict[str, bool] = field(default_factory=dict)

    def to_raw_dict(self) -> dict[str, Any]:
        """Structured groundingMetadata for UI / logs (no rendered HTML blob)."""
        supports = [
            {
                "text": s.text,
                "start_index": s.start_index,
                "end_index": s.end_index,
                "chunk_indices": s.chunk_indices,
                "confidence_scores": s.confidence_scores or None,
            }
            for s in self.supports
        ]
        return {
            "webSearchQueries": self.web_search_queries,
            "imageSearchQueries": self.image_search_queries,
            "retrievalQueries": self.retrieval_queries,
            "groundingChunks": self.sources,
            "groundingSupports": supports,
            "searchEntryPoint": self.search_entry_point,
            "retrievalMetadata": self.retrieval_metadata,
            "googleMapsWidgetContextToken": self.google_maps_widget_context_token,
            "sourceFlaggingUris": self.source_flagging_uris,
            "_fieldPresence": self.field_presence,
        }


@dataclass
class GroundedResult:
    status: str  # ok | ungrounded | error
    mode: GroundingMode = "search"
    text: str = ""
    model: str = ""
    web_search_queries: list[str] = field(default_factory=list)
    image_search_queries: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    supports: list[GroundedSpan] = field(default_factory=list)
    search_suggestions_html: Optional[str] = None
    generated_images: list[dict[str, Any]] = field(default_factory=list)
    demo_kind: str = "text_qa"  # text_qa | image_gen
    retrieval_queries: list[str] = field(default_factory=list)
    retrieval_metadata: Optional[dict[str, Any]] = None
    google_maps_widget_context_token: Optional[str] = None
    source_flagging_uris: list[dict[str, Any]] = field(default_factory=list)
    search_entry_point: Optional[dict[str, Any]] = None
    grounding_field_presence: dict[str, bool] = field(default_factory=dict)
    grounding_metadata_raw: Optional[dict[str, Any]] = None
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


def guess_mime_type(path_or_uri: str) -> str:
    suffix = Path(path_or_uri.split("?", 1)[0].split("#", 1)[0]).suffix.lower()
    return MIME_BY_EXT.get(suffix, "application/octet-stream")


def resolve_gcs_uri(path_or_uri: str) -> str:
    """Resolve a GCS object path or full gs:// URI for Vertex multimodal input."""
    raw = path_or_uri.strip()
    if not raw:
        raise ValueError("Empty GCS path")
    if raw.startswith("gs://"):
        return raw
    bucket = _env("GCS_BUCKET")
    if not bucket:
        raise RuntimeError(
            "Relative GCS path requires GCS_BUCKET in .env (or pass a full gs:// URI)."
        )
    prefix = _env("GCS_PREFIX").strip("/")
    object_path = raw.lstrip("/")
    if prefix:
        object_path = f"{prefix}/{object_path}" if not object_path.startswith(prefix + "/") else object_path
    return f"gs://{bucket}/{object_path}"


def parse_gcs_paths_csv(raw: str) -> list[str]:
    """Parse comma/newline separated GCS paths or gs:// URIs."""
    if not raw:
        return []
    paths: list[str] = []
    for chunk in raw.replace("\n", ",").split(","):
        item = chunk.strip()
        if item and item not in paths:
            paths.append(item)
    return paths


def build_multimodal_parts(gcs_paths: list[str]) -> list[types.Part]:
    parts: list[types.Part] = []
    for path in gcs_paths:
        uri = resolve_gcs_uri(path)
        parts.append(
            types.Part.from_uri(file_uri=uri, mime_type=guess_mime_type(path))
        )
    return parts


def _with_multimodal_hint(question: str, gcs_paths: Optional[list[str]]) -> str:
    if not gcs_paths:
        return question
    return (
        f"{question}\n\n"
        f"[用户附件] 本次附带 {len(gcs_paths)} 个文件（PDF/图片/音视频等）。"
        "请结合附件内容与本次工具检索结果作答。"
    )


def build_user_contents(
    question: str,
    *,
    lat_lng: Optional[types.LatLng] = None,
    gcs_paths: Optional[list[str]] = None,
) -> list[Any]:
    """Build generate_content contents: optional GCS files + text prompt."""
    text = _with_location_hint(question, lat_lng)
    text = _with_multimodal_hint(text, gcs_paths)
    parts: list[Any] = []
    for part in build_multimodal_parts(gcs_paths or []):
        parts.append(part)
    parts.append(text)
    return parts


def parse_exclude_domains_csv(raw: str) -> list[str]:
    """Parse comma/whitespace separated domains from UI or CLI."""
    if not raw:
        return []
    parts: list[str] = []
    for chunk in raw.replace("\n", ",").split(","):
        d = chunk.strip().lower()
        if not d:
            continue
        # Allow users to paste https://example.com/path → example.com
        if "://" in d:
            d = d.split("://", 1)[1]
        d = d.split("/", 1)[0].split("?", 1)[0].strip(".")
        if d.startswith("www."):
            d = d[4:]
        if d and d not in parts:
            parts.append(d)
    return parts


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


def _google_search_tool(
    mode: GroundingMode,
    *,
    exclude_domains: Optional[list[str]] = None,
) -> types.GoogleSearch:
    """Build GoogleSearch tool for web, image, or hybrid search types."""
    if mode == "image_search":
        return types.GoogleSearch(
            search_types=types.SearchTypes(
                image_search=types.ImageSearch(),
            ),
        )
    if mode == "search_hybrid":
        domains = exclude_domains if exclude_domains is not None else _parse_exclude_domains()
        return types.GoogleSearch(
            search_types=types.SearchTypes(
                web_search=types.WebSearch(),
                image_search=types.ImageSearch(),
            ),
            exclude_domains=domains or None,
            blocking_confidence=_parse_blocking_confidence(),
        )
    domains = exclude_domains if exclude_domains is not None else _parse_exclude_domains()
    return types.GoogleSearch(
        exclude_domains=domains or None,
        blocking_confidence=_parse_blocking_confidence(),
    )


def _build_config(
    *,
    mode: GroundingMode,
    lat_lng: Optional[types.LatLng] = None,
    exclude_domains: Optional[list[str]] = None,
) -> types.GenerateContentConfig:
    """Build generateContent config for modern Search or Maps grounding.

    lat_lng is optional context for either tool. It is NOT the same as enabling
    Google Maps grounding — Maps grounding requires tools=[googleMaps].

    exclude_domains: Vertex-only for googleSearch (not Gemini Developer API).
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
    elif mode == "image_search":
        tools = [types.Tool(google_search=_google_search_tool(mode))]
        system = SYSTEM_IMAGE_GEN
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=1.0,
            tools=tools,
            tool_config=tool_config,
            response_modalities=["TEXT", "IMAGE"],
        )
    else:
        tools = [types.Tool(google_search=_google_search_tool(mode, exclude_domains=exclude_domains))]
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
            "place_id": getattr(maps, "place_id", None) or getattr(maps, "placeId", None),
            "domain": None,
        }
    image = getattr(chunk, "image", None)
    if image is not None:
        source_uri = getattr(image, "source_uri", None)
        image_uri = getattr(image, "image_uri", None)
        return {
            "index": index,
            "kind": "image",
            "source_uri": source_uri,
            "image_uri": image_uri,
            "uri": source_uri or image_uri,
            "title": getattr(image, "title", None),
            "domain": getattr(image, "domain", None),
        }
    retrieved = getattr(chunk, "retrieved_context", None)
    if retrieved is not None:
        return {
            "index": index,
            "kind": "retrievedContext",
            "uri": getattr(retrieved, "uri", None),
            "title": getattr(retrieved, "title", None),
            "text": getattr(retrieved, "text", None),
            "document_name": getattr(retrieved, "document_name", None),
            "page_number": getattr(retrieved, "page_number", None),
        }
    return None


def _retrieval_metadata_to_dict(rm: Any) -> Optional[dict[str, Any]]:
    if rm is None:
        return None
    score = getattr(rm, "google_search_dynamic_retrieval_score", None)
    if score is None:
        return None
    return {"googleSearchDynamicRetrievalScore": score}


def _search_entry_point_to_dict(entry: Any) -> Optional[dict[str, Any]]:
    if entry is None:
        return None
    rendered = getattr(entry, "rendered_content", None)
    sdk_blob = getattr(entry, "sdk_blob", None)
    if not rendered and not sdk_blob:
        return None
    data: dict[str, Any] = {}
    if rendered:
        data["renderedContent"] = {
            "present": True,
            "length": len(rendered),
            "preview": rendered[:200],
        }
    if sdk_blob:
        data["sdkBlob"] = {
            "present": True,
            "length": len(str(sdk_blob)),
        }
    return data or None


def _source_flagging_to_dict(item: Any) -> dict[str, Any]:
    return {
        "sourceId": getattr(item, "source_id", None),
        "flagContentUri": getattr(item, "flag_content_uri", None),
    }


def parse_grounding_metadata(gm: Any) -> ParsedGroundingMetadata:
    """Parse official groundingMetadata fields from API response."""
    web_q = list(getattr(gm, "web_search_queries", None) or [])
    image_q = list(getattr(gm, "image_search_queries", None) or [])
    retrieval_q = list(getattr(gm, "retrieval_queries", None) or [])

    sources: list[dict[str, Any]] = []
    for i, chunk in enumerate(getattr(gm, "grounding_chunks", None) or []):
        item = _chunk_to_dict(chunk, i)
        if item:
            sources.append(item)

    supports: list[GroundedSpan] = []
    for s in getattr(gm, "grounding_supports", None) or []:
        segment = getattr(s, "segment", None)
        scores = list(getattr(s, "confidence_scores", None) or [])
        supports.append(
            GroundedSpan(
                text=getattr(segment, "text", None) or "",
                start_index=getattr(segment, "start_index", None),
                end_index=getattr(segment, "end_index", None),
                chunk_indices=list(getattr(s, "grounding_chunk_indices", None) or []),
                confidence_scores=scores,
            )
        )

    entry = getattr(gm, "search_entry_point", None)
    suggestions = getattr(entry, "rendered_content", None) if entry else None
    entry_dict = _search_entry_point_to_dict(entry)
    retrieval_meta = _retrieval_metadata_to_dict(getattr(gm, "retrieval_metadata", None))
    maps_token = getattr(gm, "google_maps_widget_context_token", None)
    flagging = [
        _source_flagging_to_dict(x)
        for x in (getattr(gm, "source_flagging_uris", None) or [])
    ]

    presence = {
        "webSearchQueries": bool(web_q),
        "imageSearchQueries": bool(image_q),
        "retrievalQueries": bool(retrieval_q),
        "groundingChunks": bool(sources),
        "groundingSupports": bool(supports),
        "searchEntryPoint": bool(entry_dict or suggestions),
        "retrievalMetadata": bool(retrieval_meta),
        "googleMapsWidgetContextToken": bool(maps_token),
        "sourceFlaggingUris": bool(flagging),
    }

    return ParsedGroundingMetadata(
        web_search_queries=web_q,
        image_search_queries=image_q,
        retrieval_queries=retrieval_q,
        sources=sources,
        supports=supports,
        search_suggestions_html=suggestions,
        search_entry_point=entry_dict,
        retrieval_metadata=retrieval_meta,
        google_maps_widget_context_token=maps_token,
        source_flagging_uris=flagging,
        field_presence=presence,
    )


def _apply_parsed_grounding(result: GroundedResult, parsed: ParsedGroundingMetadata) -> None:
    result.web_search_queries = parsed.web_search_queries
    result.image_search_queries = parsed.image_search_queries
    result.retrieval_queries = parsed.retrieval_queries
    result.sources = parsed.sources
    result.supports = parsed.supports
    result.search_suggestions_html = parsed.search_suggestions_html
    result.search_entry_point = parsed.search_entry_point
    result.retrieval_metadata = parsed.retrieval_metadata
    result.google_maps_widget_context_token = parsed.google_maps_widget_context_token
    result.source_flagging_uris = parsed.source_flagging_uris
    result.grounding_field_presence = parsed.field_presence
    result.grounding_metadata_raw = parsed.to_raw_dict()


def parse_grounding(
    response: types.GenerateContentResponse,
    *,
    model: str,
    mode: GroundingMode,
) -> GroundedResult:
    usage = None
    if response.usage_metadata is not None:
        usage = usage_metadata_to_dict(response.usage_metadata)

    if not response.candidates:
        text = (response.text or "").strip()
        return GroundedResult(
            status="error",
            mode=mode,
            demo_kind="image_gen" if is_image_generation_mode(mode) else "text_qa",
            text=text,
            model=model,
            warning="No candidates in response",
            raw_usage=usage,
        )

    candidate = response.candidates[0]
    text = extract_text_from_candidate(candidate) or (response.text or "").strip()
    generated_images = (
        extract_generated_images(candidate) if is_image_generation_mode(mode) else []
    )
    gm = getattr(candidate, "grounding_metadata", None)

    if gm is not None:
        parsed = parse_grounding_metadata(gm)
        result = GroundedResult(
            status="ok",
            mode=mode,
            demo_kind="image_gen" if is_image_generation_mode(mode) else "text_qa",
            text=text,
            model=model,
            generated_images=generated_images,
            warning=None,
            raw_usage=usage,
        )
        _apply_parsed_grounding(result, parsed)
        return result

    if is_image_generation_mode(mode) and generated_images:
        return GroundedResult(
            status="ok",
            mode=mode,
            demo_kind="image_gen",
            text=text,
            model=model,
            generated_images=generated_images,
            warning=(
                "已生成图片，但 API 未返回 grounding_metadata。"
                "参考图来源（imageSearchQueries / groundingChunks）可能为空——"
                "这是 image 生图模型的常见行为，不代表未调用图片搜索。"
            ),
            raw_usage=usage,
        )

    warning = "Missing grounding_metadata. Model may not have used the grounding tool."
    if is_image_generation_mode(mode):
        warning = (
            "未生成图片，且无 grounding_metadata。"
            "请确认模型为 gemini-3.1-flash-image，并描述要「生成」的画面而非「列出图片来源」。"
        )
    return GroundedResult(
        status="ungrounded",
        mode=mode,
        demo_kind="image_gen" if is_image_generation_mode(mode) else "text_qa",
        text=text,
        model=model,
        generated_images=generated_images,
        warning=warning,
        raw_usage=usage,
    )


def usage_metadata_to_dict(usage: Any) -> dict[str, Any]:
    return {
        "prompt_token_count": getattr(usage, "prompt_token_count", None),
        "candidates_token_count": getattr(usage, "candidates_token_count", None),
        "total_token_count": getattr(usage, "total_token_count", None),
        "thoughts_token_count": getattr(usage, "thoughts_token_count", None),
        "cached_content_token_count": getattr(usage, "cached_content_token_count", None),
    }


def grounding_metadata_to_dict(gm: Any) -> dict[str, Any]:
    parsed = parse_grounding_metadata(gm)
    data = parsed.to_raw_dict()
    if parsed.search_suggestions_html:
        data["search_suggestions_html"] = {
            "present": True,
            "length": len(parsed.search_suggestions_html),
        }
    # SSE / stream consumers expect these keys
    data["web_search_queries"] = parsed.web_search_queries
    data["image_search_queries"] = parsed.image_search_queries
    data["retrieval_queries"] = parsed.retrieval_queries
    data["sources"] = parsed.sources
    data["supports"] = data["groundingSupports"]
    data["search_entry_point"] = parsed.search_entry_point
    data["retrieval_metadata"] = parsed.retrieval_metadata
    data["google_maps_widget_context_token"] = parsed.google_maps_widget_context_token
    data["source_flagging_uris"] = parsed.source_flagging_uris
    data["field_presence"] = parsed.field_presence
    return data


def format_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _env_float(name: str) -> Optional[float]:
    raw = _env(name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def estimate_cost_usd(*, input_tokens: int, output_tokens: int = 0) -> Optional[dict[str, Any]]:
    in_price = _env_float("GEMINI_PRICE_INPUT_PER_MTOK")
    out_price = _env_float("GEMINI_PRICE_OUTPUT_PER_MTOK")
    if in_price is None and out_price is None:
        return None
    in_cost = (input_tokens / 1_000_000) * (in_price or 0.0)
    out_cost = (output_tokens / 1_000_000) * (out_price or 0.0)
    return {
        "input_usd": round(in_cost, 6),
        "output_usd": round(out_cost, 6),
        "total_usd": round(in_cost + out_cost, 6),
        "currency": "USD",
    }


def cost_from_usage(usage: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Compute USD cost from actual usage_metadata fields."""
    if not usage:
        return None
    prompt = int(usage.get("prompt_token_count") or 0)
    output = int(usage.get("candidates_token_count") or 0)
    thoughts = int(usage.get("thoughts_token_count") or 0)
    output_total = output + thoughts
    cost = estimate_cost_usd(input_tokens=prompt, output_tokens=output_total)
    if not cost:
        return None
    cost["prompt_token_count"] = prompt
    cost["candidates_token_count"] = output
    cost["thoughts_token_count"] = thoughts or None
    cost["total_token_count"] = usage.get("total_token_count")
    cost["note"] = "Based on actual usage_metadata from the API response."
    return cost


def count_tokens_for_request(
    client: genai.Client,
    *,
    model: str,
    question: str,
    mode: GroundingMode = "search",
    lat_lng: Optional[types.LatLng] = None,
    exclude_domains: Optional[list[str]] = None,
    gcs_paths: Optional[list[str]] = None,
) -> dict[str, Any]:
    effective_lat_lng = lat_lng if lat_lng is not None else _parse_lat_lng()
    contents = build_user_contents(
        question, lat_lng=effective_lat_lng, gcs_paths=gcs_paths
    )
    gen_config = _build_config(
        mode=mode,
        lat_lng=effective_lat_lng,
        exclude_domains=exclude_domains,
    )
    count_config = types.CountTokensConfig(
        system_instruction=gen_config.system_instruction,
        tools=gen_config.tools,
    )
    if is_image_generation_mode(mode) and getattr(gen_config, "response_modalities", None):
        count_config = types.CountTokensConfig(
            system_instruction=gen_config.system_instruction,
            tools=gen_config.tools,
            generation_config=types.GenerationConfig(
                response_modalities=gen_config.response_modalities,
            ),
        )
    response = client.models.count_tokens(
        model=model,
        contents=contents,
        config=count_config,
    )
    input_tokens = getattr(response, "total_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(response, "total_token_count", None)

    resolved = [resolve_gcs_uri(p) for p in (gcs_paths or [])]
    result: dict[str, Any] = {
        "model": model,
        "mode": mode,
        "input_tokens": input_tokens,
        "gcs_paths": gcs_paths or [],
        "resolved_gcs_uris": resolved,
        "note": (
            "count_tokens only measures input. "
            "Actual total usage is in usage_metadata after generate_content."
        ),
    }
    if input_tokens is not None:
        input_cost = estimate_cost_usd(input_tokens=int(input_tokens), output_tokens=0)
        if input_cost:
            result["input_cost_usd"] = {
                "input_usd": input_cost["input_usd"],
                "currency": "USD",
                "note": "Input-only cost; output billed after the request completes.",
            }
    return result


def grounded_ask_stream(
    client: genai.Client,
    question: str,
    *,
    model: str,
    mode: GroundingMode = "search",
    lat_lng: Optional[types.LatLng] = None,
    exclude_domains: Optional[list[str]] = None,
    gcs_paths: Optional[list[str]] = None,
) -> Iterator[dict[str, Any]]:
    """Yield SSE-friendly events: text_delta, grounding, usage, done."""
    if is_image_generation_mode(mode):
        raise RuntimeError(
            "Image Search 生图模式不支持流式：请使用「发送」以获取 inline 图片与 grounding 元数据。"
        )
    effective_lat_lng = lat_lng if lat_lng is not None else _parse_lat_lng()
    config = _build_config(
        mode=mode,
        lat_lng=effective_lat_lng,
        exclude_domains=exclude_domains,
    )
    contents = build_user_contents(
        question, lat_lng=effective_lat_lng, gcs_paths=gcs_paths
    )

    accumulated_text = ""
    last_grounding: Optional[dict[str, Any]] = None
    last_usage: Optional[dict[str, Any]] = None

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        if chunk.text:
            accumulated_text += chunk.text
            yield {"event": "text_delta", "data": {"text": chunk.text}}

        if chunk.candidates:
            gm = getattr(chunk.candidates[0], "grounding_metadata", None)
            if gm is not None:
                gm_dict = grounding_metadata_to_dict(gm)
                if gm_dict != last_grounding:
                    last_grounding = gm_dict
                    yield {"event": "grounding", "data": gm_dict}

        if chunk.usage_metadata is not None:
            usage = usage_metadata_to_dict(chunk.usage_metadata)
            if usage != last_usage:
                last_usage = usage
                yield {"event": "usage", "data": usage}

    status = "ok" if last_grounding else "ungrounded"
    yield {
        "event": "done",
        "data": {
            "status": status,
            "model": model,
            "mode": mode,
            "text": accumulated_text,
            "grounding": last_grounding,
            "usage": last_usage,
        },
    }


def grounded_ask(
    client: genai.Client,
    question: str,
    *,
    model: str,
    mode: GroundingMode = "search",
    max_retries: int = 1,
    lat_lng: Optional[types.LatLng] = None,
    exclude_domains: Optional[list[str]] = None,
    gcs_paths: Optional[list[str]] = None,
) -> GroundedResult:
    effective_lat_lng = lat_lng if lat_lng is not None else _parse_lat_lng()
    config = _build_config(
        mode=mode,
        lat_lng=effective_lat_lng,
        exclude_domains=exclude_domains,
    )
    last: Optional[GroundedResult] = None

    prompts = [question]
    if max_retries > 0:
        if mode == "image_search":
            prompts.append(question + IMAGE_GEN_RETRY_SUFFIX)
        elif mode == "maps":
            prompts.append(
                question
                + "\n\n[系统补强] 请务必使用 Google Maps 工具检索后再回答；"
                "若无法检索到可靠信息，请明确说明未找到，不要猜测。"
            )
        else:
            tool_name = "Google 网页+图片搜索" if mode == "search_hybrid" else "Google 搜索"
            prompts.append(
                question
                + f"\n\n[系统补强] 请务必使用 {tool_name} 工具检索后再回答；"
                "若无法检索到可靠信息，请明确说明未找到，不要猜测。"
            )

    for attempt, prompt in enumerate(prompts[: max_retries + 1]):
        contents = build_user_contents(
            prompt, lat_lng=effective_lat_lng, gcs_paths=gcs_paths
        )
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
    if result.generated_images:
        print("-" * 72)
        print(f"GENERATED IMAGES: {len(result.generated_images)}")
        for img in result.generated_images:
            print(f"  [{img.get('index')}] {img.get('mime_type')} ({len(img.get('data_uri', ''))} chars data-uri)")
    print("-" * 72)
    print("QUERIES (web):")
    for q in result.web_search_queries or ["(none)"]:
        print(f"  - {q}")
    if result.image_search_queries:
        print("-" * 72)
        print("QUERIES (image):")
        for q in result.image_search_queries:
            print(f"  - {q}")
    print("-" * 72)
    print("SOURCES:")
    if result.sources:
        for s in result.sources:
            if s.get("kind") == "image":
                print(
                    f"  [{s.get('index')}] (image) {s.get('title')}\n"
                    f"      source_uri: {s.get('source_uri')}\n"
                    f"      image_uri:  {s.get('image_uri')}"
                )
                resolved = s.get("resolved") or {}
                for key in ("source_uri", "image_uri"):
                    r = resolved.get(key)
                    if r and r.get("final"):
                        print(f"      {key} (final): {r.get('final')}")
            else:
                print(
                    f"  [{s.get('index')}] ({s.get('kind')}) {s.get('title')}\n"
                    f"      {s.get('uri')}"
                )
                r = (s.get("resolved") or {}).get("uri")
                if r and r.get("final"):
                    print(f"      uri (final): {r.get('final')}")
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
        choices=["search", "image_search", "search_hybrid", "maps"],
        default="search",
        help="Grounding: web search, image search, both, or googleMaps",
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
    parser.add_argument(
        "--exclude-domains",
        default=_env("EXCLUDE_DOMAINS"),
        help="Comma-separated domains to exclude (Search / Vertex only)",
    )
    parser.add_argument(
        "--gcs",
        default="",
        help="Comma-separated GCS object paths or gs:// URIs (Vertex multimodal)",
    )
    parser.add_argument("--stream", action="store_true", help="SSE stream to stdout")
    parser.add_argument(
        "--count-tokens",
        action="store_true",
        help="Count input tokens before generation (no output until request completes)",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--resolve-redirects",
        action="store_true",
        help="Follow grounding-api-redirect URLs and print final destinations",
    )
    args = parser.parse_args(argv)
    resolve_redirects = args.resolve_redirects or env_resolve_redirects_default()

    if args.mode == "maps" and args.question == "美国下一次可见的日全食是什么时候？":
        args.question = "这附近有什么口碑好的咖啡馆？"
    if args.mode == "image_search" and args.question == "美国下一次可见的日全食是什么时候？":
        args.question = DEFAULT_IMAGE_GEN_QUESTION
    if args.mode == "image_search" and args.model == _env("GEMINI_MODEL", "gemini-2.5-flash"):
        args.model = default_model_for_mode("image_search")

    lat_lng = None
    if args.lat is not None or args.lng is not None:
        if args.lat is None or args.lng is None:
            raise SystemExit("Provide both --lat and --lng together.")
        lat_lng = make_lat_lng(args.lat, args.lng)

    gcs_paths = parse_gcs_paths_csv(args.gcs or "")
    client = build_client()

    if args.count_tokens:
        payload = count_tokens_for_request(
            client,
            model=args.model,
            question=args.question,
            mode=args.mode,  # type: ignore[arg-type]
            lat_lng=lat_lng,
            exclude_domains=parse_exclude_domains_csv(args.exclude_domains or ""),
            gcs_paths=gcs_paths,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.stream:
        for ev in grounded_ask_stream(
            client,
            args.question,
            model=args.model,
            mode=args.mode,  # type: ignore[arg-type]
            lat_lng=lat_lng,
            exclude_domains=parse_exclude_domains_csv(args.exclude_domains or ""),
            gcs_paths=gcs_paths,
        ):
            sys.stdout.write(format_sse(ev["event"], ev["data"]))
            sys.stdout.flush()
        return 0

    result = grounded_ask(
        client,
        args.question,
        model=args.model,
        mode=args.mode,  # type: ignore[arg-type]
        max_retries=0 if args.no_retry else 1,
        lat_lng=lat_lng,
        exclude_domains=parse_exclude_domains_csv(args.exclude_domains or ""),
        gcs_paths=gcs_paths,
    )

    if resolve_redirects and result.sources:
        result.sources = enrich_sources(result.sources, enabled=True)

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
