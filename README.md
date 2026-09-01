# Grounding Demo（Gemini 2.5+ / Vertex AI）

Vertex AI + [`google-genai`](https://googleapis.github.io/python-genai/) 演示：**Google Search**（网页 / Image Search 生图 / 混合）、**Google Maps** grounding，以及 **latLng 位置上下文**、**SSE 流式**、**GCS 多模态**、**count-tokens**、**grounding redirect 解析**。

- **不含**旧版 `confidenceScores` 阈值（Gemini 2.5+ 官方为空，应忽略）
- **教程**：[learning_guide.md](./learning_guide.md)（概念、UI 对照、官方文档索引）
- **能力详解**：[§5 groundingMetadata 字段](./learning_guide.md#5-读响应时看什么) · [§5.6 字段详解](./learning_guide.md#56-groundingmetadata-字段详解) · [§8 流式 / 多模态 / Token](./learning_guide.md#8-sse-流式多模态-gcs-count-tokens) · [§9 来源 URL 与 redirect](./learning_guide.md#9-grounding-来源-url-与-redirect)

---

## 功能一览

| 能力 | CLI | Web UI / HTTP |
|------|-----|----------------|
| 网页 Search grounding | `--mode search` | Search（网页问答） |
| Image Search **生图**接地 | `--mode image_search` | Image Search（生图接地）→ `inline_data` 图片 |
| 网页 + 图片检索（文字回答） | `--mode search_hybrid` | 混合模式 |
| Maps grounding | `--mode maps` | Maps |
| latLng 位置上下文 | `--lat` / `--lng` | 坐标 / 浏览器定位 |
| 排除域名（Vertex） | `--exclude-domains` | 排除域名输入框 |
| SSE 流式 | `--stream` | 流式发送 → `POST /ask/stream` |
| GCS 多模态 | `--gcs path` | 添加附件 → `POST /upload` |
| count-tokens | `--count-tokens` | 统计输入 Token → `POST /count-tokens` |
| 解析 redirect 最终 URL | `--resolve-redirects` | 勾选「解析 grounding redirect」 |
| groundingMetadata 全字段 | `parse_grounding_metadata()` | 结果区「扩展字段」表 + JSON 快照 |

**响应字段**：`webSearchQueries` / `imageSearchQueries` / `retrievalQueries`、`groundingChunks`（web / image / maps / retrievedContext）、`groundingSupports`、`searchEntryPoint`、`retrievalMetadata` 等 — 字段表见 [learning_guide §5](./learning_guide.md#5-读响应时看什么)，详细说明见 [§5.6](./learning_guide.md#56-groundingmetadata-字段详解)。

---

## 模式对比（必读）

| 模式 | 模型建议 | 主产出 | 典型场景 |
|------|----------|--------|----------|
| **search** | `gemini-2.5-flash` | 文字 + `groundingSupports` 高亮 | 事实、新闻、规格 |
| **image_search** | `gemini-3.1-flash-image` | **生成图片** + 可选 `imageSearchQueries` | 生图前搜真实视觉参考 |
| **search_hybrid** | `gemini-2.5-flash` | 文字 + web/image 来源块 | 既要网页又要图片检索 |
| **maps** | `gemini-2.5-flash` | 文字 + Maps 来源 | 附近 POI、营业时间 |

**Image Search ≠ 网页 Search。** 前者是「搜参考图 → 生图」；后者是「搜网页 → 文字回答」。详见 [learning_guide §3.3](./learning_guide.md#33-image-search-grounding--生图与网页-search-不同)。

**Maps ≠ Search + latLng。** 坐标只是位置上下文；Maps 须启用 `googleMaps` 工具。详见 [learning_guide §2](./learning_guide.md#2-maps-grounding--search-的地理偏置)。

---

## 准备

```bash
git clone https://github.com/jayson-sheyue/groundingdemo.git
cd groundingdemo
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # 填写 GOOGLE_CLOUD_PROJECT、可选 GCS_BUCKET / IMAGE_GEN_MODEL
gcloud auth application-default login
```

`.env` 关键项见 [.env.example](./.env.example)。

---

## 运行

```bash
# 网页 Search
python demo_grounded_search.py "美国下一次可见的日全食是什么时候？" --mode search

# Image Search 生图（默认示例 prompt + gemini-3.1-flash-image）
python demo_grounded_search.py --mode image_search

# 混合检索（文字问答）
python demo_grounded_search.py "2026 世界杯主办城市" --mode search_hybrid

# Maps + 坐标 + 二次核对
python demo_grounded_search.py "附近咖啡馆" --mode maps --lat 31.2304 --lng 121.4737 --verify

# SSE 流式（stdout 输出 event-stream）
python demo_grounded_search.py "日全食何时？" --stream

# GCS 多模态（需 GCS_BUCKET 或完整 gs:// URI）
python demo_grounded_search.py "总结这份 datasheet" --gcs datasheets/chip.pdf

# 发送前统计输入 token
python demo_grounded_search.py "日全食何时？" --count-tokens

# 解析 grounding redirect → 最终 URL（依赖 certifi）
python demo_grounded_search.py "日全食何时？" --mode search --resolve-redirects

# Web UI
python web_app.py
# → http://127.0.0.1:8080
```

### HTTP 路由（Web）

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 主界面 |
| `POST` | `/ask` | 同步问答 + grounding 元数据 |
| `POST` | `/ask/stream` | SSE（`text_delta` / `grounding` / `usage` / `done`） |
| `POST` | `/count-tokens` | 仅输入 token |
| `POST` | `/upload` | 浏览器附件 → GCS |
| `POST` | `/history/clear` | 清空运行记录 |

---

## 项目结构

```
groundingdemo/
├── README.md
├── learning_guide.md          # 教程 + 官方文档索引
├── requirements.txt
├── .env.example
├── demo_grounded_search.py    # 核心：grounded_ask / stream / count-tokens
├── redirect_resolve.py        # 可选：grounding redirect → 最终 URL
├── gcs_upload.py              # 浏览器上传 → GCS
├── web_app.py                 # Flask UI
└── templates/index.html
```

---

## 官方文档索引

完整说明与 API 字段对照见 [learning_guide.md §10](./learning_guide.md#10-官方文档索引)。

### Grounding

| 主题 | 链接 |
|------|------|
| 概览 | [Grounding overview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/overview) |
| Google Search（含 Image Search） | [Grounding with Google Search](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-search?hl=zh-cn) |
| Image Search 专节 | [§ Grounding with Google Image Search](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-search?hl=zh-cn#grounding-with-google-image-search) |
| Google Maps | [Grounding with Google Maps](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-maps) |
| GroundingMetadata（REST） | [GroundingMetadata](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GroundingMetadata) |

### Vertex / SDK / 多模态

| 主题 | 链接 |
|------|------|
| Python SDK | [google-genai](https://googleapis.github.io/python-genai/) |
| 流式生成 | [Stream responses](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-chat-prompts-gemini#stream-responses) |
| GCS 多模态 | [Multimodal prompts (GCS)](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-multimodal-prompts#gcs) |
| Count Tokens | [Get token count](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/get-token-count) |
| Vertex 定价 | [Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) |

### 已知限制（来源 URL）

Google Search / Image Search grounding 的 `uri` / `source_uri` / `image_uri` **通常为 redirect**（`vertexaisearch.cloud.google.com`），**Vertex 与 Gemini API 均如此**。本仓库提供可选解析；详见 [learning_guide §9](./learning_guide.md#9-grounding-来源-url-与-redirect)。

---

## 降幻觉（现实路径）

与 Web UI **一一对应**的说明见 **[learning_guide §4](./learning_guide.md#4-ui-上的降幻觉手段必须看清能做什么不能做什么)**。

摘要：选对模式与模型 → latLng + 提示注入 → Search 排除域名 → 校验 `grounding_metadata` 并重试 → 展示来源与引用 → **Search Suggestions 原样展示（`searchEntryPoint`）** → `retrievalQueries` 仅在企业 RAG 场景出现 → 可选二次核对（**非零幻觉保证**）。字段说明：[§5.6](./learning_guide.md#56-groundingmetadata-字段详解)。
