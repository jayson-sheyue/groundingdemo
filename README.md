# Grounding Demo (Vertex AI + google-genai)

独立项目：用 **Vertex AI** 上的 Gemini，通过 **Grounding with Google Search** 做联网检索回答，并在应用层校验 `groundingMetadata`，尽量降低幻觉、提高可核对性。

- 命令行：`demo_grounded_search.py`
- 网页预览：`web_app.py`（回答高亮 + 来源 + **原样渲染** Google Search Suggestions）
- 教程：[learning_guide.md](./learning_guide.md)

## 官方文档

- [Grounding overview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/overview)
- [Grounding with Google Search](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-search)（中文：[zh-cn](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-search?hl=zh-cn)）
- [Grounding API](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/grounding)
- [Gemini API: Google Search grounding](https://ai.google.dev/gemini-api/docs/google-search)

要点（官方）：

- 理想效果请使用 **temperature `1.0`**
- 开启 `googleSearch` 工具；可用 `exclude_domains`、`latLng`
- 模型**不一定每次都会搜索**；无 `groundingMetadata` 时不算 grounded result
- 生产环境须展示 Search Suggestions（`searchEntryPoint.renderedContent`）与来源 / grounding supports

## 准备

1. GCP 项目已开通 Vertex AI，账号有调用权限  
2. 本地认证：`gcloud auth application-default login`（或服务账号）  
3. 安装依赖：

```bash
cd groundingdemo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env：填入 GOOGLE_CLOUD_PROJECT 等
```

## 运行

### 命令行

```bash
python demo_grounded_search.py
python demo_grounded_search.py "2026 年 FIFA 世界杯主办地是哪里？" --verify --json
```

退出码：`0` = 成功接地；`2` = 未接地；`1` = 其它错误。

### 网页 UI（推荐用来看搜索建议条样式）

```bash
python web_app.py
# 浏览器打开 http://127.0.0.1:8080
```

页面会展示：

1. 带引用来源的回答（2.5+ 通常无数值置信度，以引用为准）  
2. **Google 返回的 Search Suggestions HTML（原样嵌入，不改 CSS）**  
3. 实际搜索词、来源列表、`groundingSupports` 明细  
4. 可选「二次核对」

## Demo 里做了什么（降幻觉手段）

| 手段 | 实现 |
|------|------|
| Search grounding | `tools=[Tool(google_search=...)]` |
| temperature=1.0 | 官方推荐 |
| System prompt | 无可靠来源不编造、冲突并列 |
| `exclude_domains` | `.env` → `EXCLUDE_DOMAINS`（Vertex） |
| 钓鱼/风险站屏蔽 | `.env` → `BLOCKING_CONFIDENCE`（Vertex） |
| **弱置信阈值** | **Gemini 2.5+ 官方不返回分数，对新模型基本无产品意义。** 勿当主卖点。详见 [learning_guide §4.2](./learning_guide.md#42-弱置信详解新模型上基本可忽略) |
| **地理偏置** | 请求传 `latLng` 才生效，不会自动定位。Web：城市预设 / 浏览器定位。详见 [learning_guide §4.1](./learning_guide.md#41-地理偏置详解可选的-latlng) |
| 校验 metadata | 缺失 → 自动重试；仍无则 `ungrounded` |
| 第二轮核对 | CLI `--verify` / Web 勾选 |
| 合规展示预览 | Web UI 渲染 `renderedContent` |

更完整的教学与代码拆解见 **[learning_guide.md](./learning_guide.md)**。

## 项目结构

```
groundingdemo/
├── README.md
├── learning_guide.md
├── requirements.txt
├── .env.example
├── .gitignore
├── demo_grounded_search.py
├── web_app.py
└── templates/
    └── index.html
```

## 注意

- 同一 `generateContent` 在 Vertex/Enterprise 文档路径下，**不要**把 `googleSearch` 与 function calling / RAG retrieval 混用；二次核对是**另一次请求**。
- Gemini 3 的 Search grounding 可能按内部 search query 次数计费。
- Grounding 降低幻觉，不能保证零幻觉；关键事实请结合来源人工复核。
