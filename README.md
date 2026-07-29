# Grounding Demo (Vertex AI + google-genai)

独立项目：用 **Vertex AI** 上的 Gemini，通过 **Grounding with Google Search** 做联网检索回答，并在应用层校验 `groundingMetadata`，尽量降低幻觉、提高可核对性。

与同目录下的 SmartAgents / TSE 等仓库无关，请单独安装与运行。

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

1. GCP 项目已开通 Vertex AI，账号有 `aiplatform.endpoints.predict` 等权限  
2. 本地认证其一即可：
   - `gcloud auth application-default login`
   - 或服务账号 `GOOGLE_APPLICATION_CREDENTIALS`
3. 安装依赖：

```bash
cd /Users/apple/Desktop/Demo/groundingdemo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env：填入 GOOGLE_CLOUD_PROJECT 等
```

## 运行

```bash
# 默认示例问题
python demo_grounded_search.py

# 自定义问题 + 第二轮引用核对 + JSON 摘要
python demo_grounded_search.py "2026 年 FIFA 世界杯主办地是哪里？" --verify --json

# 指定模型 / 置信度阈值
python demo_grounded_search.py "芝加哥本周末天气？" \
  --model gemini-2.5-flash \
  --min-confidence 0.65
```

退出码：`0` = 成功接地；`2` = 未接地（无 metadata）；`1` = 其它错误。

## Demo 里做了什么（降幻觉手段）

| 手段 | 实现 |
|------|------|
| Search grounding | `tools=[Tool(google_search=...)]` |
| temperature=1.0 | 官方推荐 |
| System prompt | 无可靠来源不编造、冲突并列 |
| `exclude_domains` | `.env` 中 `EXCLUDE_DOMAINS`（Vertex） |
| 钓鱼/风险站屏蔽 | `.env` 中 `BLOCKING_CONFIDENCE`（Vertex） |
| 地理偏置 | `.env` 中 `USER_LAT_LNG=lat,lng` |
| 校验 metadata | 缺失 → 自动重试一次强化提示；仍无则 `ungrounded` |
| 置信度 | `groundingSupports.confidenceScores` 低于阈值标为 WEAK |
| 可观测性 | 打印 `webSearchQueries`、来源、支撑片段 |
| 第二轮核对 | `--verify`：另一次请求（无 Search）只根据返回来源判 Supported/Unsupported |

**说明**：本 CLI 打印 Search Suggestions 的存在性；若做成 Web UI，须按官方 ToS **原样渲染** `renderedContent`，不得改样式。

## 项目结构

```
groundingdemo/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
└── demo_grounded_search.py
```

## 注意

- 同一 `generateContent` 在 Vertex/Enterprise 文档路径下，**不要**把 `googleSearch` 与 function calling / RAG retrieval 混用；本 demo 的 `--verify` 是**第二次独立请求**。
- Gemini 3 系列 Search grounding 可能按**内部 search query 次数**计费，详见官方 Billing 说明。
- Grounding 降低幻觉，不能保证零幻觉；关键事实请结合来源人工复核。
