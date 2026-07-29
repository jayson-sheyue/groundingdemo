# Grounding 学习指南（Gemini 2.5+）

本仓库只演示**当前代**能力：Google Search grounding、Google Maps grounding、可选 `latLng` 位置上下文。  
**不包含**旧版 `confidenceScores` 阈值方案（官方：Gemini 2.5+ 该字段为空，应忽略）。

> 官方文档：  
> - [Grounding 概览](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/overview)  
> - [Grounding with Google Search](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-search?hl=zh-cn)  
> - [Grounding with Google Maps](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-maps)  
> - [GroundingMetadata](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GroundingMetadata)

---

## 1. Grounding 是什么？

大模型会“编”。Grounding 让回答先挂到**可核对的外部数据**上，再生成文字并尽量带回引用。

本 demo 支持两种**不同工具**（数据源不同）：

| 模式 | API 工具 | 数据从哪来 | 典型问题 |
|------|----------|------------|----------|
| Search | `googleSearch` | 公开网页 | 赛事结果、产品规格、新闻 |
| Maps | `googleMaps` | Google Maps 地点生态 | 附近餐厅、营业时间、地方商户 |

另有第三样东西——**`latLng` 位置上下文**——见下一节。它**不是**第三种 grounding 工具。

---

## 2. Maps grounding ≠ Search 的「地理偏置」

这是最容易混的一点。

### 2.1 地理偏置 / 位置上下文（`latLng`）

```python
tool_config=types.ToolConfig(
    retrieval_config=types.RetrievalConfig(
        lat_lng=types.LatLng(latitude=31.2304, longitude=121.4737),
    )
)
```

- 写在 `toolConfig.retrievalConfig` 里  
- 含义：告诉工具「用户大概在这」  
- **Search** 模式下：网页搜索结果更偏本地（天气、本地政策等）  
- **Maps** 模式下：官方也建议传坐标，让「附近」类查询更准  
- **不传也可以调用工具**；只是少了位置偏好  

→ 这就是本项目 UI 里的「纬度 / 经度 / 使用我的位置」。

### 2.2 Grounding with Google Maps（独立工具）

```python
tools=[types.Tool(google_maps=types.GoogleMaps())]
# 通常再配上面的 lat_lng
```

- 启用的是 **`googleMaps` 工具**，不是给 Search 加个坐标就变成 Maps  
- 回答基于 **Maps 地点数据**（商户、地标等），不是一般网页 SERP  
- 适合本地生活；不适合「世界杯主办地」这类全球事实题  

### 2.3 对照表

| | Search + latLng | Maps grounding |
|--|-----------------|----------------|
| 工具字段 | `googleSearch` | `googleMaps` |
| 要不要 latLng | 可选（本地题建议传） | 强烈建议传（尤其「附近」） |
| 数据源 | 公开网页 | Google Maps |
| 是否同一功能 | **否** | **否** |

一句话：**坐标是位置上下文；Maps / Search 才是 grounding 工具。传了坐标 ≠ 启用了 Maps。**

### 2.4 常见坑：传了 latLng，模型却说「无法获取位置」

`latLng` 写在 **toolConfig** 里，主要给**检索后端**做偏置；**不会自动变成模型可见的「用户消息」**。  
模型若只看到「这附近有什么咖啡馆」，仍可能按通用安全话术说「我拿不到你的实时定位」。

本仓库做法：一旦有坐标，会在用户内容末尾附加：

```text
[终端用户位置] latitude=..., longitude=...。请将「这附近」解释为该坐标附近…
```

另外：问「附近咖啡馆」请用 **Maps 模式**；Search + 坐标往往仍去搜网页榜单，本地 POI 体验远不如 Maps。

### 2.5 二次核对是干什么的？

**不是再搜一遍**，而是：

1. 第一轮：Search/Maps grounding → 得到回答 + 来源列表  
2. 第二轮：**关掉 grounding 工具**，只把「回答全文 + 本轮来源」塞给模型，让它逐条判 `Supported / Unsupported / Partial`

意义：抓「有引用外壳、正文夸大」的句子。  
**不能保证第二轮零幻觉**——它仍是模型；来源网页错了也会一起错。详见下方 [§4 UI 手段表](#4-ui-上的降幻觉手段必须看清能做什么不能做什么) 与 [§4.1](#41-二次核对不能保证无幻觉)。

---

## 3. 最小调用示例

### 3.1 Search grounding

```python
from google import genai
from google.genai import types

client = genai.Client(vertexai=True, project="YOUR_PROJECT", location="global")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="美国下一次可见的日全食是什么时候？",
    config=types.GenerateContentConfig(
        temperature=1.0,  # Search grounding 官方建议
        tools=[types.Tool(google_search=types.GoogleSearch())],
    ),
)
print(response.text)
print(response.candidates[0].grounding_metadata)
```

### 3.2 Maps grounding + 坐标

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="这附近有什么口碑好的咖啡馆？",
    config=types.GenerateContentConfig(
        temperature=1.0,
        tools=[types.Tool(google_maps=types.GoogleMaps())],
        tool_config=types.ToolConfig(
            retrieval_config=types.RetrievalConfig(
                lat_lng=types.LatLng(latitude=31.2304, longitude=121.4737),
            )
        ),
    ),
)
```

```bash
python demo_grounded_search.py "日全食何时？" --mode search
python demo_grounded_search.py "附近咖啡馆" --mode maps --lat 31.2304 --lng 121.4737 --verify
```

---

## 4. UI 上的降幻觉手段（必须看清：能做什么 / 不能做什么）

下面与 Web UI（`python web_app.py`）**一一对应**。这些是**降低**胡说概率的工程手段，**不是**零幻觉保证。

### 总表

| # | UI 上你看到的 | 背后在做什么 | 降幻觉作用 | 明确局限 |
|---|---------------|--------------|------------|----------|
| 1 | **Search / Maps 单选** | `tools=[googleSearch]` 或 `googleMaps` | 选对数据源，避免「用网页搜附近店 / 用地图查全球新闻」 | 选错模式仍会偏题 |
| 2 | **latLng（城市芯片 / 浏览器定位）** | `toolConfig.retrievalConfig.latLng` + 提示词注入位置 | 本地题检索更贴；并避免模型嘴上说「没有位置」 | 坐标≠Maps 工具；Search+坐标仍可能出网页榜单 |
| 3 | **缺 metadata 时重试** | 无 `grounding_metadata` 则强化提示再请求一次 | 减少「开了工具却没真搜」就当事实交卷 | 重试后仍可能 ungrounded |
| 4 | **status = ok / ungrounded** | 解析是否存在 grounding 元数据 | 未接地时UI标黄，不当已核实 | 有 metadata ≠ 内容正确 |
| 5 | **回答里的引用高亮 `[n]`** | 用 `groundingSupports` 把句子标到来源 | 用户可点源核对，降低盲信 | 2.5+ 无置信分数；有引用仍可能读偏 |
| 6 | **来源列表** | `groundingChunks`（web / maps） | 可点击原始出处 | 网页/点评本身可能错或过时 |
| 7 | **引用片段表** | supports 的 segment ↔ chunk 索引 | 看清「哪句靠哪条源」 | 不是正确性分数 |
| 8 | **Google Search Suggestions** | 原样渲染 `searchEntryPoint.renderedContent` | 合规 + 用户可进 Google SERP 自查 | 仅 Search 模式；生产必须原样展示 |
| 9 | **二次核对勾选** | 第二轮无 grounding，只根据来源判 Supported/Unsupported | 抓正文相对来源的夸大/编造 | **第二轮也会幻觉**，不能当终审 |
| 10 | **运行记录** | 进程内保存 mode / geo / status / exclude | 便于对比实验 | 重启 Web 即清空 |
| 11 | **排除域名（Search）** | `GoogleSearch(exclude_domains=[...])` | 降低噪声站 / 竞品站进入 grounding | **仅 Vertex**；Gemini API Key 路径不支持；Maps 模式无此参数 |
| 12 | （代码内，UI 不单独开关）**System prompt** | 无来源不编造、冲突并列等 | 约束生成行为 | 提示词无法强制模型 100% 遵守 |
| 13 | （代码内）**temperature=1.0** | 官方对 Search grounding 的建议配置 | 按官方推荐取值 | 不是「温度越低越准」 |

### 4.1 二次核对不能保证无幻觉

请在客户材料里写清楚：

- 第二轮**缩小了**编造空间（不能搜网、只能看已给来源），但裁判仍是 LLM。  
- 合理产品策略：对 `Unsupported` / `Partial` **降级展示或拦下**，而不是只展示第二轮的「全部 Supported」。  
- 高风险字段需要规则校验或人工，而不是「两轮模型互审完毕」。

### 4.1b 排除域名 `exclude_domains`（Vertex Search）

```python
types.Tool(
    google_search=types.GoogleSearch(
        exclude_domains=["spam.example", "low-quality.com"],
    )
)
```

| 问题 | 答案 |
|------|------|
| API 支持吗？ | **Vertex / 本仓库路径：支持。** SDK 写明 *not supported in Gemini API*（指 AI Studio / API Key 的 Gemini Developer API）。 |
| 填什么？ | **域名**（如 `example.com`），不是完整文章 URL；UI 若粘贴 URL 会自动取主机名。 |
| 上限 | 文档/SDK：默认最多约 **2000** 个域名。 |
| Maps 能用吗？ | **不能**——这是 `googleSearch` 参数，与 Maps 工具无关。 |
| UI / CLI | Web「排除域名」输入框；CLI `--exclude-domains a.com,b.com`；或 `.env` 的 `EXCLUDE_DOMAINS`。 |

作用：减少垃圾站、论坛灌水站、不想引用的站点进入 grounding 来源，从而间接降低「被脏源带偏」的风险。  
**不能**替代白名单私有知识库；排除后仍可能从其它网页编造。

### 4.2 推荐操作路径（与 UI 操作一致）

```text
① 选对模式：事实/网页 → Search；附近门店 → Maps
② 本地题：点「使用我的位置」或城市芯片（写入 latLng + 提示词）
③ 需要时填写「排除域名」（仅 Search）
④ 勾选「缺 metadata 时重试」
⑤ （可选）勾选「二次核对」
⑥ 发送后检查：
     - status 是否 ok
     - 工具卡片是否为预期的 googleSearch / googleMaps
     - 坐标卡片是否「已传」；Search 时排除域名是否生效展示
     - 来源能否点开、引用高亮是否合理
     - Search 时是否出现 Suggestions 条
     - 若开了核对：Unsupported 句是否需要删改
```

### 4.3 已明确不做的（旧版）

- **不提供** `confidenceScores` 弱支撑阈值滑块（Gemini 2.5+ 官方为空，应忽略）。  
- 勿向客户承诺「开了 grounding / 二次核对 = 零幻觉」。

---

## 5. 读响应时看什么

| 字段 | 含义 |
|------|------|
| `grounding_metadata` 是否存在 | 是否真正接地（UI: status） |
| `webSearchQueries` | Search 实际搜词（UI: 检索词） |
| `groundingChunks` | 来源（UI: 来源列表） |
| `groundingSupports` | 哪段回答对应哪些来源（UI: 高亮 + 片段表） |
| `searchEntryPoint.renderedContent` | Search 建议条（UI: Suggestions，须原样） |
| `confidenceScores` | **2.5+ 忽略** |

---

## 6. 和本仓库代码 / UI 对应

| 手段 | 代码 | UI |
|------|------|-----|
| Search / Maps | `grounded_ask(..., mode=)` | 顶部单选 |
| latLng + 提示注入 | `_build_config` + `_with_location_hint` | 地理盒子 / 定位按钮 |
| 排除域名 | `GoogleSearch(exclude_domains=...)` | 「排除域名」输入框（Search） |
| 缺 metadata 重试 | `grounded_ask` 第二次强化提示 | 「缺 metadata 时重试」 |
| 引用高亮 / 来源 / 片段 | `parse_grounding` + `annotate_answer_html` | 结果区各块 |
| Search Suggestions | `search_suggestions_html` | Suggestions 盒子 |
| 二次核对 | `verify_claims` | 勾选 + 「二次核对」结果块 |
| 运行记录 | `web_app._RUN_HISTORY` | 「运行记录」面板 |
| System / temperature | `_build_config` | （无单独控件，每次请求固定启用） |

```bash
pip install -r requirements.txt
cp .env.example .env
gcloud auth application-default login
python web_app.py   # http://127.0.0.1:8080
```

---

## 7. 给客户的一句话

**Search 用网页、Maps 用地图地点、latLng 只是位置上下文；UI 用选工具、传坐标、校验是否接地、展示可点来源与可选二次核对来降低幻觉——均不能等同于事实保证。新模型请忽略 confidenceScores。**
