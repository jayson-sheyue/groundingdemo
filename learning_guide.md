# Grounding 学习指南（Gemini 2.5+ / Vertex AI）

本仓库演示 **Vertex AI** 上当前代 Grounding：`googleSearch`（网页 / Image Search 生图 / 混合）、`googleMaps`、可选 `latLng`、SSE 流式、GCS 多模态、count-tokens、redirect 解析。  
**不包含**旧版 `confidenceScores` 阈值（Gemini 2.5+ 官方为空，应忽略）。

> 快速上手：[README.md](./README.md)  
> 官方文档全表：[§10 官方文档索引](#10-官方文档索引)

---

## 1. Grounding 是什么？

大模型会“编”。Grounding 让回答先挂到**可核对的外部数据**上，再生成文字并尽量带回引用。

本 demo 支持三种 grounding **工具族**（数据源不同）：

| 模式 | API 工具 | 数据从哪来 | 典型问题 / 产出 |
|------|----------|------------|-----------------|
| Search（网页） | `googleSearch`（`webSearch`） | 公开网页 | 文字事实问答 + 引用高亮 |
| Image Search（生图） | `googleSearch` + `imageSearch` + `responseModalities: IMAGE` | Google 图片索引 → **生成图** | 描述画面 → `inline_data` 图片 |
| Search 混合 | `webSearch` + `imageSearch` | 网页 + 图片检索 | 文字回答，来源含 web/image 块 |
| Maps | `googleMaps` | Google Maps 地点 | 附近门店、营业时间 |

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

### 2.4 注意事项：传了 latLng，模型却说「无法获取位置」

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

### 3.3 Image Search grounding · 生图（与网页 Search 不同）

官方定位：**生图前**通过 Google 图片搜索检索真实视觉参考，再输出 `IMAGE`。不是「列出图片来源」的文字问答。

```python
response = client.models.generate_content(
    model="gemini-3.1-flash-image",
    contents="请使用图片搜索作参考，生成一张写实照片：巴黎埃菲尔铁塔日落全景。",
    config=types.GenerateContentConfig(
        temperature=1.0,
        response_modalities=["TEXT", "IMAGE"],
        tools=[
            types.Tool(
                google_search=types.GoogleSearch(
                    search_types=types.SearchTypes(
                        image_search=types.ImageSearch(),
                    ),
                )
            )
        ],
    ),
)
# 主产出：candidates[0].content.parts 里的 inline_data 图片
# 可选：grounding_metadata.imageSearchQueries / groundingChunks[].image
```

| | 网页 Search | Image Search 生图 |
|--|-------------|-------------------|
| 模型 | `gemini-2.5-flash` | `gemini-3.1-flash-image` |
| 你要什么 | 文字答案 | **生成一张图** |
| 主产出 | text + `groundingSupports` | `inline_data` 图片 |
| 检索 | webSearch | imageSearch（视觉参考） |
| 价值 | 事实可核对 | 地标/产品外观更贴近真实 |

```bash
python demo_grounded_search.py --mode image_search   # 默认生图示例 prompt
python demo_grounded_search.py --mode search "日全食何时？"
```

Web UI：选 **Image Search（生图接地）** → 描述要生成的画面 → 结果区展示**生成的图片** + `imageSearchQueries` + 参考来源块。

官方：[Grounding with Google Image Search](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-search?hl=zh-cn#grounding-with-google-image-search)

---

## 4. UI 上的降幻觉手段（必须看清：能做什么 / 不能做什么）

下面与 Web UI（`python web_app.py`）**一一对应**。这些是**降低**胡说概率的工程手段，**不是**零幻觉保证。

### 总表

| # | UI 上你看到的 | 背后在做什么 | 降幻觉作用 | 明确局限 |
|---|---------------|--------------|------------|----------|
| 1 | **Search / Image / 混合 / Maps 单选** | `googleSearch`（web / image / 混合）或 `googleMaps` | 选对数据源，避免「用网页搜附近店 / 用地图查全球新闻 / 用图片搜纯文字事实」 | 选错模式仍会偏题 |
| 2 | **latLng（城市芯片 / 浏览器定位）** | `toolConfig.retrievalConfig.latLng` + 提示词注入位置 | 本地题检索更贴；并避免模型嘴上说「没有位置」 | 坐标≠Maps 工具；Search+坐标仍可能出网页榜单 |
| 3 | **缺 metadata 时重试** | 无 `grounding_metadata` 则强化提示再请求一次 | 减少「开了工具却没真搜」就当事实交卷 | 重试后仍可能 ungrounded |
| 4 | **status = ok / ungrounded** | 解析是否存在 grounding 元数据 | 未接地时UI标黄，不当已核实 | 有 metadata ≠ 内容正确 |
| 5 | **回答里的引用高亮 `[n]`** | 用 `groundingSupports` 把句子标到来源 | 用户可点源核对，降低盲信 | 2.5+ 无置信分数；有引用仍可能读偏 |
| 6 | **来源列表** | `groundingChunks`（web / image / maps） | 可点击原始出处；image 模式区分 `source_uri` 与 `image_uri` | 网页/图片/点评本身可能错或过时 |
| 7 | **引用片段表** | supports 的 segment ↔ chunk 索引 | 看清「哪句靠哪条源」 | 不是正确性分数 |
| 8 | **Google Search Suggestions** | 原样渲染 `searchEntryPoint.renderedContent` | 合规 + 用户可进 Google SERP 自查 | 仅 Search 模式；生产必须原样展示 |
| 9 | **二次核对勾选** | 第二轮无 grounding，只根据来源判 Supported/Unsupported | 抓正文相对来源的夸大/编造 | **第二轮也会幻觉**，不能当终审 |
| 10 | **运行记录** | 进程内保存 mode / geo / status / exclude | 便于对比实验 | 重启 Web 即清空 |
| 11 | **排除域名（Search）** | `GoogleSearch(exclude_domains=[...])` | 降低噪声站 / 竞品站进入 grounding | **仅 Vertex**；Gemini API Key 路径不支持；Maps 模式无此参数 |
| 12 | （代码内，UI 不单独开关）**System prompt** | 无来源不编造、冲突并列等 | 约束生成行为 | 提示词无法强制模型 100% 遵守 |
| 13 | （代码内）**temperature=1.0** | 官方对 Search grounding 的建议配置 | 按官方推荐取值 | 不是「温度越低越准」 |
| 14 | **解析 grounding redirect** | `redirect_resolve.py` 跟随 HTTPS 重定向 | 展示最终 URL（维基等） | 非 API 字段；部分站点会拒绝 HEAD |

### 4.1 二次核对不能保证无幻觉

说明要点：

- 第二轮**缩小了**编造空间（不能搜网、只能看已给来源），但裁判仍是 LLM。  
- 推荐处理方式：对 `Unsupported` / `Partial` **降级展示或拦下**，而不是仅依据第二轮的「全部 Supported」。  
- 高风险字段需要规则校验或人工复核，而不能依赖「两轮模型互审」作为终审。

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

### 4.3 范围说明（本仓库未实现）

- **不提供** `confidenceScores` 弱支撑阈值滑块（Gemini 2.5+ 官方为空，应忽略）。  
- Grounding 与二次核对**不能等同于零幻觉保证**；文档与界面表述应避免此类承诺。

---

## 5. 读响应时看什么

`groundingMetadata` 与 `generateContent` **同一次响应**返回，无单独「查检索词」API。官方 REST 定义见 [GroundingMetadata](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GroundingMetadata)。

### 5.1 顶层字段一览

| 官方字段 | 含义 | 本 demo UI |
|----------|------|------------|
| `webSearchQueries` | 网页 Search 实际搜词 | 「检索词（webSearchQueries）」 |
| `imageSearchQueries` | Image Search 视觉检索词 | 「检索词（imageSearchQueries）」 |
| `retrievalQueries` | 企业 RAG / Vertex AI Search 等检索词 | 「检索词（retrievalQueries）」+ 扩展区 |
| `groundingChunks` | 来源块数组（见 §5.2） | 「来源（groundingChunks）」 |
| `groundingSupports` | 回答片段 ↔ 来源索引（见 §5.3） | 引用高亮 + 「引用片段」表 |
| `searchEntryPoint` | Search Suggestions（须合规展示） | Suggestions 盒子 + 扩展区元数据 |
| `retrievalMetadata` | 如 dynamic retrieval 分数 | 「groundingMetadata 扩展字段」 |
| `googleMapsWidgetContextToken` | Maps 小组件上下文 token | 扩展区 |
| `sourceFlaggingUris` | 来源标记 / 举报 URI | 扩展区 |
| `grounding_metadata` 是否存在 | 是否真正接地 | UI: `status`（ok / ungrounded） |

### 5.2 `groundingChunks[]` 子类型

| 子字段 | 常用属性 | 说明 |
|--------|----------|------|
| `web` | `uri`, `title`, `domain` | 网页来源；`uri` 多为 **redirect**（`vertexaisearch.cloud.google.com`），**无** `original_uri` |
| `image` | `source_uri`, `image_uri`, `title` | 参考图着陆页 + 图直链，亦多为 redirect |
| `maps` | `uri`, `title`, `placeId` | Maps POI |
| `retrievedContext` | `uri`, `title`, `text`, `documentName`, `pageNumber` | 企业语料 / RAG 片段 |

### 5.3 `groundingSupports[]`

| 子字段 | 含义 |
|--------|------|
| `segment` | `text` + `startIndex` / `endIndex`（回答中的片段） |
| `groundingChunkIndices` | 对应 `groundingChunks` 下标 |
| `confidenceScores` | **Gemini 2.5+ 通常为空**，勿做阈值过滤；旧模型可能有值 |

### 5.4 其他响应部分

| 字段 | 含义 |
|------|------|
| `searchEntryPoint.renderedContent` | Search 建议条 HTML（**须原样展示**，勿删改） |
| `inline_data`（Image 生图） | 生成图片，在 `content.parts` 中（非 groundingMetadata） |

### 5.5 集成要点

- **检索词** = 模型（或工具）实际发起的查询，在 metadata 中透明可见。  
- **来源 URL** = API 返回的 redirect，不是站点原始链接；可选 `--resolve-redirects` 解析最终 URL（见 [§9](#9-grounding-来源-url-与-redirect)）。  
- **扩展字段表 + JSON 快照** = 结果区「groundingMetadata 扩展字段」，便于对照官方 schema 完成集成。

### 5.6 groundingMetadata 字段详解

本节说明集成过程中常见的字段问题：**三种检索词的区别**、**`retrievalQueries` 为何常为空**、**`searchEntryPoint` 的作用**、**来源与引用的用法**、**redirect URL 是否为异常**。

#### 5.6.1 三种「检索词」：对应三类数据源

| 字段 | 数据源 | 本仓库何时有值 | 用途 |
|------|--------|----------------|------|
| `webSearchQueries` | 公开互联网（Google 网页搜索） | Search / 混合模式，且 grounding 成功 | **可审计**：展示模型实际使用的网页检索词 |
| `imageSearchQueries` | Google 图片索引 | Image 生图 / 混合模式 | **生图参考**：展示生成前使用的视觉检索词（地标、产品外观等） |
| `retrievalQueries` | **企业私有语料**（Vertex AI Search、RAG Datastore、Corpus 等） | **本仓库默认无**（未配置企业检索工具） | **内部溯源**：标明答案所依据的内部文档检索词 |

三者**互不替代**：

```text
用户问题
    ├─ webSearchQueries      → groundingChunks[].web
    ├─ imageSearchQueries    → groundingChunks[].image
    └─ retrievalQueries      → groundingChunks[].retrievedContext
```

**本仓库仅配置 `googleSearch` / `googleMaps`**，因此界面中 `retrievalQueries` 显示为「（无）」属**正常情况**，不代表 grounding 失败。要获得 `retrievalQueries`，须在 `tools` 中接入 Vertex AI Search / 企业 RAG（响应通常同时包含 `retrievedContext` 块，含 `documentName`、`pageNumber`、`text` 等）。

#### 5.6.2 `retrievalQueries`：适用场景与边界

**作用：** 与 `webSearchQueries` 对称，标明向**企业知识库**发起的检索 query，用于：

1. **合规留痕**：金融、医疗、政务等场景需记录信息是否来自核准内部文档  
2. **RAG 调试**：检索词异常时，优先检查索引、分块与同义词配置  
3. **与 `retrievalMetadata` 联动**：例如 `googleSearchDynamicRetrievalScore` 反映 dynamic retrieval 是否判定「需要查库」

**使用边界：**

- 存在 `retrievalQueries` **不保证**答案正确，仅表示已向语料库发起检索  
- 不存在 `retrievalQueries` **不表示**未使用 grounding——公开 Search 场景通常不填充此字段  
- 公开 Search 与企业 RAG **可并存**，三种检索词可能同时出现在同一次响应中

#### 5.6.3 `searchEntryPoint`：Search Suggestions 与合规展示

**定义：** 启用 **Google Search grounding** 且检索成功时，API 在 `groundingMetadata.searchEntryPoint` 返回 **Search Suggestions（搜索建议条）**，主要包括：

```text
searchEntryPoint.renderedContent   # 完整 HTML（常数千字符，含样式与可点击 chip）
searchEntryPoint.sdkBlob           # 部分 SDK / 客户端场景（可选）
```

**在响应中的位置：**

```text
Google Search grounding 成功
    ↓
模型文字回答 + groundingSupports 高亮
    + groundingChunks 来源列表
    + webSearchQueries 检索词
    + searchEntryPoint.renderedContent  ← 须按条款展示的 UI 条
```

**主要用途：**

| 用途 | 说明 |
|------|------|
| **合规** | 使用 Google Search 数据时，须按服务条款**原样展示**该 HTML，不可删改 chip 文案，也不可自行仿制外观相同的替代条 |
| **用户核对** | 用户可点击 chip 跳转 Google SERP，自行核实事实 |
| **与检索词互补** | `webSearchQueries` 为机器可读的 query 列表；Suggestions 条为面向终端用户的官方展示形态 |

**本仓库两处展示：**

| UI 位置 | 内容 |
|---------|------|
| 「Google Search Suggestions」区域 | **完整** `renderedContent`，原样渲染 |
| 扩展区「searchEntryPoint（元数据）」 | 仅 `present` / `length` / `preview`（前 200 字符），避免 JSON 快照包含完整 HTML |

扩展区出现 `"length": 5168` 且 preview 含 `.chip` 样式，表明 **Search grounding 已成功**，Suggestions 条已正常返回。

**何时为空：** Maps 模式、纯 Image 生图、未触发 Search、`status = ungrounded` 时常见。

**上线检查项：**

- [ ] 在答案或来源附近插入 `renderedContent`，**不**修改 HTML 结构与文案  
- [ ] 不以扩展区 preview 代替完整 Suggestions 条  
- [ ] Maps / 纯生图场景可不展示该区域（对应模式通常无此字段）

#### 5.6.4 `groundingChunks` 与 `groundingSupports`：来源与引用

| 结构 | 说明的问题 | 用途 |
|------|------------|------|
| **`groundingChunks`** | 依据来自**何处** | 可点击 URL / 文档名 / 地图 POI；支持跳转原文与对账 |
| **`groundingSupports`** | **哪句话**对应哪些 chunk | 驱动引用高亮 `[0,1]`；片段表展示 segment ↔ 索引 |

注意：

- 正文中的 Markdown 链接 **不可作为来源依据**，须以 `groundingChunks` 为准  
- `groundingSupports` 表示模型将该片段关联到来源，**不是**正确性评分  
- Gemini 2.5+ 的 `confidenceScores` 通常为空，**不宜**据此做「低于阈值则隐藏句子」等过滤（见 §5.3）

#### 5.6.5 `status = ok` 与 `ungrounded`

| status | 含义 | 推荐 UI 处理 |
|--------|------|--------------|
| `ok` | 响应包含 `grounding_metadata`（且本仓库可解析） | 可展示来源与引用；内容正确性仍需独立判断 |
| `ungrounded` | 无 metadata，或生图模式未返回标准 metadata | 降级展示；避免标注「已联网核实」；可启用「缺 metadata 时重试」 |

**作用：** 区分「已配置 grounding 工具且本轮返回 metadata」与「未接地」两种情况。  
**局限：** `ok` 仅表示 metadata 存在，**不保证**检索质量或摘要无偏。

#### 5.6.6 来源 URL 为 redirect：API 设计行为

`groundingChunks` 中常见：

```text
https://vertexaisearch.cloud.google.com/grounding-api-redirect/...
```

而非目标站直链。Vertex 与 Gemini API **均如此**，且 **无 `original_uri` 字段**（详见 [§9](#9-grounding-来源-url-与-redirect)）。

**平台侧原因：** 统一归因、合规跳转、计费与滥用防护。  
**集成侧处理：** redirect 链接可合规展示与点击；若需最终 URL，可选用 `--resolve-redirects`（`redirect_resolve.py`），此为**工程增强**，非 API 保证字段。

#### 5.6.7 扩展字段速查

| 字段 | 典型场景 | 用途 |
|------|----------|------|
| `retrievalMetadata` | Dynamic retrieval / 企业检索 | 评估是否触发检索、调整阈值 |
| `googleMapsWidgetContextToken` | Maps grounding + 地图组件 | 嵌入 Google Maps UI |
| `sourceFlaggingUris` | 来源标记 / 举报 | 合规与用户反馈 |
| `_fieldPresence`（本仓库扩展） | 集成调试 | 查看本次响应中哪些官方字段有值 |

#### 5.6.8 结构总览（网页 Search 成功时）

```text
                    ┌─────────────────────────────────────┐
                    │         groundingMetadata            │
                    ├─────────────────────────────────────┤
 webSearchQueries ──┤ 实际使用的网页检索词                  │
 imageSearchQueries ┤ （生图/混合）视觉检索词               │
 retrievalQueries ──┤ （企业 RAG）内部检索词；本仓库常为空   │
                    ├─────────────────────────────────────┤
 groundingChunks ───┤ 可核对出处（web/image/maps/RAG）      │
 groundingSupports ─┤ 句子级引用高亮                        │
 searchEntryPoint ──┤ 合规 Suggestions + 跳转 SERP          │
 retrievalMetadata ─┤ 检索策略元数据（如有）                │
                    └─────────────────────────────────────┘
                              +
                    candidates[].content.parts（模型回答正文）
```

---

## 6. 和本仓库代码 / UI 对应

| 手段 | 代码 | UI |
|------|------|-----|
| Image Search 生图 | `grounded_ask(..., mode=image_search)` + `response_modalities` | 「生图接地」+ 生成图画廊 |
| latLng + 提示注入 | `_build_config` + `_with_location_hint` | 地理盒子 / 定位按钮 |
| 排除域名 | `GoogleSearch(exclude_domains=...)` | 「排除域名」（Search / 混合） |
| 缺 metadata 重试 | `grounded_ask` 第二次强化提示 | 「缺 metadata 时重试」 |
| 引用高亮 / 来源 / 片段 | `parse_grounding` + `annotate_answer_html` | 结果区各块 + **groundingMetadata 扩展字段** |
| Search Suggestions | `search_suggestions_html` | Suggestions 盒子 |
| 二次核对 | `verify_claims` | 勾选 + 核对结果 |
| 运行记录 | `web_app._RUN_HISTORY` | 「运行记录」面板 |
| SSE 流式 | `grounded_ask_stream` | 「流式发送」/ `POST /ask/stream` |
| GCS 多模态 | `build_user_contents` / `gcs_upload.py` | 「添加附件」 |
| count-tokens | `count_tokens_for_request` | 「统计输入 Token」 |
| redirect 解析 | `redirect_resolve.enrich_sources` | 勾选「解析 grounding redirect」 |
| System / temperature | `_build_config` | （代码内固定） |

```bash
pip install -r requirements.txt
cp .env.example .env
gcloud auth application-default login
python web_app.py   # http://127.0.0.1:8080
```

---

## 7. 总结

**网页 Search 用于事实问答，Image Search 在生图前检索视觉参考，Maps 用于地点查询；`latLng` 仅为位置上下文。** 来源 URL 在 API 中多为 Google redirect，而非原始链接。通过选择合适工具、校验 grounding 状态、展示来源与引用，并原样展示 Search Suggestions，可降低幻觉风险——**上述手段均不能等同于事实准确性保证。**

---

## 8. SSE 流式、多模态 GCS、count-tokens

本 demo 已支持三项企业常用能力。官方参考：

| 能力 | 本仓库入口 | 官方文档 |
|------|------------|----------|
| **SSE 流式** | CLI `--stream`；`POST /ask/stream`；`grounded_ask_stream()` | [流式生成](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-chat-prompts-gemini#stream-responses) · [generate_content_stream](https://googleapis.github.io/python-genai/genai.html#genai.models.Models.generate_content_stream) · [streamGenerateContent REST](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#streamGenerateContent) |
| **多模态 GCS** | CLI `--gcs`；Web「GCS 对象」；`.env` `GCS_BUCKET` / `GCS_PREFIX` | [多模态 GCS URI](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-multimodal-prompts#gcs) · [PDF 理解](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/document-understanding) · [图片理解](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding) |
| **count-tokens** | CLI `--count-tokens`；`POST /count-tokens`；`count_tokens_for_request()` | [Count Tokens](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/get-token-count) · [SDK count_tokens](https://googleapis.github.io/python-genai/genai.html#genai.models.Models.count_tokens) · [定价](https://cloud.google.com/vertex-ai/generative-ai/pricing) |

### 8.1 SSE 事件格式

`POST /ask/stream` 返回 `text/event-stream`，事件类型：

| event | 含义 |
|-------|------|
| `text_delta` | 回答文本增量 `{"text":"..."}` |
| `grounding` | `grounding_chunks` / `webSearchQueries` / `imageSearchQueries` / `groundingSupports` / Suggestions 快照 |
| `usage` | `usage_metadata`（`prompt_token_count`、`candidates_token_count`、`total_token_count` 等） |
| `done` | 流结束摘要 `status` / 全文 / 最终 grounding & usage |
| `elapsed` | 服务端耗时 ms |
| `error` | 异常信息 |

流式模式**不做**「缺 metadata 重试」（避免双次流式请求）；普通 `POST /ask` 仍支持重试。

**Web UI 展示差异：** 流式过程中为**纯文本 + 光标**（逐字打出）；收到 `done` 且存在 `groundingSupports` 后，前端才套用与「发送」相同的引用高亮，并展示检索词 / 来源。流式开始时也会隐藏上一次「发送」的结果区，避免混淆。

### 8.2 多模态 + Grounding（同一轮请求）

Web UI：**在问题框下「添加附件」→ 输入文字 → 发送**。附件在发送时自动上传 GCS，与文字、Search/Maps grounding **同一轮** `generate_content`：

```text
contents = [Part(PDF/图片 gs://...), ..., "你的问题..."]
config   = { tools: [googleSearch 或 googleMaps], ... }
```

**可以多模态 + grounding 同时使用。** 例如：「根据这份 datasheet，用网页搜索核对最新价格」——模型既读 PDF，又触发 Search grounding，响应里仍可有 `grounding_metadata`（来源、引用高亮）。

| 能力 | 说明 |
|------|------|
| 附件 + Search | 读文档 + 搜网核实 / 补充 |
| 附件 + Maps | 读图片/文档 + 查地图地点（较少见，但 API 允许） |
| 引用高亮 | grounding 针对**回答中来自搜索/地图的句子**；附件内容本身不生成 web chunk |

官方：[多模态 GCS](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-multimodal-prompts#gcs) · [Grounding overview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/overview) · [Function calling / tools 与多模态](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling)

上传实现：`POST /upload` → `build_user_contents()` 把 `gs://` URI 与问题文本拼成 `contents`。

### 8.3 count-tokens 与实际用量

| 时机 | API | 返回什么 |
|------|-----|----------|
| **发送前** | `count_tokens` | 仅**输入** token（问题 + 附件 + tools 配置） |
| **发送后** | `usage_metadata` | **实际** prompt / candidates / total（流式最后一帧也有） |

```bash
python demo_grounded_search.py "问题" --count-tokens   # 只看输入
python demo_grounded_search.py "问题" --stream         # 流式最后一帧带 usage
```

**不需要手填「预估输出 token」。** 输出 token 只有模型生成结束后才知道；费用估算应基于 `usage_metadata`，在 Web UI 结果区或流式 `[usage 实际]` 日志中查看。

可选：`.env` 设置 `GEMINI_PRICE_INPUT_PER_MTOK` / `GEMINI_PRICE_OUTPUT_PER_MTOK`，发送后自动换算实际美元费用。

### 8.4 代码对应

| 能力 | 函数 | Web 路由 |
|------|------|----------|
| 流式 | `grounded_ask_stream` / `format_sse` | `POST /ask/stream` |
| GCS 多模态 | `build_user_contents` / `resolve_gcs_uri` | 表单 `gcs_paths` |
| count-tokens | `count_tokens_for_request` / `estimate_cost_usd` | `POST /count-tokens` |

---

## 9. Grounding 来源 URL 与 redirect

### 9.1 现象

`groundingChunks` 中的 URL 常为：

```text
https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQ...
```

而非 `https://en.wikipedia.org/...` 等原始地址。适用于：

- 网页 Search：`chunk.web.uri`
- Image Search：`chunk.image.source_uri`、`chunk.image.image_uri`

官方文档写明 Image 来源 **“are returned as redirect URLs”**（[§ Image Search Response](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-search?hl=zh-cn#grounding-with-google-image-search)）。

### 9.2 Vertex vs Gemini API

| 问题 | 答案 |
|------|------|
| 只有 Vertex 才 redirect 吗？ | **否。** Gemini Developer API（API Key）同样返回 `vertexaisearch.cloud.google.com` redirect。 |
| API 有 `original_uri` 字段吗？ | **目前没有。** 社区有 [功能请求](https://discuss.ai.google.dev/t/feature-request-provide-actual-source-urls-in-grounding-metadata/107352)。 |
| `chunk.web.domain` 能用吗？ | 有时有域名提示，**不是**完整 URL。 |
| 与其他产品对比？ | 部分搜索/问答产品直接暴露 citation URL；Google Search grounding 经统一 redirect 网关（归因、合规、计费）。 |

### 9.3 本 demo 的做法

| 方式 | 说明 |
|------|------|
| **展示 redirect** | 默认：链接即 API 返回值，合规可点击 |
| **解析最终 URL** | 勾选「解析 grounding redirect」或 CLI `--resolve-redirects`；`redirect_resolve.py` 用 `certifi` 做 HTTPS，跟随 `Location` |
| **不要信正文里的 URL** | 模型可能在文字里写 Markdown 假链接，必须以 `groundingChunks` 为准 |

环境变量：`.env` 中 `RESOLVE_GROUNDING_REDIRECTS=true` 可默认开启解析。

---

## 10. 官方文档索引

### 10.1 Grounding 产品与 API

| # | 文档 | 说明 |
|---|------|------|
| 1 | [Grounding overview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/overview) | 总览：Search / Maps / 元数据 |
| 2 | [Grounding with Google Search](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-search?hl=zh-cn) | 网页 Search、`exclude_domains`、Suggestions |
| 3 | [§ Image Search](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-search?hl=zh-cn#grounding-with-google-image-search) | `imageSearch`、`source_uri` / `image_uri`、redirect |
| 4 | [§ Response（Search）](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-search?hl=zh-cn#response) | `groundingMetadata` 字段说明 |
| 5 | [Grounding with Google Maps](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-maps) | `googleMaps`、地点数据 |
| 6 | [GroundingMetadata（REST）](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GroundingMetadata) | `webSearchQueries`、`groundingChunks`、`groundingSupports` |
| 7 | [GroundingChunk（REST）](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GroundingChunk) | `web` / `maps` / `image` 块结构 |

### 10.2 Vertex AI 生成与多模态

| # | 文档 | 说明 |
|---|------|------|
| 8 | [Generate content](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#generateContent) | `generateContent` REST |
| 9 | [Stream generate content](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#streamGenerateContent) | 流式 REST |
| 10 | [Send chat prompts (stream)](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-chat-prompts-gemini#stream-responses) | 流式概念与示例 |
| 11 | [Multimodal prompts (GCS)](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-multimodal-prompts#gcs) | `gs://` 输入 |
| 12 | [Document understanding (PDF)](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/document-understanding) | PDF 附件 |
| 13 | [Image understanding](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding) | 图片输入 |
| 14 | [Count tokens](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/get-token-count) | 输入 token 统计 |
| 15 | [Function calling & tools](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) | `tools` 与多模态同请求 |
| 16 | [Vertex AI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) | Token 定价 |

### 10.3 Python SDK

| # | 文档 | 说明 |
|---|------|------|
| 17 | [google-genai 文档](https://googleapis.github.io/python-genai/) | Client、`GenerateContentConfig` |
| 18 | [Models.generate_content](https://googleapis.github.io/python-genai/genai.html#genai.models.Models.generate_content) | 同步生成 |
| 19 | [Models.generate_content_stream](https://googleapis.github.io/python-genai/genai.html#genai.models.Models.generate_content_stream) | 流式 |
| 20 | [Models.count_tokens](https://googleapis.github.io/python-genai/genai.html#genai.models.Models.count_tokens) | count-tokens |
| 21 | [GoogleSearch / ImageSearch types](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleSearch) | `search_types` 配置 |

### 10.4 模型与限制

| # | 文档 | 说明 |
|---|------|------|
| 22 | [Gemini models](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini) | 2.5 / 3.x 模型列表 |
| 23 | [Grounding support by model](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/overview) | 哪些模型支持 grounding |

### 10.5 社区参考（非官方承诺）

| # | 链接 | 说明 |
|---|------|------|
| 24 | [Feature: actual source URLs](https://discuss.ai.google.dev/t/feature-request-provide-actual-source-urls-in-grounding-metadata/107352) | Gemini API 同样只有 redirect |
| 25 | [python-genai #1512](https://github.com/googleapis/python-genai/issues/1512) | 维护者确认无 original URI 字段 |

---

## 附录：章节索引

| 章节 | 内容 |
|------|------|
| [§1](#1-grounding-是什么) | 概念与四种模式 |
| [§2](#2-maps-grounding--search-的地理偏置) | latLng vs Maps |
| [§3](#3-最小调用示例) | 代码示例 |
| [§4](#4-ui-上的降幻觉手段必须看清能做什么不能做什么) | UI 降幻觉对照 |
| [§5](#5-读响应时看什么) | 响应字段 |
| [§5.6](#56-groundingmetadata-字段详解) | 检索词 / Suggestions / 来源 / redirect 字段说明 |
| [§6](#6-和本仓库代码--ui-对应) | 代码 / UI 映射 |
| [§7](#7-总结) | 核心结论 |
| [§8](#8-sse-流式多模态-gcs-count-tokens) | 流式、GCS、Token |
| [§9](#9-grounding-来源-url-与-redirect) | redirect URL 说明 |
| [§10](#10-官方文档索引) | 官方文档全表 |

