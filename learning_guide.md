# Grounding with Google Search 学习指南

用通俗语言讲清楚：什么是 grounding、怎么调用 API、如何尽量提高回答准确性。  
配套代码见本仓库的 `demo_grounded_search.py`（命令行）与 `web_app.py`（网页预览）。

> 官方权威文档（请以官方为准）：  
> - [Grounding 概览](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/overview)  
> - [使用 Google 搜索建立依据](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-search?hl=zh-cn)  
> - [Gemini API：Google Search grounding](https://ai.google.dev/gemini-api/docs/google-search)

---

## 1. 先建立直觉：模型为什么会“胡说”？

大模型本质上是在**预测下一个词**，不是在查数据库。训练数据有截止日期，对“昨天的比赛结果”“下周天气”这类问题，它可能：

- 用过期知识瞎猜  
- 把多个相似事实拼错  
- 用很自信的语气说出错误答案（这就是常说的**幻觉**）

**Grounding（接地 / 建立依据）** 的意思是：回答时先去拿**可核对的外部信息**（这里是 Google 公开网页搜索结果），再据此写答案，并尽量附上出处。

官方说法是：把输出“拴”在数据源上，**降低**编造概率——注意是降低，不是消灭。

```text
没有 grounding：  用户问题 → 模型凭记忆直接答（容易过时 / 编造）

有 Search grounding：
  用户问题
    → 模型判断要不要搜
    → 生成搜索词，调用 Google 搜索
    → 阅读结果片段
    → 综合写成回答 + groundingMetadata（查询词、来源、句子级引用）
```

---

## 2. 开启 Search grounding：最小可运行例子

本仓库使用 **Vertex AI + `google-genai` SDK**。

### 2.1 环境

```bash
pip install google-genai python-dotenv
gcloud auth application-default login   # 或配置服务账号
export GOOGLE_CLOUD_PROJECT=你的项目ID
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=true
```

### 2.2 最核心的一行：挂上 `googleSearch` 工具

```python
from google import genai
from google.genai import types

client = genai.Client(
    vertexai=True,
    project="YOUR_PROJECT",
    location="global",
    http_options=types.HttpOptions(api_version="v1"),
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="美国下一次可见的日全食是什么时候？",
    config=types.GenerateContentConfig(
        # 官方：Search grounding 理想效果请用 temperature=1.0
        temperature=1.0,
        tools=[
            types.Tool(google_search=types.GoogleSearch())
        ],
    ),
)

print(response.text)
# 若成功接地，还有：
# response.candidates[0].grounding_metadata
```

对应 REST 概念（简化）：

```json
{
  "contents": [{ "role": "user", "parts": [{ "text": "你的问题" }] }],
  "generationConfig": { "temperature": 1.0 },
  "tools": [{ "googleSearch": {} }]
}
```

**记住**：只打开工具 ≠ 模型一定去搜。模型会自己判断“搜了会不会更好”。所以下一节的**元数据校验**非常重要。

---

## 3. 读懂返回：`groundingMetadata` 才是“有没有真搜到”的证据

成功接地时，候选结果里会有 `grounding_metadata`，常见字段：

| 字段 | 你可以把它理解成 | 用来干什么 |
|------|------------------|------------|
| `webSearchQueries` | 模型实际发给 Google 的搜索词 | 调试：搜偏了就改问题 / 改写查询 |
| `groundingChunks` | 用到的网页来源（标题、域名、URI） | 展示引用列表；做域名白/黑名单 |
| `groundingSupports` | **哪一段回答文字**对应**哪些来源**（2.5+ 通常无 confidenceScores） | 文内引用；有分才可做强弱着色 |
| `searchEntryPoint.renderedContent` | Google 提供的「搜索建议」条 HTML/CSS | **生产环境必须原样展示**（服务条款） |

官方也写明：来源相关性低、信息不完整等原因，可能导致**不返回** grounding 元数据——此时回答**不算** grounded result。

### 3.1 应用层硬规则（强烈建议）

```python
gm = response.candidates[0].grounding_metadata
if gm is None:
    # 不要把这段当“已核实事实”直接给用户
    raise RuntimeError("未接地：缺少 grounding_metadata，请重试或明确告知用户")
```

本仓库的 `grounded_ask()` 会在缺失时**自动再问一次**（补一句“请务必使用搜索…”），仍没有则标记 `status=ungrounded`。

---

## 4. 提高准确性：从“能跑”到“尽量靠谱”

下面按**投入产出比**从高到低讲。多数都是「官方能力 + 你自己写的几行校验」。

### 手段 A：temperature = 1.0（官方明确建议）

很多人以为“温度越低越准确”。对 **Search grounding**，官方写的是：为获得理想结果，使用 temperature **`1.0`**。  
不要为了“稳”盲目改成 `0`——那是另一套调参直觉，不适用于本场景的官方推荐。

### 手段 B：写好 System Instruction（逼模型“无据不编”）

API 本身不会 magically 禁止编造；你要用提示词约定行为：

```python
SYSTEM = """你是注重事实的助手。
1. 事实性问题优先且仅依据本次搜索结果。
2. 数字/日期/人名找不到来源就说「未找到可靠来源」，禁止用训练记忆补全。
3. 来源冲突就并列说明，不要装作只有一种答案。
4. 区分事实与推断。
"""

config = types.GenerateContentConfig(
    system_instruction=SYSTEM,
    temperature=1.0,
    tools=[types.Tool(google_search=types.GoogleSearch())],
)
```

### 手段 C：排除噪声域名 / 屏蔽高风险站点（Vertex）

```python
types.GoogleSearch(
    exclude_domains=["low-quality.example", "spam.example"],
    # Vertex：按钓鱼置信度屏蔽可疑站点（枚举名见 SDK）
    blocking_confidence=types.PhishBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
)
```

环境变量写法见 `.env.example` 的 `EXCLUDE_DOMAINS`、`BLOCKING_CONFIDENCE`。

### 手段 D：地理偏置（天气、本地政策、门店）

把终端用户的经纬度放进 `toolConfig.retrievalConfig.latLng`，让搜索更偏本地。
**不会自动定位**——坐标必须由你的应用在运行时提供。详见下方 **[§4.1 地理偏置详解](#41-地理偏置详解可选的-latlng)**。

```python
tool_config=types.ToolConfig(
    retrieval_config=types.RetrievalConfig(
        lat_lng=types.LatLng(latitude=31.2304, longitude=121.4737),
    )
)
```

CLI：`python demo_grounded_search.py "明天天气" --lat 31.2304 --lng 121.4737`
Web：城市预设 / 手动填坐标 /「使用我的位置」。

### 手段 E：看 `webSearchQueries`，把问题改写成“好搜的问法”

模型搜的是它自己改写的 query。你问“那个球星最近怎么样”，搜出来可能很散。更好的做法：

```text
差：那个比赛谁赢了？
好：2024 年欧洲杯决赛谁赢了？比分多少？
```

应用层可先做**查询改写**（补时间、实体全称、地区），再调用 grounding。

### 手段 F：`confidenceScores`（仅旧代模型有意义）

官方 [GroundingSupport](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GroundingMetadata) 写明：
**Gemini 2.5 及之后 `confidenceScores` 为空，应忽略。**
客户若只用新模型，**不要把“弱置信阈值”当主降幻觉手段**；应依赖引用来源、Search Suggestions、二次核对等。
详见 **[§4.2](#42-弱置信详解新模型上基本可忽略)**。

### 手段 G：缺失 metadata 时重试 / 拒答

```python
# 伪代码
result = ask(question)
if result 没有 grounding_metadata:
    result = ask(question + "请务必使用 Google 搜索后再答；没有来源就说找不到")
if 还是没有:
    返回「未能通过网页检索核实，请稍后再试」  # 不要假装已核实
```

### 手段 H：第二轮“只核对、不搜索”（高风险场景）

Vertex / Enterprise 文档路径下：同一请求里 **不要**把 `googleSearch` 和 function calling / RAG retrieval 混用。
所以核对应是**另一次** `generateContent`（本仓库 `--verify` / 网页勾选「二次核对」）：

1. 第一轮：Search grounding → 得到答案 + 来源列表
2. 第二轮：关闭 Search，只把「断言 + 来源」交给模型，输出 Supported / Unsupported

适合价格、法规、医疗等“说错代价大”的字段。

### 手段 I：UI 上展示来源与 Search Suggestions（合规 + 降信任风险）

官方要求（生产环境）：

1. **原样渲染** `searchEntryPoint.renderedContent`（颜色、字体都不要改）
2. 点击后直接进入 Google 搜索结果页
3. 展示 grounding supports / 来源（文内 + 汇总）

用户能点开源链接核对，比你在后端再调一次模型往往更能兜住错误。
本仓库 `web_app.py` 就是为了让你**直观看到**这套样式（含弱置信着色与地理偏置状态）。

---

## 4.1 地理偏置详解（可选的 latLng）

### 它解决什么问题？

同一句「明天会下雨吗？」：

- 不传坐标 → 模型可能搜到泛化结果，或偏向训练里更常见的地区
- 传入上海坐标 → 搜索更倾向上海本地天气信息

适合：天气、门店营业、本地政策、附近活动等**位置敏感**问题。
对「2026 世界杯主办地」这类全球事实，偏置通常影响不大。

### 是不是“启用参数就自动实现”？

| 步骤 | 自动？ | 说明 |
|------|--------|------|
| 按给定 lat/lng 偏置搜索 | ✅ 你传了才会 | 官方 `retrievalConfig.latLng` |
| 知道提问者在哪 | ❌ | Gemini / Search **不会**自动读浏览器或 IP（本 API 路径下要你自己传） |
| 写死在 `.env` 给所有用户 | ⚠️ 仅适合本地 demo | 生产应按**每次请求 / 每个用户**提供坐标 |

所以：**开发者确实无法提前预测每个提问者的位置**；正确做法是运行时采集：

1. **浏览器 Geolocation**（用户授权）→ 本 Web UI 的「使用我的位置」
2. **用户资料里保存的城市** → 服务端查出后填入请求
3. **问题里已写地名** → 模型搜索词常已带城市；`latLng` 是额外加强，不是唯一手段
4. **留空** → 完全合法，表示不做地理偏置

### 请求长什么样？

```python
from google.genai import types

config = types.GenerateContentConfig(
    temperature=1.0,
    tools=[types.Tool(google_search=types.GoogleSearch())],
    tool_config=types.ToolConfig(
        retrieval_config=types.RetrievalConfig(
            lat_lng=types.LatLng(latitude=31.2304, longitude=121.4737),
        )
    ),
)
```

本仓库：

- 环境变量默认：`USER_LAT_LNG=31.2304,121.4737`（可选）
- 每次覆盖：`grounded_ask(..., lat_lng=make_lat_lng(lat, lng))`
- CLI：`--lat` / `--lng`
- Web：城市芯片、手动输入、浏览器定位；结果页「地理偏置这次用了吗？」卡片会显示本次是否启用

### 怎么在 UI 上验证？

1. 问题填「明天天气怎么样？」
2. 先选「不使用偏置」发送一次，看搜索词与来源
3. 再点「上海」或「旧金山」各发一次，对比 `webSearchQueries` 与回答是否更本地

官方说明见：[Grounding with Google Search – Considerations（地理位置）](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-search#considerations)。

---

## 4.2 弱置信详解（新模型上基本可忽略）

### 客户会问：没分数了，阈值还有意义吗？

**对 Gemini 2.5 / 3.x 等新模型：几乎没有产品意义。**

官方 REST 说明（`GroundingSupport.confidenceScores`）：

> For Gemini 2.0 and before, this list has the same size as `groundingChunkIndices`.  
> **For Gemini 2.5 and later, this list is empty and should be ignored.**

来源：[GroundingMetadata](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GroundingMetadata)

因此：

| 模型代际 | `confidenceScores` | 应用层阈值滑块 |
|----------|-------------------|----------------|
| ≤ Gemini 2.0 | 通常有 0~1 分 | 可用来标弱句 |
| ≥ Gemini 2.5（客户主流） | **官方规定为空** | **调了也没数字可比** |

客户不会“开倒车”只用 2.0。方案里应明确：**数值置信度不是新模型 grounding 的可靠能力**；有来源映射（`groundingChunkIndices`）仍然有价值，但那是“引用了谁”，不是“有多可信的分数”。

### 新模型上什么还有意义？

按优先级：

1. **校验是否真的接地**（有无 `groundingMetadata`）  
2. **展示来源 + 文内引用 + Search Suggestions**（可人工点进网页核对）  
3. **System prompt：无据不编造**  
4. **二次核对 / URL context**（高风险字段）  
5. **地理偏置、排除域名**（场景相关）  
6. ~~依赖 `confidenceScores` 阈值过滤~~ ← 新模型上不要当主路径  

### 本仓库为何还留着阈值控件？

仅作对照演示：若偶尔打到仍返回分数的旧代模型，或阅读旧文档样例时能看懂字段。  
**面向客户的方案叙述应以“2.5+ 无分、应忽略”为准，把降幻觉预算花在引用展示与核对流程上。**


---

## 5. 端到端推荐流水线（可直接当方案图）

```text
① 用户提问
② （可选）查询改写：补时间 / 实体 / 地区
③ generateContent
      - tools: googleSearch
      - temperature: 1.0
      - system: 无据不编
      - 可选 exclude_domains / latLng / blocking_confidence
④ 检查 grounding_metadata
      - 无 → 重试或拒答
      - 有 → 继续
⑤ 解析 groundingSupports
      - 低置信句标记或剔除
⑥ （可选）第二轮 verify（无 Search）
⑦ 展示：回答 + 文内引用 + 来源列表 + Search Suggestions（原样 HTML）
```

---

## 6. 常见误区

| 误区 | 更准确的理解 |
|------|----------------|
| “开了 grounding 就不会幻觉” | 只会降低；网页本身也可能错，模型也可能读偏 |
| “挂了工具就一定搜了” | 以有没有 `groundingMetadata` 为准 |
| “温度调到 0 最准” | Search grounding 官方建议 1.0 |
| “和 function calling 塞同一个请求” | Vertex/Enterprise 文档：search 与非 search 工具不要混用 |
| “自己改 Search Suggestions 的 CSS 更美观” | 违反展示要求；必须原样渲染 |
| “Custom Search JSON API 还能随便新开” | 该 API 已不对新客户开放；全网检索优先走 Search grounding |

---

## 7. 和本仓库代码怎么对应

| 学习点 | 代码 / UI 位置 |
|--------|----------------|
| 建 Vertex 客户端 | `build_client()` |
| Search + temperature + 排除域 + 地理 | `_build_search_config(lat_lng=...)` |
| 解析 metadata / 引用 | `parse_grounding()`；`min_confidence` 仅旧代有分时有用 |
| 缺失重试 | `grounded_ask()` |
| 第二轮核对 | `verify_claims()` / CLI `--verify` |
| 置信度 UI（标注 2.5+ 常无分） | `web_app.py`；勿对客户宣传为新模型主能力 |
| 地理偏置：预设 / 定位 / 状态卡 | Web「地理偏置」盒子 + 结果「用了吗？」 |
| 网页展示建议条与引用 | `templates/index.html` |

动手建议：

1. 先跑 CLI：`python demo_grounded_search.py "你的问题" --json`  
2. 再开网页：`python web_app.py` → 浏览器打开提示的地址  
3. 对比：同问题在「有 grounding / 无 metadata 重试后」的差异  

---

## 8. 计费与合规速记

- **Gemini 3 + Search grounding**：可能按模型内部发出的**每个搜索 query**计费（一问多搜会多计）。详见官方 Billing 说明。  
- 生产展示 Search Suggestions 与来源是**条款要求**，不是可选项。  
- 每日配额等上限以当前 GCP 文档为准。

---

## 9. 一句话收束

**Search grounding 给你的是“检索 + 生成 + 可机器解析的引用元数据”；新模型上准确性更取决于：是否真的接地、是否展示可点来源、提示词是否禁止无据编造，以及必要时的二次核对——而不是 `confidenceScores` 阈值。**  
把“未接地 / 低置信”当成失败态处理，比指望模型自觉更可靠。

下一步：打开 [`README.md`](./README.md) 看启动命令，或直接运行网页 UI 观察真实的 Google Search Suggestions 样式。
