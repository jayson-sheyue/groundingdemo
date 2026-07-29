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

一句话：**坐标是味精；Maps / Search 才是菜系。加了味精不等于换了菜。**

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

本仓库 CLI：

```bash
python demo_grounded_search.py "日全食何时？" --mode search
python demo_grounded_search.py "附近咖啡馆" --mode maps --lat 31.2304 --lng 121.4737
```

---

## 4. 新模型上如何提高准确性（现实做法）

官方已写明：Gemini **2.5+** 的 `confidenceScores` **为空，应忽略**。  
因此不要再设计「弱置信阈值过滤」当主方案。

推荐流水线：

1. **选对工具**：事实/网页 → Search；本地商户 → Maps  
2. **需要本地时传 latLng**（运行时定位 / 用户城市，不要写死给所有人）  
3. **校验 `grounding_metadata`**：没有 → 重试或告知未核实  
4. **展示来源与引用片段**（`groundingChunks` + `groundingSupports`）  
5. Search 生产环境须展示 **Search Suggestions**（`searchEntryPoint.renderedContent` 原样渲染）  
6. **System prompt**：无来源不编造  
7. 高风险字段可做**二次核对**（另一次请求，只根据返回来源判定 Supported/Unsupported）  

可选：Vertex 上 Search 可用 `exclude_domains`、钓鱼站 `blocking_confidence`。

---

## 5. 读响应时看什么

| 字段 | 含义 |
|------|------|
| `grounding_metadata` 是否存在 | 是否真正接地 |
| `webSearchQueries` | Search 实际搜词（调试用） |
| `groundingChunks` | 来源（web 或 maps） |
| `groundingSupports` | 哪段回答对应哪些来源索引 |
| `searchEntryPoint.renderedContent` | Search 建议条 HTML（须原样展示） |
| `confidenceScores` | **2.5+ 忽略** |

---

## 6. 和本仓库代码对应

| 能力 | 位置 |
|------|------|
| Search / Maps 切换 | `grounded_ask(..., mode=)` / Web 单选 |
| latLng 上下文 | `_build_config` + Web 地理盒子 |
| 缺 metadata 重试 | `grounded_ask` |
| 二次核对 | `verify_claims` / Web 勾选 |
| 引用高亮 UI | `web_app.py` + `templates/index.html` |

```bash
pip install -r requirements.txt
cp .env.example .env   # 填 GOOGLE_CLOUD_PROJECT
gcloud auth application-default login
python web_app.py      # http://127.0.0.1:8080
```

---

## 7. 给客户的一句话

**Grounding with Google Search 用网页；Grounding with Google Maps 用地图地点；latLng 只是给这两种工具的可选位置上下文——三者不要混为一谈。新模型请用引用与核对流程降幻觉，不要依赖已废弃的置信度分数。**
