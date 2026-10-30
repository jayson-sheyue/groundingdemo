# Gemini Grounding Lab

一个用于学习和演示 **Gemini Enterprise Agent Platform（Vertex AI）Grounding** 的 Web / Python 项目。选择数据来源、输入问题并运行后，可以查看模型回答、来源、引用关系和 SDK 响应字段。

本项目演示十种数据来源：Google Search、Image Search 生图、Google Maps、URL Context、Agent Search、RAG Engine、自定义 Search API、Elasticsearch、Parallel / Exa Web 和 Enterprise Web。每种模式的模型兼容性、区域、权限、开通方式和计费可能不同，**代码中存在某个选项不代表项目一定可以调用该能力**。

- [学习指南](./learning_guide.md)：概念、模式对比、字段解释、示例和官方文档。
- Web 页面中的 **“第一次使用？”**：从选择模式到检查来源的操作步骤。
- 启动 Web 后打开 [参数与返回字段手册](http://127.0.0.1:8080/api-reference)：依据服务器安装的 SDK 动态列出相关字段。远程部署时，在服务地址后加 `/api-reference`。

## 五分钟开始

1. 准备有权访问 Vertex AI 的 Google Cloud 项目，启用相应 API、计费和所需 Grounding 能力。搜索数据源、RAG corpus、伙伴服务及所选模型需要各自的权限和前置配置。
2. 安装 [Python 3.11 或更高版本](https://www.python.org/downloads/)及 [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)。项目当前锁定 `google-genai==2.34.0`；部署容器使用 Python 3.12。
3. 获取源码并配置：

   ```bash
   git clone https://github.com/jayson-sheyue/groundingdemo.git
   cd groundingdemo
   python3 -m venv .venv
   source .venv/bin/activate       # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env           # Windows: copy .env.example .env
   ```

4. 编辑 `.env`，填写 `GOOGLE_CLOUD_PROJECT` 和 `GOOGLE_CLOUD_LOCATION`。`GCS_BUCKET` 仅在需要上传附件或使用相对 GCS 路径时配置；不要把真实凭据提交到仓库。使用本地 Application Default Credentials：

   ```bash
   gcloud auth application-default login
   ```

5. 启动：

   ```bash
   python web_app.py
   ```

   浏览器打开 [http://127.0.0.1:8080](http://127.0.0.1:8080)。先选择模式、填写对应资源，再填写问题和支持该工具的模型 ID。高级选项可以先留空。点击“发送”，查看回答、来源和引用；可展开 SDK 响应快照查看字段。

默认文本模型 ID 是 `gemini-2.5-flash`，图片模式有独立默认模型；它们是 Demo 默认值，并非适用于所有模式或项目的推荐。可在界面或 CLI 中修改，实际可用性以项目、区域和模型支持情况为准。

## 选择数据源

| 模式 | 数据来源及用途 | 主要前提 |
|---|---|---|
| Search | Google 网页搜索；回答公开事实 | 模型与项目支持 Google Search grounding |
| Image Search | 搜索图片作为视觉参考并生成图片 | 支持该工具和图片输出的模型；输出与来源可能不同步返回 |
| Maps | Google Maps 地点信息 | 项目支持 Maps grounding；“附近”问题可提供坐标 |
| URL Context | 指定公开网页或 PDF | 在问题中提供可访问 URL；查看 `urlContextMetadata` |
| Agent Search | 企业网站或文档搜索 | 创建并授权访问 data store 或 engine，二选一填写 |
| RAG Engine | 受管 corpus 中的文件和片段 | 创建 corpus、导入文件，并授权服务身份访问 |
| 自定义 Search API | 自有搜索服务 | 符合 `SIMPLE_SEARCH` 协议的 endpoint 和 API Key |
| Elasticsearch | 已有 Elastic 索引 | endpoint、API Key、index、search template |
| Parallel / Exa Web | 第三方 Web 检索 | 满足对应订阅或 API Key 及项目权限要求 |
| Enterprise Web | 面向企业的 Web 检索 | 确认项目、区域和组织配置已开通 |

图片模式使用“发送”获取完整生成结果；其余文本模式可以尝试流式发送。详细前提及官方链接见 [学习指南](./learning_guide.md)。

## 请求参数

表单只显示所选模式相关的连接参数。**留空表示 Demo 不显式设置该项**，实际默认行为由服务、模型和数据源决定；例外是 `temperature` 留空时，本 Demo 显式传入 `1.0`。

| 位置 | 参数 | 用途与注意事项 |
|---|---|---|
| 通用 | `temperature`、`max_output_tokens` | 控制生成随机性与输出长度；具体模型可能有不同限制 |
| 检索上下文 | `language_code`、`latLng` | 可选语言和用户坐标；坐标不等于启用 Maps 工具 |
| Search | `exclude_domains`、`blocking_confidence` | 排除域名和设置钓鱼风险拦截等级；后者不是答案可信度阈值。留空时可使用 `BLOCKING_CONFIDENCE` 环境变量 |
| Maps | `enable_widget` | 当前 SDK 将该字段标为弃用，可能不再生效；本 Demo 使用默认 Places，不提供 Routing 配置 |
| Agent Search | data store / engine、`filter`、`max_results` | data store 与 engine 二选一；`max_results` 为 1–10，过滤语法取决于数据源 |
| RAG Engine | `rag_corpus`、`rag_file_ids`、`top_k`、距离 / 相似度阈值、`metadata_filter`、混合检索和重排 | 检索参数按请求生效；距离与相似度阈值二选一。混合检索当前仅限支持该能力的 Weaviate 后端；重排模型和费用需另行确认 |
| Elasticsearch | endpoint、index、template、`num_hits` | `num_hits` 请求检索的片段数量；服务端需要可访问该 endpoint |
| 伙伴检索 | 伙伴选择、API Key | 凭据仅供本次请求使用，不回填表单或写入运行记录 |

距离阈值例如 `0.5` 表示只保留距离**小于** `0.5` 的候选片段；相似度阈值则保留相似度**大于**阈值的片段。这些数值不是答案正确率，应使用实际语料和问题评测。`top_k` 是数量上限，过滤后可能不足该数量。来源：[RAG Engine API](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/rag-api)。

### Python 示例：RAG Engine

此示例需先配置项目与 Application Default Credentials，使用实际可用的模型和 corpus。`0.5` 是演示值，不是通用推荐值。

```python
from google import genai
from google.genai import types

client = genai.Client(vertexai=True, project="PROJECT_ID", location="global")
rag_tool = types.Tool(
    retrieval=types.Retrieval(
        vertex_rag_store=types.VertexRagStore(
            rag_resources=[
                types.VertexRagStoreRagResource(
                    rag_corpus="projects/PROJECT_ID/locations/REGION/ragCorpora/CORPUS_ID"
                )
            ],
            rag_retrieval_config=types.RagRetrievalConfig(
                top_k=10,
                filter=types.RagRetrievalConfigFilter(
                    vector_distance_threshold=0.5
                ),
            ),
        )
    )
)
response = client.models.generate_content(
    model="MODEL_ID",
    contents="请根据知识库说明产品的退款政策。",
    config=types.GenerateContentConfig(tools=[rag_tool]),
)
print(response.text)
for candidate in response.candidates or []:
    if candidate.grounding_metadata:
        print(candidate.grounding_metadata.grounding_chunks)    # 来源片段
        print(candidate.grounding_metadata.grounding_supports)  # 回答与来源的关系
```

官方示例：[Generate responses using the RAG file](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-rag-generate-content)。

### CLI 示例

```bash
# Google Search
python demo_grounded_search.py "解释什么是 Grounding 并列出来源" --mode search --model MODEL_ID

# RAG Engine：检索参数随本次请求发送
python demo_grounded_search.py "知识库中的退款政策是什么？" \
  --mode rag_engine --model MODEL_ID \
  --rag-corpus projects/PROJECT_ID/locations/REGION/ragCorpora/CORPUS_ID \
  --rag-top-k 10 --rag-distance 0.5

# Agent Search：data store 与 engine 二选一
python demo_grounded_search.py "总结企业资料" --mode agent_search --model MODEL_ID \
  --agent-search-datastore projects/PROJECT_ID/locations/global/collections/default_collection/dataStores/DATASTORE_ID

# Maps：坐标是位置上下文
python demo_grounded_search.py "附近有什么咖啡馆？" --mode maps --model MODEL_ID --lat 31.2304 --lng 121.4737

# 流式输出或统计输入 token
python demo_grounded_search.py "解释 Grounding" --mode search --model MODEL_ID --stream
python demo_grounded_search.py "解释 Grounding" --mode search --model MODEL_ID --count-tokens

# 查看全部 CLI 选项
python demo_grounded_search.py --help
```

CLI 的 `--verify` 仅在响应中包含有正文的 `retrievedContext` 来源时执行二次核对；它不会重新检索，也不能保证正确。`--no-retry` 关闭普通生成请求的补充尝试。

## 阅读响应与排查

- `response.text` 只包含模型回答；`candidates[].grounding_metadata` 才包含可用时返回的 grounding 元数据。
- `groundingChunks` 描述来源；`groundingSupports` 将回答片段关联到来源索引。高亮位置按 UTF-8 字节偏移计算。
- `webSearchQueries`、`imageSearchQueries`、`retrievalQueries` 和其他辅助字段是否存在，取决于数据源和响应。URL Context 使用独立的 `urlContextMetadata`。
- **单个字段为空或缺少 metadata，不能单独证明没执行检索；有来源或引用也不能保证答案正确。** 可以使用同一 corpus、问题与参数单独调用 `retrieveContexts` 核对召回片段，但它不能证明另一次生成请求内部的结果完全相同。
- 普通发送可在缺少元数据或有效结果时追加提示并再调用一次模型；这会增加耗时和费用，不保证成功。流式发送不使用该重试。
- 普通结果支持查看整理后的 metadata、SDK 解析后的非空 metadata 和响应对象；为控制大小，响应快照省略 HTTP 响应头及内嵌二进制媒体。流式快照仅展示最后一个带 metadata 的事件，未保证合并此前事件。
- `usageMetadata` 是服务报告的 token 用量。Demo 可按环境变量估算部分 token 成本，但不代表全部工具、检索或服务费用。

更多背景和字段说明见 [学习指南第 11 节](./learning_guide.md#11-请求参数与结果排查实操)与 [RAG Engine 输出说明](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/rag-output-explained)。

## HTTP 路由与代码

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/` | 主界面 |
| `GET` | `/api-reference` | 根据当前 SDK 类型生成的参数与返回字段手册 |
| `POST` | `/ask` | 普通生成和 Grounding 展示 |
| `POST` | `/ask/stream` | SSE 流式生成 |
| `POST` | `/count-tokens` | 统计输入 token |
| `POST` | `/upload` | 上传浏览器附件到配置的 GCS bucket |
| `POST` | `/history/clear` | 清空当前进程的运行记录 |

核心实现：`demo_grounded_search.py` 构造工具和解析响应；`web_app.py` 提供 Web 路由；`templates/index.html` 是主界面，`templates/api_reference.html` 是 SDK 字段手册；`gcs_upload.py` 负责上传；`redirect_resolve.py` 可选解析来源链接。运行记录保存在进程内，不是持久化或跨实例历史。

## 开发验证与部署

```bash
python -m unittest discover -s tests -v
node tests/test_citations.js  # 需要 Node.js
python -m compileall -q web_app.py demo_grounded_search.py
```

测试包含本地模拟，不等于所有模型、数据源和权限已完成真实云端联调。`Dockerfile` 和 `deploy-cloudrun.sh` 提供现有 Cloud Run 构建与部署方式；脚本包含项目、服务账号和 IAP 配置，请先根据目标环境检查，再执行。部署需要相应 Cloud Run、Cloud Build、Artifact Registry 和服务账号权限。

官方入口：[Grounding 概览](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/overview) · [Google Gen AI SDK](https://googleapis.github.io/python-genai/) · [GroundingMetadata](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/GroundingMetadata) · [Google Cloud 定价](https://cloud.google.com/vertex-ai/generative-ai/pricing)。
