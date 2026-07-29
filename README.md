# Grounding Demo（Gemini 2.5+）

Vertex AI + `google-genai`：演示 **Google Search** / **Google Maps** grounding，以及可选的 **latLng 位置上下文**。

- **不含**旧版 `confidenceScores` 阈值方案（2.5+ 官方为空，应忽略）  
- 教程：[learning_guide.md](./learning_guide.md)（含 **Maps ≠ 地理偏置** 说明）

## 官方文档

- [Grounding overview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/overview)
- [Grounding with Google Search](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-search?hl=zh-cn)
- [Grounding with Google Maps](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/grounding/grounding-with-google-maps)

## Search 地理偏置 vs Maps grounding

| | Search + `latLng` | Maps grounding |
|--|-------------------|----------------|
| 工具 | `googleSearch` | `googleMaps` |
| `latLng` | 可选，偏置网页结果 | 强烈建议，「附近」更准 |
| 数据 | 公开网页 | Google Maps 地点 |

**不是一回事。** 详见 learning_guide §2。

## 准备

```bash
cd groundingdemo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 填写 GOOGLE_CLOUD_PROJECT
gcloud auth application-default login
```

## 运行

```bash
# CLI
python demo_grounded_search.py "日全食何时？" --mode search
python demo_grounded_search.py "附近咖啡馆" --mode maps --lat 31.2304 --lng 121.4737 --verify

# Web UI
python web_app.py
# http://127.0.0.1:8080
```

## 项目结构

```
groundingdemo/
├── README.md
├── learning_guide.md
├── requirements.txt
├── .env.example
├── demo_grounded_search.py
├── web_app.py
└── templates/index.html
```

## 降幻觉（新模型现实路径）

与 Web UI **一一对应**的手段说明（能做什么 / 不能做什么）见  
**[learning_guide.md §4](./learning_guide.md#4-ui-上的降幻觉手段必须看清能做什么不能做什么)**。

摘要：选对 Search/Maps → 需要时传 latLng（并注入提示）→ 校验是否接地并重试 → 展示可点来源与引用 → Search 原样展示 Suggestions → 可选二次核对（**不能保证第二轮零幻觉**）。  
**不做**旧版 confidenceScores 阈值。
