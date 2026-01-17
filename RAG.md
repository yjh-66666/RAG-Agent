# 企业级 RAG + Agent 项目总结（用于简历与面试）

> 适用目标：北京中厂（后端/平台/AI应用）实习面试。内容围绕你的 `RAG.py` 项目：FastAPI + RAG + Agent + 鉴权 + 审计 + 监控。

---

## 1. 项目一句话介绍（简历最上面用）

- 搭建企业知识库问答服务，提供文档导入/检索/Agent 对话 API，支持 JWT/API Key 鉴权、审计日志与 Prometheus 监控，LLM 使用 DeepSeek，Embedding 本地化。

---

## 2. 项目亮点（你这个项目最加分的点）

### 2.1 工程化与可用性
- 提供完整 API：`/ingest`、`/search`、`/query`、`/agent_chat`、`/health`、`/metrics`，并可通过 Swagger (`/docs`) 直接调试。
- 文档入库链路完整：支持 `txt/pdf/docx` → 解析 → chunk 切分 → embedding → Chroma 持久化。

### 2.2 企业特性
- 鉴权：支持 JWT（默认）与 API Key 两种模式。
- 权限控制：按 `roles` 与 `department` 做访问控制过滤（检索后过滤）。
- 审计：落盘 `audit_logs.jsonl`，记录 ingestion/search/agent 行为与 request_id。

### 2.3 安全意识
- 具备基础 Prompt Injection / 敏感信息（key/token）检测与拒答逻辑。

### 2.4 可观测性（你可在面试里重点讲）
- Prometheus 指标：接口耗时、请求数、RAG 空召回、检索命中数分布、相似度分数分布、LLM 调用耗时、token 用量（若可获取）。

---

## 3. 简历写法（可直接复制改成你自己的）

### 3.1 项目名称（建议）
- 企业级知识库问答系统（RAG + Agent + API）

### 3.2 技术栈
- Python、FastAPI、LangChain、Chroma、Sentence-Transformers（BGE Embedding）、DeepSeek（OpenAI SDK 兼容）、JWT、Prometheus

### 3.3 项目描述（2-3 行）
- 设计并实现企业文档知识库问答服务，支持文档导入、向量检索与 Agent 对话。
- 引入鉴权与审计机制，保障不同角色/部门的访问边界，并通过 Prometheus 暴露性能与质量指标。

### 3.4 个人职责（要点式）
- 实现文档解析（PDF/DOCX/TXT）与文本切分（chunk/overlap），构建 Chroma 向量索引并持久化。
- 设计检索与问答接口，支持角色/部门过滤与“新鲜度”加权排序。
- 构建 Agent 路由：区分 search/doc_info/rag_answer，输出引用来源并记录审计日志。
- 增加监控指标：请求耗时/成功率、空召回比例、检索得分分布、LLM 调用耗时与 token 用量。

### 3.5 可量化指标（建议你补齐的方向）
> 面试官很吃“量化”，没有就先跑个本地压测/小评测集。
- 文档量：N 份文档，切分后 M 个 chunk。
- 性能：检索 P95 耗时、LLM 调用 P95 耗时、整体问答耗时。
- 质量：空召回比例、TopK 命中率（自建 20~50 条 Q/A 小集人工评估）。

---

## 4. 面试官高频追问清单（含回答要点）

### 4.1 端到端链路
- 问：从用户提问到返回答案，发生了什么？
- 答：query → embedding → vector search（TopK）→ 权限/部门过滤 → freshness 加权排序 → 拼 context → LLM 生成 → 返回 answer + sources。

### 4.2 为什么要切分？chunk_size/overlap 怎么选？
- 要点：
  - LLM 上下文长度限制。
  - chunk 太大：召回不准/噪声大；太小：语义断裂。
  - overlap 用于跨段信息连续。

### 4.3 Chroma 的 score 是啥？越大越相关还是越小越相关？
- 要点：不同实现可能返回相似度或距离；需要确认 Chroma 返回的 `score` 语义。
- 建议做法：打印/采样验证 + 文档化；监控 score 分布。

### 4.4 权限控制做在“检索前”还是“检索后”？
- 你当前：检索后过滤。
- 追问：会不会泄露？
- 要点：
  - 后过滤有“侧信道”风险与性能浪费；更强方案是按 ACL 分库/分 collection、或在向量库层做 metadata filter。

### 4.5 如何避免 Prompt Injection / 数据泄露？
- 你当前：输入侧规则检测 + 拒答 + 审计。
- 可扩展：输出侧再扫描、工具调用白名单、敏感字段脱敏。

### 4.6 你如何评估 RAG 的效果？
- 要点：
  - 离线：构建评测集（问题-标准答案-应命中文档），算 topk 命中率。
  - 在线：空召回率、引用覆盖率（回答中引用是否覆盖 sources）、用户反馈。

### 4.7 为什么 embedding 本地、LLM 用 API？
- 要点：成本/隐私/稳定性权衡：embedding 可本地化，LLM 用成熟服务；只上传检索片段并可脱敏。

### 4.8 增量更新、重复入库、删除怎么做？
- 要点：
  - 以 `file_hash + file_path` 做幂等。
  - 版本变更触发重新切分/重写向量。
  - 删除需要 vectorstore 支持按 metadata 删除或重建。

### 4.9 为什么会出现 sources 重复？怎么优化？
- 要点：chunk 检索导致同一文件多个 chunk 命中。
- 优化：按 `source_file/document_id` 聚合、MMR 去冗余、每文件最多保留 N 个 chunk。

### 4.10 线上可观测性你怎么做？
- 要点：Prometheus + 关键指标：
  - p95 延迟
  - 空召回比例
  - 检索分数分布
  - LLM 耗时
  - token 用量/估算费用

---

## 5. 你可以主动讲的“后续优化路线”（加分）

- 混合检索：BM25 + 向量检索融合，提高召回。
- rerank：引入 reranker 模型提升排序质量。
- 多租户：department/tenant 分库或分 collection。
- 评测体系：自动化评测集 + 版本对比。
- 缓存：同 query 结果缓存、embedding 缓存。

---

## 6. 推荐你准备的 Demo 演示流程（面试现场可讲）

1. `/auth/dev_login` 获取 token
2. Swagger `Authorize`
3. `/ingest` 导入 demo_docs
4. `/agent_chat` 提问：例如“总结年假政策并给出引用”
5. 打开 `/metrics` 展示延迟、空召回、LLM 耗时等指标
