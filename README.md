# RAG 企业知识库系统

基于 **RAG（Retrieval-Augmented Generation，检索增强生成）+ LangGraph（工作流编排框架）+ Milvus（向量数据库）** 的企业知识库问答系统。

系统把企业内部的 Markdown 文档构建为向量知识库。用户提问时，先从知识库中检索相关原文，再交由大语言模型（LLM）依据检索到的上下文作答；并通过工作流层的多次质量评估与问题改写机制，压制"幻觉"（hallucination，即模型凭空编造事实），使回答有据可依。

---

## 目录

- [一、项目解决什么问题](#一项目解决什么问题)
- [二、主要功能](#二主要功能)
- [三、安装方法](#三安装方法)
- [四、使用方法](#四使用方法)
- [五、输入输出示例](#五输入输出示例)
- [六、配置说明](#六配置说明)
- [七、系统架构与目录结构](#七系统架构与目录结构)
- [八、常见问题](#八常见问题)

---

## 一、项目解决什么问题

### 1.1 企业知识问答面临的三个现实问题

1. **知识散落、检索低效。** 制度文档、产品手册、技术报告分散在个人电脑与共享盘里，员工要靠文件名和目录结构"凭记忆找"，找到之后再自行阅读并归纳。
2. **直接询问大模型会产生幻觉。** 通用大模型的参数里没有企业内部数据，也无从得知企业内部的术语、流程与口径。对它直接提问，得到的回答可能流畅但失真，且无法溯源到具体条款。
3. **通用问题与内部问题混在一起。** 员工既会问"我们的报销标准是什么"，也会问"当前主流的向量数据库有哪些"。若一刀切全部丢给知识库，知识库外的部分将无据可依；若一律交给联网搜索，内部口径又答不准。

### 1.2 本项目的解法

本项目用一条**检索 + 评估 + 改写**的工作流链路，把上述问题拆解并逐一闭环：

| 环节 | 技术手段 | 解决的问题 |
|---|---|---|
| 知识入库 | 文档解析 → 标题正文合并 → 语义切片 → 混合向量入库 | 知识散落、检索低效 |
| 召回 | 密集向量（BGE）+ 稀疏向量（BM25）经 RRF 融合检索 | 纯向量检索易漏掉专有名词、型号、编号 |
| 质量控制 | 文档相关性评分 → 幻觉检测 → 答题检测 | 幻觉、检索噪声污染回答 |
| 动态兜底 | 问题改写重检索 / 路由到联网搜索 | 知识库覆盖不到的问题 |

具体而言，系统在生成回答前后设置了**三道质量闸门**：

- **文档相关性评分**：检索回来的每篇文档先由 LLM 判定与问题是否相关，不相关的直接丢弃，避免"检索到了但其实不相关"的片段混进上下文。
- **幻觉检测**：生成后校验回答是否确实基于检索到的上下文，未基于上下文则重新生成。
- **答题检测**：校验回答是否真正回答了问题，未答上则改写问题后重新检索。

这三道闸门由 LangGraph 状态图（state graph，用节点与条件边编排的有向图）编排成可循环的工作流；只有在回答通过全部校验后才会返回给用户。

### 1.3 适用场景与边界

**适用**：内部制度/产品手册/技术文档问答、客服知识库、研发文档助手等"答案必须出自指定资料"的场景。

**边界（需明确）**：

- 系统只回答**当前已入库语料**覆盖的问题；语料之外的通用问题依赖联网搜索分支，需要额外配置 Tavily API Key。
- 回答质量取决于三方面：语料本身是否准确完整、Embedding 模型的语义表达能力、LLM 的能力上限。检索不到不等于不存在，只是当前知识库没有覆盖。
- 文档解析当前只实现了 **Markdown** 格式（`datas/md/`）。PDF 虽已安装 `pypdf` 依赖，但解析与切片逻辑尚未接入主流程。

---

## 二、主要功能

### 2.1 功能清单

| 功能 | 说明 | 实现位置 |
|---|---|---|
| 文档解析与入库 | 解析 Markdown → 标题/正文合并 → 语义切片 → 写入 Milvus；支持多进程批量入库 | `documents/` |
| 混合向量检索 | 密集向量（BGE）+ 稀疏向量（BM25）双路召回，经 RRF（Reciprocal Rank Fusion，倒数排名融合）合并 | `documents/milvus_db.py`、`tools/retriever_tools.py` |
| 自适应 RAG 工作流 | 路由 → 检索 → 评分 → 生成 → 幻觉检测 → 答题检测 → 问题改写重检索 | `graph2/` |
| 自我纠正 RAG 工作流 | 智能体自主决定是否检索 → 相关性评分 → 不相关则改写问题重试 | `graph/` |
| 问题路由 | 判断问题应走向量知识库还是网络搜索（Tavily） | `graph2/query_route_chain.py` |
| 联网搜索兜底 | 知识库外的通用问题走 Tavily 检索后作答 | `graph2/web_search_node.py` |
| REST API | FastAPI 提供问答、流式问答、健康检查、入库四个接口 | `api.py` |
| SSE 流式输出 | 逐节点推送执行进度与最终回答 | `api.py`、`service/rag_service.py` |
| 可视化界面 | Streamlit 网页端聊天界面，含会话历史与一键清空 | `streamlit_app.py` |
| 命令行交互 | 交互式菜单与子命令两种启动方式 | `main.py` |

### 2.2 两种工作流的分工

项目内置两套独立的 RAG 工作流，均以 LangGraph 实现：

**① 自我纠正 RAG（Corrective RAG，`graph/`）** — 基于"智能体 + 工具调用"的范式。LLM 作为智能体自主决定是否调用检索工具；检索后评估文档相关性，若不相关则改写问题后重新检索。

```
用户提问 → agent（智能体决定是否调用检索工具）
              ├── 无需检索 ──────────────→ 结束（直接返回）
              ↓ 调用检索工具
           retrieve（检索）
              ↓ 相关性评分
        ┌─────┴─────┐
     不相关         相关
        ↓            ↓
     rewrite ──→ agent（改写后重新决策，可再次发起检索）
                  ↓
               generate（生成）→ 结束
```

**② 自适应 RAG（Adaptive RAG，`graph2/`）** — 在检索前增加问题路由，在生成后增加双重质量校验，是功能更完整的一套。

```
用户提问 → route_question（问题路由）
            ├──→ web_search（联网搜索）──→ generate（生成）
            └──→ retrieve（检索）
                    ↓ grade_documents（文档相关性评分）
              ┌─────┴─────┐
           全部无关         有相关文档
              ↓              ↓
        transform_query ──→ generate（生成）
        （≤2 次后转联网）      ↓ 幻觉检测 + 答题检测
              ↑            ┌──┴──┐
              └────────── 没用   合格
                        （改写重检索）  ↓
                                      结束
```

> **FastAPI 与 Streamlit 默认调用第 ② 套（`graph2/`）**；`graph/` 可通过 `python main.py rag1` 单独启动。两者都通过 `tools/retriever_tools.py` 复用同一个检索器。

### 2.3 检索参数（可调）

| 参数 | 当前值 | 位置 | 含义 |
|---|---|---|---|
| `k` | 4 | `tools/retriever_tools.py` | 每次检索返回的文档数 |
| `score_threshold` | 0.1 | `tools/retriever_tools.py` | 相似度低于该值的文档被丢弃 |
| `ranker_type` / `ranker_params` | `rrf` / `{"k": 100}` | `tools/retriever_tools.py` | 密集与稀疏两路结果的融合方式 |
| `filter` | `{"category": "content"}` | `tools/retriever_tools.py` | 只检索"标题+正文"合并后的内容片段 |
| `dense` 维度 | 512 | `documents/milvus_db.py` | 需与 Embedding 模型输出维度一致 |
| 索引 | HNSW（内积 IP）+ 稀疏倒排（BM25） | `documents/milvus_db.py` | 密集走 HNSW 图索引，稀疏走 BM25 倒排索引 |

---

## 三、安装方法

### 3.1 环境要求

| 项目 | 要求 | 说明 |
|---|---|---|
| Python | 3.10 ~ 3.13 | 依赖版本已在 `requirements.txt` 中锁定 |
| 操作系统 | Windows / macOS / Linux | 入库的多进程实现对 Windows 做过适配 |
| 向量数据库 | Milvus 2.5+ 单机版（Standalone） | **必须是 Standalone，不能用 Milvus Lite** —— Lite 不支持 BM25 稀疏检索 |
| Docker | 建议安装 | 用于一条命令拉起 Milvus；也可连接远程 Milvus |
| 大模型 API | 任意 OpenAI 兼容接口 | 需可访问；国内可用通义千问、DeepSeek 等 |
| 联网搜索 | Tavily API Key（可选） | 仅"自适应 RAG"的联网分支需要 |

### 3.2 安装 Python 依赖

```bash
# 建议使用虚拟环境
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> **注意**：`sentence-transformers` 会一并安装 PyTorch、Transformers 等较大依赖，首次安装耗时较长。如需 GPU 加速，请自行安装对应 CUDA 版本的 PyTorch。
>
> `requirements.txt` 中显式列出了 `markdown`：`unstructured` 的基础依赖不含它，而解析 Markdown 时会导入 `unstructured.partition.md`，缺失会直接报 `ModuleNotFoundError: No module named 'markdown'`。

### 3.3 启动 Milvus 向量数据库

项目自带 `docker-compose.yml`，一条命令拉起 Milvus 单机版（含 etcd、MinIO、Milvus 三个容器）：

```bash
docker compose up -d
```

启动完成后向量库地址为 `http://localhost:19530`。可通过健康检查确认：

```bash
curl http://localhost:19530/healthz
```

若使用远程 Milvus，无需执行本步，直接在 `.env` 中把 `MILVUS_URI` 改为远程地址即可。

### 3.4 配置环境变量

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

编辑 `.env`，至少需要填写 `OPENAI_API_KEY`：

```ini
OPENAI_API_KEY=你的_API_Key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
MILVUS_URI=http://localhost:19530
COLLECTION_NAME=t_collection01
```

`.env` 已被 `.gitignore` 忽略，不会提交到 Git。若改用国产模型，只需替换前两项：

```ini
OPENAI_API_KEY=你的_deepseek_key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

### 3.5 安装自检

逐条确认以下三项，可确保环境就绪：

```bash
# ① 依赖齐全（应输出"全部模块导入正常"或仅剩 Milvus 连接类报错）
python -c "import api, main, streamlit_app; print('依赖 OK')"

# ② API Key 已读取
python -c "from utils.env_utils import OPENAI_API_KEY; print('Key OK' if OPENAI_API_KEY else '缺少 Key')"

# ③ 向量库可连接（启动 api.py 后执行）
curl http://127.0.0.1:8000/health
```

> 首次运行时 BGE 向量模型（`BAAI/bge-small-zh-v1.5`）需联网下载到本地缓存。国内建议设置镜像：
> ```bash
> # Windows
> set HF_ENDPOINT=https://hf-mirror.com
> # macOS / Linux
> export HF_ENDPOINT=https://hf-mirror.com
> ```

---

## 四、使用方法

### 4.1 语料入库

把待入库的 Markdown 文档放入 `datas/md/` 目录（当前已内置 436 个示例文档），然后执行：

```bash
# 多进程批量入库（推荐，海量数据更快）
python main.py ingest
```

也可以通过 REST API 单进程入库：

```bash
curl -X POST http://127.0.0.1:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"md_dir": ""}'
```

> ⚠️ **重要**：入库前会**清空并重建**目标集合（`documents/milvus_db.py` 的 `create_collection` 先释放集合、删除索引、再删除集合）。执行前请确认 `COLLECTION_NAME` 配置无误，原有数据将无法恢复。

### 4.2 命令行对话

```bash
# 交互式菜单
python main.py

# 直接启动自适应 RAG（推荐）
python main.py rag2

# 直接启动自我纠正 RAG
python main.py rag1
```

菜单选项：`1` 自适应 RAG 对话、`2` 自我纠正 RAG 对话、`3` 批量入库、`q` 退出。对话中输入 `q` / `exit` / `quit` 结束。

### 4.3 启动 REST API

```bash
python api.py
# 或
uvicorn api:app --host 0.0.0.0 --port 8000
```

启动后访问接口文档（Swagger UI，由 FastAPI 自动生成）：**http://127.0.0.1:8000/docs**

| 接口 | 方法 | 请求体 | 说明 |
|---|---|---|---|
| `/health` | GET | — | 健康检查，返回 `{"status": "ok"}` |
| `/chat` | POST | `{"question": "..."}` | 问答，返回完整回答（阻塞至全部节点执行完毕） |
| `/chat/stream` | POST | `{"question": "..."}` | 问答，SSE 流式返回执行进度与回答 |
| `/ingest` | POST | `{"md_dir": ""}` | 入库文档（单进程），`md_dir` 留空则用 `datas/md` |

### 4.4 启动可视化界面

```bash
streamlit run streamlit_app.py
```

浏览器自动打开 **http://localhost:8501**，在聊天框直接提问即可。侧边栏提供使用说明与"清空对话历史"按钮。

### 4.5 迁移到自有业务场景

1. **替换语料**：清空 `datas/md/`，放入自有 Markdown 文档，重新执行 `python main.py ingest`。
2. **调整检索器描述**：`tools/retriever_tools.py` 中 `create_retriever_tool` 的第三个参数描述了知识库的业务范围，LLM 会据此判断检索工具的适用性。
3. **调整路由提示词**：`graph2/query_route_chain.py` 的 `system` 变量描述了知识库覆盖范围，需同步修改为自有业务主题。
4. **确认向量维度**：若更换 Embedding 模型，须同步修改 `documents/milvus_db.py` 中 `dense` 字段的 `dim`，并重建集合。

---

## 五、输入输出示例

> **示例来源说明**：5.1 的片段统计、5.2 的节点序列与回答文本、5.3 的执行轨迹，均取自本项目的实际运行结果；5.4 / 5.5 的接口报文依据 `api.py` 与 `service/rag_service.py` 的实际返回结构给出，接口路径与响应字段与 FastAPI 实际注册结果一致。示例语料为 `datas/md/` 中的 Milvus 官方文档，换成自有语料后，检索到的片段与回答内容会相应变化。

### 5.1 入库：输入与输出

**输入**：`datas/md/` 目录下的 Markdown 文件（以 `product_faq.md` 为例）

**输出**（控制台日志，实测）：

```text
20261028 10:46:39 | MainProcess | MainThread | markdown_parser.parse_markdown_to_documents:31 | INFO: 文件解析后的docs长度: 88
20261028 10:46:39 | MainProcess | MainThread | markdown_parser.parse_markdown_to_documents:35 | INFO: 文件合并后的长度: 33
20261028 10:46:39 | MainProcess | MainThread | markdown_parser.parse_markdown_to_documents:38 | INFO: 语义切割后的长度: 33
```

三行日志对应三个处理阶段：

- `文件解析后的docs长度` —— Unstructured 按元素（element）解析出的原始节点数
- `文件合并后的长度` —— 标题与其下属正文合并为一条后的片段数
- `语义切割后的长度` —— 超过 5000 字符的片段再按语义边界切分后的最终片段数

写入环节的日志格式（`documents/write_milvus.py` 的多进程写入进程）：

```text
20261028 10:46:39 | MainProcess | MainThread | write_milvus.milvus_writer_process:70 | INFO: 累计已写入: 33 个文档
20261028 10:46:39 | MainProcess | MainThread | write_milvus.milvus_writer_process:75 | INFO: 写入进程结束，总计写入 33 个文档
```

> 计数含义：单文件写入时为该文件的片段数（上例 33 对应 `product_faq.md`）；执行 `python main.py ingest` 时为**全目录累加**的片段总数。
>
> 若通过 API 的 `/ingest`（单进程）入库，逐文件日志格式为 `已入库 product_faq.md：33 个片段`。

**实测各文件的片段产出**：

| 输入文件 | 解析元素数 | 合并后 | 语义切割后 |
|---|---|---|---|
| `product_faq.md` | 88 | 33 | 33 |
| `overview.md` | 85 | 17 | 17 |
| `operational_faq.md` | 51 | 19 | 19 |
| `performance_faq.md` | 22 | 7 | 7 |
| **合计** | **246** | **76** | **76** |

> 注：76 个片段中有 69 个的 `category` 为 `content`（即"标题+正文"合并片段），其余为 `Title` 与 `NarrativeText`。检索器的 `filter` 只取 `category == "content"`，因此实际可召回 69 个片段。

**入库后的集合结构**（`documents/milvus_db.py` 中定义的 schema）：

```text
id              INT64              主键，自增
text            VARCHAR(6000)      片段正文（启用 jieba 分词器，供 BM25 使用）
category        VARCHAR(1000)      片段类别：Title / content / NarrativeText
source          VARCHAR(1000)      源文件路径
filename        VARCHAR(1000)      源文件名
filetype        VARCHAR(1000)      文件类型
title           VARCHAR(1000)      所属标题
category_depth  INT64              标题层级深度
sparse          SPARSE_FLOAT_VECTOR  BM25 稀疏向量（由 text 自动生成）
dense           FLOAT_VECTOR(512)    BGE 密集向量
```

### 5.2 命令行问答：输入与输出

**输入**（`python main.py rag2` 后在提示符输入）：

```text
用户：Milvus 支持哪些索引类型？
```

**输出**（实测；`graph2` 以 `pprint` 逐节点打印，字符串带引号）：

```text
"Node 'retrieve':"
'\n---\n'
"Node 'grade_documents':"
'\n---\n'
"Node 'generate':"
'\n---\n'
"Node 'transform_query':"
'\n---\n'
"Node 'retrieve':"
'\n---\n'
"Node 'grade_documents':"
'\n---\n'
"Node 'generate':"
'\n---\n'
'根据上下文，Milvus 支持的索引类型及特点如下：

1.  **HNSW**：基于图的索引，适合高搜索效率场景（有 GPU 版本 GPU_CAGRA）。
2.  **FLAT**：适合小规模数据集的精确搜索（有 GPU 版本 GPU_BRUTE_FORCE）。
3.  **IVF_FLAT**：量化索引，平衡精度与查询速度（有 GPU 版本 GPU_IVF_FLAT）。
4.  **IVF_SQ8**：量化索引，显著降低内存消耗。
5.  **IVF_PQ**：量化索引，追求高查询速度（有 GPU 版本 GPU_IVF_PQ）。
6.  **SCANN**：类似 IVF_PQ，利用 SIMD 技术优化计算。
7.  **DiskANN**：基于 Vamana 图，适合大规模数据集搜索。

若未指定索引类型，Milvus 默认执行暴力搜索（brute-force search）。'
```

### 5.3 工作流执行轨迹：输入与输出

同一个问题的内部执行过程（`graph2` 节点日志，实测）：

```text
---ROUTE QUESTION---
---路由到RAG系统---                                    ← 路由到向量知识库
---去知识库中检索文档---                                 ← retrieve
---CHECK DOCUMENT RELEVANCE TO QUESTION---              ← grade_documents
---GRADE: 打印不相关标识,并丢掉doc---                   ← 4 篇文档中筛掉 3 篇
---GRADE: 打印相关标识---
---ASSESS GRADED DOCUMENTS---
---决策：生成最终回答---                                 ← decide_to_generate
---检查生成内容是否存在幻觉---                            ← 幻觉检测
---判定：生成内容基于参考文档---
---评估：生成回答与问题的匹配度---                        ← 答题检测
---判定：生成内容未能准确回答问题---
---TRANSFORM QUERY---                                   ← 改写问题后重新检索
---去知识库中检索文档---
---CHECK DOCUMENT RELEVANCE TO QUESTION---
---决策：生成最终回答---
---检查生成内容是否存在幻觉---
---判定：生成内容基于参考文档---
---评估：生成回答与问题的匹配度---
---判定：生成内容准确回答问题---                        ← 通过全部校验，返回答案
```

**对应的工作流节点序列**：

```text
retrieve → grade_documents → generate → transform_query → retrieve → grade_documents → generate
```

可以清楚看到"检索结果不理想 → 改写问题 → 重新检索 → 回答合格"的自纠正闭环。

### 5.4 REST API：输入与输出

**① 健康检查**

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok"}
```

**② 问答（非流式）**

请求：

```bash
curl -X POST http://127.0.0.1:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "Milvus 支持哪些索引类型？"}'
```

响应：

```json
{
  "question": "Milvus 支持哪些索引类型？",
  "answer": "根据上下文，Milvus 支持的索引类型及特点如下：\n\n1.  **HNSW**：基于图的索引，适合高搜索效率场景（有 GPU 版本 GPU_CAGRA）。\n2.  **FLAT**：适合小规模数据集的精确搜索（有 GPU 版本 GPU_BRUTE_FORCE）。\n..."
}
```

**③ 入库**

```bash
curl -X POST http://127.0.0.1:8000/ingest \
     -H "Content-Type: application/json" \
     -d '{"md_dir": ""}'
```

响应：

```json
{"status":"ok","ingested":76}
```

> `ingested` 为本次入库的**总片段数**。上例的 `76` 是 `datas/md/` 仅含 5.1 节那 4 个文件时的实测值；若使用项目内置的完整语料目录（436 个文件），该数值远大于此。

失败时返回：

```json
{"status":"error","message":"<异常信息>"}
```

### 5.5 SSE 流式问答：输入与输出

**输入**与 `/chat` 相同：

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
     -H "Content-Type: application/json" \
     -d '{"question": "Milvus 支持哪些索引类型？"}'
```

**输出**（`text/event-stream`，每个事件为 `data: <JSON>\n\n`，实测节选）：

```text
data: {"node": "retrieve"}

data: {"node": "grade_documents"}

data: {"node": "generate"}

data: {"answer": "提供的上下文中未包含 Milvus 支持的具体索引类型信息。..."}

data: {"node": "transform_query"}

data: {"node": "retrieve"}

data: {"node": "grade_documents"}

data: {"node": "generate"}

data: {"answer": "根据上下文，Milvus 支持的索引类型及特点如下：1.  **HNSW**：基于图的索引..."}

data: [DONE]
```

事件负载有两种形态：

| 负载 | 含义 |
|---|---|
| `{"node": "<节点名>"}` | 当前执行到的 LangGraph 节点，用于前端展示执行进度 |
| `{"answer": "<回答文本>"}` | `generate` 节点产出的回答内容 |

结束标志为固定字符串 `data: [DONE]`。

> 前端消费示例：
> ```javascript
> const es = new EventSource("/chat/stream", { method: "POST", body: JSON.stringify({ question }) });
> es.onmessage = (e) => {
>   if (e.data === "[DONE]") { es.close(); return; }
>   const item = JSON.parse(e.data);
>   if (item.node) console.log("节点:", item.node);
>   if (item.answer) console.log("回答:", item.answer);
> };
> ```

### 5.6 Streamlit 界面：输入与输出

- **输入**：页面底部聊天框输入问题并回车，例如 `Milvus 支持哪些索引类型？`
- **输出**：聊天区先显示用户消息气泡，再显示"正在检索并生成回答..."加载态，随后输出助手消息气泡（内容同 5.2 的回答）。侧边栏"清空对话历史"按钮会重置会话。

---

## 六、配置说明

`.env` 配置项一览（完整模板见 `.env.example`）：

| 变量 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | 空 | 大模型 API Key |
| `OPENAI_BASE_URL` | ✅ | `https://api.openai.com/v1` | 接口地址，兼容任意 OpenAI 兼容服务 |
| `OPENAI_MODEL` | ✅ | `gpt-4o-mini` | 模型名称 |
| `DEEPSEEK_API_KEY` | ❌ | 空 | DeepSeek 专用 Key，当前代码未直接使用 |
| `TAVILY_API_KEY` | ❌ | 空 | 联网搜索分支所需；不填则该分支不可用 |
| `MILVUS_URI` | ✅ | `http://localhost:19530` | Milvus 地址 |
| `COLLECTION_NAME` | ✅ | `t_collection01` | 集合名 |
| `EMBEDDING_MODEL_NAME` | ✅ | `BAAI/bge-small-zh-v1.5` | 本地 Embedding 模型 |
| `EMBEDDING_DEVICE` | ✅ | `cpu` | 计算设备，`cpu` 或 `cuda` |

配置读取逻辑位于 `utils/env_utils.py`：优先读取项目根目录的 `.env`（`override=True`，即 `.env` 中的值优先于系统环境变量），`.env` 不存在时回退到系统环境变量。

---

## 七、系统架构与目录结构

### 7.1 技术栈

| 层次 | 技术 | 说明 |
|---|---|---|
| 大模型（LLM） | OpenAI 兼容接口 | 默认 `gpt-4o-mini`，可替换为任意兼容模型 |
| 工作流编排 | LangGraph | 用状态图编排"路由→检索→评分→生成→改写"全流程 |
| 向量数据库 | Milvus 2.5+ | 密集向量 + BM25 稀疏向量的混合检索 |
| 向量化（Embedding） | BGE（`bge-small-zh-v1.5`） | 本地开源中文向量模型，512 维 |
| 语义切片 | SemanticChunker | 按语义边界切割超长文本 |
| 文档解析 | Unstructured | Markdown 元素级解析 |
| 网络搜索 | Tavily | 知识库外问题的兜底分支 |
| Web 服务 | FastAPI + Uvicorn | REST API + SSE 流式 |
| 可视化 | Streamlit | 网页端问答界面 |

### 7.2 目录结构

```text
RAG企业知识库/
├── main.py                  # 项目入口（菜单 / ingest / rag1 / rag2）
├── api.py                   # FastAPI 服务（REST API + SSE 流式）
├── streamlit_app.py         # Streamlit 可视化问答界面
├── docker-compose.yml       # 一键启动 Milvus 单机版
├── requirements.txt         # 依赖清单（版本已锁定）
├── .env.example             # 环境变量模板（复制为 .env 后填写）
├── .gitignore
├── README.md
├── service/                 # 服务封装（供 API / Streamlit 复用）
│   └── rag_service.py       #   工作流调用封装（惰性加载图）
├── datas/
│   └── md/                  # 语料目录：Markdown 文档（内置 436 篇示例）
├── documents/               # 数据管道：解析、切片、入库
│   ├── markdown_parser.py   #   Markdown 解析 + 标题合并 + 语义切割
│   ├── milvus_db.py         #   Milvus 建表 / 连接 / 写入
│   └── write_milvus.py      #   多进程批量入库
├── graph/                   # 自我纠正 RAG（Corrective RAG）
│   ├── graph1.py            #   状态图定义与运行
│   ├── agent_node.py        #   智能体节点
│   ├── generate_node.py     #   生成节点
│   ├── rewrite_node.py      #   问题改写节点
│   └── graph_state1.py      #   状态定义
├── graph2/                  # 自适应 RAG（Adaptive RAG）
│   ├── graph_2.py           #   状态图定义与运行
│   ├── query_route_chain.py #   问题路由
│   ├── retriever_node.py    #   检索节点
│   ├── grade_documents_node.py    # 文档相关性评分
│   ├── generate_node2.py    #   生成节点
│   ├── transform_query_node.py    # 查询改写节点
│   ├── web_search_node.py   #   联网搜索节点
│   ├── grader_chain.py      #   相关性评分器
│   ├── grade_hallucinations_chain.py  # 幻觉检测器
│   ├── grade_answer_chain.py        # 答题检测器
│   └── graph_state2.py      #   状态定义
├── llm_models/              # 模型封装
│   ├── all_llm.py           #   对话模型 + Tavily 搜索工具
│   └── embeddings_model.py  #   云端 / 本地两套 Embedding
├── tools/
│   └── retriever_tools.py   # 检索器与检索工具（两套工作流共用）
└── utils/                   # 配置读取、日志、打印工具
```

### 7.3 数据流

```text
datas/md/*.md
      │
      ▼  UnstructuredMarkdownLoader（element 级解析）
      ▼  merge_title_content（标题与下属正文合并，category → content）
      ▼  SemanticChunker（仅对 > 5000 字符的片段做语义切分）
   Document 列表
      │
      ▼  Milvus.add_documents
      │    ├─ dense  ← BGE-small-zh-v1.5（512 维，L2 归一化）
      │    └─ sparse ← BM25 内置函数（基于 text 字段 + jieba 分词）
   Milvus 集合
      │
      ▼  检索（dense IP + sparse BM25 → RRF 融合 → k=4, 阈值 0.1）
   候选片段
      │
      ▼  graph2 工作流：路由 / 评分 / 生成 / 幻觉检测 / 答题检测 / 改写
   最终回答
```

---

## 八、常见问题

**1. 安装后运行报 `ModuleNotFoundError: No module named 'markdown'`**

`unstructured` 的基础依赖不包含 `markdown`，但解析 Markdown 时会导入 `unstructured.partition.md`。`requirements.txt` 中已显式列出 `markdown`，若环境创建较早请手动补装：

```bash
pip install markdown==3.11.2
```

**2. 首次运行卡在下载模型**

BGE 向量模型需联网下载到本地缓存。国内建议设置镜像：

```bash
# Windows
set HF_ENDPOINT=https://hf-mirror.com
# macOS / Linux
export HF_ENDPOINT=https://hf-mirror.com
```

**3. 更换 Embedding 模型后检索报错**

需同步修改 `documents/milvus_db.py` 中 `dense` 字段的 `dim`（`bge-small-zh-v1.5` = 512 维，`bge-large-zh-v1.5` = 1024 维），并重建集合。

**4. 联网搜索报"未配置 TAVILY_API_KEY"**

到 [tavily.com](https://tavily.com) 申请免费 Key 填入 `.env`。不填则知识库外的通用问题无法走联网搜索分支。

**5. BM25 全文检索不生效**

Milvus Lite 不支持 BM25 内置函数，必须使用 Milvus 单机版（Standalone，远程或 `docker compose` 本地均可）。本项目的 BM25 依赖 `text` 字段启用 jieba 分词器，该配置在 `documents/milvus_db.py` 中。

**6. 问答响应偏慢**

工作流每次问答需多次调用 LLM（路由 + 逐篇文档评分 + 生成 + 幻觉检测 + 答题检测），并可能因"答题检测未通过"而多轮重试，调用次数会成倍增加。降低延迟可从减小 `k`（检索文档数）、换用更快的模型、或在 `service/rag_service.py` 的 `ask()` 中为 `graph.invoke` 传入 `config={"recursion_limit": N}` 限制最大轮次。

**7. 结构化评分报错或输出不对**

qwen 系列模型对 JSON 结构化输出较敏感。代码已在各评分提示词中加入"只输出 JSON"指令，请勿删除这些提示词。

**8. 自适应 RAG 出现多轮循环，或接口返回 500**

当"答题检测"判定回答未解决用户问题时，工作流会改写问题并重新检索，该路径**没有独立的次数上限**（`transform_count` 仅在"检索结果全部无关"时才作为切换到联网搜索的判据）。

LangGraph 的默认执行步数上限为 25（`DEFAULT_RECURSION_LIMIT`），超出后会抛出 `GraphRecursionError`；`api.py` 未捕获该异常，接口将返回 500。正常情况下回答会收敛；如需限制最大轮次或提升上限，可在 `service/rag_service.py` 的 `ask()` / `ask_stream()` 中为 `graph.invoke` / `graph.stream` 传入 `config={"recursion_limit": N}`（N 为允许的最大执行步数）。

若问题本身在知识库中无解、反复触发改写重检索，可同时在 `graph2/graph_2.py` 的 `grade_generation_v_documents_and_question` 中，对"未解决"分支增加重试次数计数以提前收敛。

**9. 入库后检索不到内容**

按以下顺序排查：① 确认 `COLLECTION_NAME` 与入库时一致；② 确认 `datas/md/` 下的文档确为 `.md` 结尾且非空；③ 确认检索器的 `filter`（`category == "content"`）与入库片段的 `category` 一致 —— 纯标题片段（`category == "Title"`）不会进入候选集；④ 确认 Embedding 模型与入库时一致。

**10. Windows 下多进程入库异常**

批量入库采用 `multiprocessing` 双进程（解析进程 + 写入进程）实现。若在 Windows 上遇到 `spawn` 相关报错，可改用单进程接口：

```bash
python main.py ingest   # 多进程，推荐
# 或调用 API 的 /ingest（单进程）
```
