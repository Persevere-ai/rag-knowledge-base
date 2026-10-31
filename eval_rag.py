"""
RAG 企业知识库 —— 评测脚本

用途：
    量化「检索增强生成（RAG）到底比直接问大模型好多少」，为简历提供真实对比数据。

    做法：准备一组带标准答案关键词的问题，分别用两种方式作答：
      ① 直接问大模型（baseline，无检索上下文）
      ② 走本项目真实的 graph2「自适应 RAG」工作流（路由→检索→评分→生成→幻觉检测→答题检测）
    再用「关键词命中」打分，统计两者正确率，得出「RAG 相对直接提问的提升」。

关于向量库：
    项目默认连远程 Milvus（本机当前不可达）。为让评测可在离线环境跑通，本脚本用一个
    「本地检索器」替换 graph2 的检索节点（通过注入假模块绕过 Milvus 连接），
    工作流的其它部分（路由、评分器、生成、幻觉/答题检测）全部是项目真实代码。
    本地检索器支持两种后端：
      dense   —— BGE 密集向量 + 余弦相似度（与项目 Milvus 的 dense 字段同源）
      keyword —— 字符 bigram 词面重叠（纯 Python，无需下载模型，快速冒烟用）
      milvus  —— 走项目真实的 Milvus 混合检索（BGE 密集 + BM25 稀疏 + RRF 融合），
                 需要远程/本地 Milvus 可用

运行方式（在 RAG企业知识库/ 目录下）：
    .venv\\Scripts\\python eval_rag.py --retriever milvus --limit 3   # 真实 Milvus（推荐）
    .venv\\Scripts\\python eval_rag.py --retriever keyword --limit 3   # 快速冒烟
    .venv\\Scripts\\python eval_rag.py --retriever dense               # 本地 BGE 密集（离线）
    .venv\\Scripts\\python eval_rag.py --list                          # 只看测试集

依赖：
    - 使用本项目 .venv（已装 langgraph / langchain / sentence-transformers）
    - 已配置 .env 的 OPENAI_API_KEY（脚本用真实大模型跑）
"""

import argparse
import csv
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# 官方文档语料（测试集问题都出自这 4 篇；--corpus all 时索引全部 datas/md）
OFFICIAL_DOCS = ["operational_faq.md", "overview.md", "performance_faq.md", "product_faq.md"]

# 测试集：question 问题（英文，与语料一致）；keywords 标准答案必须包含的关键信息点（用于打分，脚本会归一大小写）
TEST_CASES = [
    {"id": 1, "question": "How much does Milvus cost?",
     "keywords": ["free", "open-source", "apache"]},
    {"id": 2, "question": "What is the maximum number of collections and partitions in a Milvus instance?",
     "keywords": ["65535"]},
    {"id": 3, "question": "What is the maximum amount of data per insert operation in Milvus?",
     "keywords": ["1024", "grpc"]},
    {"id": 4, "question": "What is the maximum vector dimension supported in Milvus by default?",
     "keywords": ["32768"]},
    {"id": 5, "question": "How to set nlist and nprobe for IVF indexes?",
     "keywords": ["sqrt", "nlist"]},
    {"id": 6, "question": "Does Milvus support non-x86 architectures?",
     "keywords": ["x86", "simd", "avx"]},
    {"id": 7, "question": "Where does Milvus store metadata?",
     "keywords": ["etcd"]},
    {"id": 8, "question": "What's the difference between a search and a query in Milvus?",
     "keywords": ["scalar", "boolean", "similarity"]},
    {"id": 9, "question": "Why do I get fewer than k vectors when searching for topk vectors?",
     "keywords": ["nprobe", "nlist", "cluster"]},
    {"id": 10, "question": "Does Milvus support inserting and searching data simultaneously?",
     "keywords": ["message queue", "query node"]},
]


def normalize(text: str) -> str:
    """小写 + 去掉千分位逗号，方便数字类关键词匹配（65,535 → 65535）。"""
    return (text or "").lower().replace(",", "")


def score_keywords(answer: str, keywords: list[str]) -> tuple[int, int]:
    """返回 (命中数, 总数)。"""
    a = normalize(answer)
    hits = sum(1 for kw in keywords if normalize(kw) in a)
    return hits, len(keywords)


def bigrams(s: str) -> set:
    s = normalize(s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


class LocalRetriever:
    """本地检索器：不依赖 Milvus，产出 langchain Document 列表，供 graph2 的 retrieve 节点调用。"""

    def __init__(self, docs: list, backend: str = "auto", top_k: int = 4):
        self.docs = docs
        self.backend = backend
        self.top_k = top_k
        self._emb = None
        self._matrix = None

    def _prepare_dense(self):
        from llm_models.embeddings_model import bge_embedding
        print(f"  正在用 BGE 对 {len(self.docs)} 个片段做向量化（首次运行需下载模型）...")
        self._matrix = np.array(bge_embedding.embed_documents([d.page_content for d in self.docs]))
        self._emb = bge_embedding

    def _dense_search(self, query: str):
        q = np.array(self._emb.embed_query(query))
        scores = q @ self._matrix.T  # BGE 已归一化，内积≈余弦相似度
        idx = np.argsort(-scores)[: self.top_k]
        return [self.docs[i] for i in idx]

    def _keyword_search(self, query: str):
        qb = bigrams(query)
        if not qb:
            return self.docs[: self.top_k]
        scored = [(len(qb & bigrams(d.page_content)) / len(qb), d) for d in self.docs]
        scored.sort(key=lambda x: -x[0])
        return [d for _, d in scored[: self.top_k]]

    def invoke(self, query: str):
        if self.backend in ("dense", "auto"):
            try:
                if self._matrix is None:
                    self._prepare_dense()
                return self._dense_search(query)
            except Exception as e:  # 模型下载失败 / 无网络等，回退到 keyword
                print(f"  [warn] 密集检索不可用（{e}），回退到 keyword 检索")
        return self._keyword_search(query)


def build_corpus(corpus: str):
    """解析语料并返回 content 片段列表（复用项目自己的解析器，只取 category=='content'，与检索 filter 一致）。"""
    from documents.markdown_parser import MarkdownParser

    md_dir = ROOT / "datas" / "md"
    if corpus == "all":
        files = sorted(md_dir.glob("*.md"))
    else:  # official
        files = [md_dir / f for f in OFFICIAL_DOCS]

    parser = MarkdownParser()
    docs = []
    for f in files:
        raw = parser.parse_markdown(str(f))          # 元素级解析
        merged = parser.merge_title_content(raw)     # 标题与正文合并
        for d in merged:
            if d.metadata.get("category") == "content":  # 只保留可检索的 content 片段
                docs.append(d)
    print(f"语料解析完成：{len(files)} 个文件 → {len(docs)} 个 content 片段")
    return docs


def ask_direct(question: str) -> str:
    """直接问大模型（无检索上下文），作为基线。"""
    from llm_models.all_llm import llm
    resp = llm.invoke(f"Answer the following question concisely and accurately. "
                     f"If you are not sure, say you don't know.\n{question}")
    return resp.content if hasattr(resp, "content") else str(resp)


def ask_rag(question: str, graph) -> str:
    """走本项目真实的 graph2 自适应 RAG 工作流。"""
    result = graph.invoke({"question": question})
    return result.get("generation", "") or ""


def main():
    ap = argparse.ArgumentParser(description="RAG 企业知识库评测")
    ap.add_argument("--retriever", choices=["auto", "dense", "keyword", "milvus"], default="auto")
    ap.add_argument("--corpus", choices=["official", "all"], default="official")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部）")
    ap.add_argument("--list", action="store_true", help="只打印测试集")
    args = ap.parse_args()

    if args.list:
        for c in TEST_CASES:
            print(f"#{c['id']:>2} {c['question']}  -> 关键词: {c['keywords']}")
        return

    cases = TEST_CASES[: args.limit] if args.limit > 0 else TEST_CASES

    # 1. 构建语料与检索器
    if args.retriever == "milvus":
        # 走项目真实的 Milvus 混合检索：不注入替身，让 graph2 用真实的 tools.retriever_tools
        print("使用真实 Milvus 混合检索（BGE 密集 + BM25 稀疏 + RRF）")
    else:
        docs = build_corpus(args.corpus)
        local_retriever = LocalRetriever(docs, backend=args.retriever, top_k=4)
        # 2. 注入假模块，让 graph2 的 retrieve 节点用本地检索器（绕过 Milvus 连接）
        fake = types.ModuleType("tools.retriever_tools")
        fake.retriever = local_retriever
        sys.modules["tools.retriever_tools"] = fake

    # 3. 加载真实的自适应 RAG 工作流（此刻才 import，确保上面的注入先生效）
    from graph2.graph_2 import graph
    print(f"评测模式：检索器={args.retriever} 语料={args.corpus} 题目数={len(cases)}\n")

    rows = []
    for c in cases:
        q = c["question"]
        t0 = time.time()
        try:
            rag_ans = ask_rag(q, graph)
        except Exception as e:
            rag_ans = f"<ERROR: {e}>"
        t_rag = round(time.time() - t0, 1)

        t0 = time.time()
        try:
            direct_ans = ask_direct(q)
        except Exception as e:
            direct_ans = f"<ERROR: {e}>"
        t_direct = round(time.time() - t0, 1)

        rag_hits, total = score_keywords(rag_ans, c["keywords"])
        direct_hits, _ = score_keywords(direct_ans, c["keywords"])

        rows.append({
            "id": c["id"], "question": q,
            "rag_hits": rag_hits, "direct_hits": direct_hits, "total": total,
            "rag_ans": rag_ans, "direct_ans": direct_ans,
            "t_rag": t_rag, "t_direct": t_direct,
        })
        print(f"#{c['id']:>2} 关键词命中 RAG={rag_hits}/{total}  直接={direct_hits}/{total}  "
              f"(RAG {t_rag}s / 直接 {t_direct}s) | {q}")

    # 4. 汇总（正确 = 命中率 >= 50%）
    def acc(key):
        ok = sum(1 for r in rows if r[key] / r["total"] >= 0.5)
        return ok, len(rows)

    rag_ok, n = acc("rag_hits")
    direct_ok, _ = acc("direct_hits")
    rag_acc = round(rag_ok / n * 100, 1)
    direct_acc = round(direct_ok / n * 100, 1)

    print("\n========== 评测结果汇总 ==========")
    print(f"直接问大模型正确率：{direct_ok}/{n} = {direct_acc}%")
    print(f"RAG 工作流正确率：  {rag_ok}/{n} = {rag_acc}%")
    print(f"提升：{round(rag_acc - direct_acc, 1)} 个百分点")

    # 5. 导出
    csv_path = ROOT / "eval_rag_result.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["id", "question", "rag_hits", "direct_hits", "total", "rag_ans", "direct_ans"])
        for r in rows:
            w.writerow([r["id"], r["question"], r["rag_hits"], r["direct_hits"], r["total"],
                        r["rag_ans"], r["direct_ans"]])
    print(f"\n明细已保存到：{csv_path}")


if __name__ == "__main__":
    main()
