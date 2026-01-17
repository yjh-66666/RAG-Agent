"""企业级 RAG + Agent + API（最终体：单文件版 最终版3.py）

面向“中厂 AI 大模型应用开发实习”的可展示版本：
- RAG：文档摄入、权限/部门过滤、知识新鲜度重排、问答生成、审计日志
- Agent：工具路由（search/doc_info/rag_answer）、Guardrails（提示注入/敏感信息）、权限不绕过
- API：FastAPI + /docs + /metrics（Prometheus）

快速开始：
1) 安装依赖：pip install -r requirements_v2.txt
2) 配置 .env：
   OPENAI_API_KEY=sk-...
3) 准备文档：把 pdf/docx/txt 放到 ./demo_docs 或用 /ingest 指定路径
4) 启动服务：python zhi_liao_knowledge_assistant/最终版3.py
5) 打开接口文档：http://127.0.0.1:8000/docs

提示：
- 默认 LLM: OpenAI（面试演示最稳）
- 可切换 HuggingFace：在 .env 里设置 LLM_PROVIDER=huggingface，并按需安装 transformers/accelerate/torch
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import prometheus_client as prom
from langchain.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from pydantic import BaseModel, Field, BaseSettings

# JWT：建议安装 PyJWT（pip install pyjwt），这里不做“手写回退”，避免产生不严谨的安全误解。
try:
    import jwt as pyjwt  # type: ignore
except Exception:  # pragma: no cover
    pyjwt = None


# =============================================================================
# 日志与 Prometheus 指标
# =============================================================================
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s")

REQUEST_LATENCY = prom.Histogram(
    "rag_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint", "handler"],
)
REQUEST_COUNTER = prom.Counter(
    "rag_requests_total",
    "Total number of requests",
    ["endpoint", "handler", "status"],
)


def monitor_requests(endpoint: str):
    """装饰器：监控函数调用延迟与成功/失败计数。"""

    def decorator(func: callable) -> callable:
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = time.time()
            status = "success"
            try:
                return func(*args, **kwargs)
            except Exception:
                status = "error"
                raise
            finally:
                REQUEST_LATENCY.labels(endpoint=endpoint, handler=func.__name__).observe(time.time() - start)
                REQUEST_COUNTER.labels(endpoint=endpoint, handler=func.__name__, status=status).inc()

        return wrapper

    return decorator


# =============================================================================
# 通用返回结构（API/Agent/内部方法统一）
# =============================================================================
class ApiResponse(BaseModel):
    request_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


def new_request_id() -> str:
    return str(uuid.uuid4())


def ok(data: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
    rid = request_id or new_request_id()
    return ApiResponse(request_id=rid, success=True, data=data).dict()


def fail(code: str, message: str, request_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rid = request_id or new_request_id()
    return ApiResponse(
        request_id=rid,
        success=False,
        error={"code": code, "message": message, "details": details or {}},
    ).dict()


# =============================================================================
# 异常
# =============================================================================
class RAGError(Exception):
    """RAG 系统基础异常。"""


class DocumentProcessingError(RAGError):
    """文档处理异常。"""


class VectorStoreError(RAGError):
    """向量库异常。"""


class GuardrailViolation(RAGError):
    """安全策略/护栏触发。"""


# =============================================================================
# 配置与数据模型
# =============================================================================
class RAGConfig(BaseSettings):
    # ----------------
    # Auth
    # ----------------
    auth_mode: str = "jwt"  # jwt | api_key

    # JWT
    jwt_secret: str = "dev_jwt_secret_change_me"
    jwt_algorithm: str = "HS256"
    jwt_exp_minutes: int = 60 * 24  # 默认 1 天，面试演示足够

    # API Key 回退：格式 "alice:hr|public,bob:public,admin:admin"
    api_keys: str = ""

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    vectorstore_type: str = "chroma"  # chroma | faiss
    vectorstore_path: str = "./chroma_db"
    metadata_db_path: str = "./metadata_db"

    # Embedding/LLM
    use_openai: bool = True
    openai_api_key: Optional[str] = None
    openai_chat_model: str = "gpt-4o-mini"

    llm_provider: str = "openai"  # openai | huggingface
    hf_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hf_device: str = "cpu"  # cpu | cuda | mps

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Concurrency
    max_workers: int = 4

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class DocumentSource(BaseModel):
    file_path: str
    file_hash: str
    file_size: int
    last_modified: datetime
    uploaded_by: str = "system"
    upload_time: datetime = Field(default_factory=datetime.now)
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 1
    department: Optional[str] = None
    access_control: List[str] = Field(default_factory=list)
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditLog(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    user: str
    action: str
    document_id: str
    details: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# DocumentProcessor
# =============================================================================
class DocumentProcessor:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.SUPPORTED_EXTS = (".pdf", ".docx", ".txt")

    def _calculate_file_hash(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    def _enrich_document_metadata(self, document: Document, source_info: DocumentSource) -> None:
        document.metadata.update(
            {
                "document_id": source_info.document_id,
                "source_file": source_info.file_path,
                "file_hash": source_info.file_hash,
                "uploaded_by": source_info.uploaded_by,
                "upload_time": source_info.upload_time.isoformat(),
                "last_modified": source_info.last_modified.isoformat(),
                "version": source_info.version,
                "department": source_info.department,
                "access_control": ",".join(source_info.access_control) if source_info.access_control else "public",
                **source_info.custom_metadata,
            }
        )

    def _process_single_document(self, file_path: Path, metadata_args: Dict[str, Any]) -> Tuple[List[Document], DocumentSource]:
        if not file_path.exists():
            raise DocumentProcessingError(f"文件不存在: {file_path}")

        file_hash = self._calculate_file_hash(file_path)
        last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)

        try:
            uploaded_by = os.getlogin()
        except Exception:
            uploaded_by = os.environ.get("USER") or os.environ.get("USERNAME") or "system"

        source_info = DocumentSource(
            file_path=str(file_path.absolute()),
            file_hash=file_hash,
            file_size=file_path.stat().st_size,
            last_modified=last_modified,
            uploaded_by=uploaded_by,
            **metadata_args,
        )

        loader_map = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt": lambda p: TextLoader(p, encoding="utf-8"),
        }
        suffix = file_path.suffix.lower()
        if suffix not in loader_map:
            raise DocumentProcessingError(f"不支持的文件类型: {suffix}")

        loader = loader_map[suffix](str(file_path))
        documents = loader.load()
        for doc in documents:
            self._enrich_document_metadata(doc, source_info)

        logger.info(f"✓ 成功加载: {file_path.name} ({len(documents)} 个块)")
        return documents, source_info

    def _expand_paths(self, paths: Iterable[Union[str, Path]]) -> List[Path]:
        expanded: List[Path] = []
        for p in paths:
            p = Path(p)
            if p.is_dir():
                expanded.extend(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTS)
            elif p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS:
                expanded.append(p)
        return expanded

    @monitor_requests("document_processing")
    def load_and_process_documents(self, paths: Iterable[Union[str, Path]], **metadata_kwargs: Any) -> Tuple[List[Document], Dict[str, DocumentSource]]:
        all_documents: List[Document] = []
        all_sources: Dict[str, DocumentSource] = {}

        expanded_files = self._expand_paths(paths)
        if not expanded_files:
            logger.warning("在指定路径下未找到任何支持的文件。")
            return [], {}

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_path = {executor.submit(self._process_single_document, fp, metadata_kwargs): fp for fp in expanded_files}
            for future, fp in future_to_path.items():
                try:
                    docs, source = future.result()
                    all_documents.extend(docs)
                    all_sources[source.document_id] = source
                except Exception as e:
                    logger.error(f"✗ 处理文件失败 {fp}: {e}")

        return all_documents, all_sources

    def split_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return splitter.split_documents(documents)


# =============================================================================
# VectorStoreManager
# =============================================================================
class VectorStoreManager:
    def __init__(self, config: RAGConfig, embeddings: Any):
        self.config = config
        self.embeddings = embeddings
        self.vectorstore: Optional[Union[Chroma, FAISS]] = None
        os.makedirs(self.config.vectorstore_path, exist_ok=True)

    def build_or_load(self, documents: Optional[List[Document]] = None) -> None:
        try:
            if self.config.vectorstore_type == "chroma":
                if documents:
                    self.vectorstore = Chroma.from_documents(documents, self.embeddings, persist_directory=self.config.vectorstore_path)
                else:
                    self.vectorstore = Chroma(persist_directory=self.config.vectorstore_path, embedding_function=self.embeddings)
            elif self.config.vectorstore_type == "faiss":
                if documents:
                    self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                    self.vectorstore.save_local(self.config.vectorstore_path)
                else:
                    self.vectorstore = FAISS.load_local(self.config.vectorstore_path, self.embeddings)
            else:
                raise VectorStoreError(f"不支持的向量数据库类型: {self.config.vectorstore_type}")

            logger.info(f"✓ 向量数据库已就绪 (类型: {self.config.vectorstore_type})")
        except Exception as e:
            raise VectorStoreError(f"向量数据库初始化失败: {e}")

    def add_documents(self, documents: List[Document]) -> None:
        if not self.vectorstore:
            raise VectorStoreError("向量库未初始化，无法添加文档。")
        self.vectorstore.add_documents(documents)
        if isinstance(self.vectorstore, Chroma):
            self.vectorstore.persist()
        else:
            self.vectorstore.save_local(self.config.vectorstore_path)

    def similarity_search_with_score(self, query: str, k: int, **kwargs: Any) -> List[Tuple[Document, float]]:
        if not self.vectorstore:
            raise VectorStoreError("向量库未初始化，无法执行搜索。")
        return self.vectorstore.similarity_search_with_score(query, k=k, **kwargs)


# =============================================================================
# EnterpriseRAGSystem
# =============================================================================
class EnterpriseRAGSystem:
    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        self.config = config or RAGConfig()
        self.doc_processor = DocumentProcessor(self.config)
        self.embeddings = self._init_embeddings()
        self.vector_store = VectorStoreManager(self.config, self.embeddings)

        self.llm: Optional[Any] = None

        self.current_user: str = "system"
        self.metadata_db: Dict[str, DocumentSource] = {}
        os.makedirs(self.config.metadata_db_path, exist_ok=True)
        self._load_metadata_db()

    def _init_embeddings(self) -> Any:
        if self.config.use_openai and self.config.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.config.openai_api_key
            logger.info("使用 OpenAI Embeddings: text-embedding-ada-002")
            return OpenAIEmbeddings(model="text-embedding-ada-002")

        logger.info(f"使用 HuggingFace Embeddings: {self.config.embedding_model}")
        return HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _load_metadata_db(self) -> None:
        fpath = Path(self.config.metadata_db_path) / "metadata.json"
        if not fpath.exists():
            return
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for doc_id, doc_data in data.items():
            self.metadata_db[doc_id] = DocumentSource(**doc_data)

    def _save_metadata_db(self) -> None:
        fpath = Path(self.config.metadata_db_path) / "metadata.json"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump({k: v.dict() for k, v in self.metadata_db.items()}, f, default=str, indent=2, ensure_ascii=False)

    def _log_audit(self, action: str, document_id: str, details: Optional[Dict[str, Any]] = None) -> None:
        entry = AuditLog(user=self.current_user, action=action, document_id=document_id, details=details or {})
        log_file = Path(self.config.metadata_db_path) / "audit_logs.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry.json(ensure_ascii=False) + "\n")

    def set_current_user(self, user: str) -> None:
        self.current_user = user

    @monitor_requests("ingest")
    def ingest_documents(self, paths: Iterable[Union[str, Path]], **metadata_kwargs: Any) -> Dict[str, Any]:
        self._log_audit("ingestion_started", "system", {"paths": [str(p) for p in paths]})

        documents, sources = self.doc_processor.load_and_process_documents(paths, **metadata_kwargs)
        if not documents:
            return {"status": "no_documents"}

        chunks = self.doc_processor.split_documents(documents)
        self.vector_store.build_or_load(chunks)

        self.metadata_db.update(sources)
        self._save_metadata_db()
        self._log_audit("ingestion_completed", "system", {"doc_ids": list(sources.keys()), "chunk_count": len(chunks)})

        return {"status": "ok", "documents": len(sources), "chunks": len(chunks)}

    def _calculate_freshness_score(self, doc: Document) -> float:
        doc_type = (doc.metadata.get("doc_type") or doc.metadata.get("document_type") or "technical").lower()
        last_modified_str = doc.metadata.get("last_modified") or doc.metadata.get("upload_time")
        try:
            last_modified = datetime.fromisoformat(str(last_modified_str))
        except Exception:
            return 0.5

        delta_days = (datetime.now() - last_modified).days
        full, expire = (90, 180) if doc_type in ["policy", "政策"] else (365, 730)

        if delta_days <= full:
            return 1.0
        if delta_days >= expire:
            return 0.0
        return max(0.0, 1 - (delta_days - full) / (expire - full))

    def _blend_scores(self, similarity_score: float, freshness_score: float, weight: float) -> float:
        return similarity_score * (1 - weight * freshness_score)

    @monitor_requests("search")
    def search_documents(
        self,
        query: str,
        k: int = 5,
        department: Optional[str] = None,
        user_roles: Optional[List[str]] = None,
        freshness_weight: float = 0.3,
    ) -> Tuple[List[Document], List[float]]:
        if not self.vector_store.vectorstore:
            raise VectorStoreError("向量库未初始化，请先 ingest。")

        user_roles = user_roles or ["public"]
        raw = self.vector_store.similarity_search_with_score(query, k=k * 3)

        processed: List[Tuple[Document, float, float]] = []
        for doc, score in raw:
            doc_access = str(doc.metadata.get("access_control", "public")).split(",")
            if not ("public" in doc_access or any(r in doc_access for r in user_roles)):
                continue
            if department and doc.metadata.get("department") != department:
                continue

            freshness = self._calculate_freshness_score(doc)
            doc.metadata["freshness_score"] = freshness
            processed.append((doc, score, freshness))

        if freshness_weight > 0:
            processed.sort(key=lambda x: self._blend_scores(x[1], x[2], freshness_weight))
        else:
            processed.sort(key=lambda x: x[1])

        final_docs = [d for d, _, _ in processed[:k]]
        final_scores = [s for _, s, _ in processed[:k]]

        self._log_audit(
            "search_documents",
            "search_operation",
            {
                "query": query,
                "k": k,
                "department": department,
                "user_roles": user_roles,
                "freshness_weight": freshness_weight,
                "results_count": len(final_docs),
            },
        )

        return final_docs, final_scores

    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        src = self.metadata_db.get(document_id)
        return src.dict() if src else None

    # ----------------
    # LLM
    # ----------------
    def init_llm(self, force_reload: bool = False) -> None:
        if self.llm and not force_reload:
            return

        provider = self.config.llm_provider.lower()
        if provider == "openai":
            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key 未配置，请设置 OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = self.config.openai_api_key

            from langchain.chat_models import ChatOpenAI

            self.llm = ChatOpenAI(model_name=self.config.openai_chat_model, temperature=0.1)
            return

        if provider == "huggingface":
            try:
                from langchain.llms import HuggingFacePipeline
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            except ImportError:
                raise ImportError("需要安装 transformers/accelerate/torch")

            tok = AutoTokenizer.from_pretrained(self.config.hf_model_name)
            model = AutoModelForCausalLM.from_pretrained(self.config.hf_model_name)
            device = -1 if self.config.hf_device == "cpu" else 0
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tok,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                device=device,
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
            return

        raise ValueError(f"不支持的 llm_provider: {provider}")

    def rag_answer(self, question: str, docs: List[Document]) -> str:
        """给定 docs（必须已过滤权限），生成最终答案；输出带来源编号。"""
        self.init_llm()

        context = "\n\n".join(
            f"[{i+1}] {d.page_content}\n(source_file={d.metadata.get('source_file')}, document_id={d.metadata.get('document_id')})"
            for i, d in enumerate(docs)
        )

        prompt = (
            "你是一位专业的企业知识库助手。\n"
            "请仅基于上下文回答，不要编造。\n"
            "如果信息不足，请回答：根据现有知识库信息，无法回答该问题。\n"
            "请在回答中使用引用标记，例如：[1][2]。\n\n"
            f"上下文：\n{context}\n\n"
            f"问题：{question}\n\n"
            "回答："
        )

        if hasattr(self.llm, "invoke"):
            resp = self.llm.invoke(prompt)
            return getattr(resp, "content", None) or str(resp)
        return str(self.llm(prompt))


# =============================================================================
# Agent
# =============================================================================
class AgentRequest(BaseModel):
    request_id: Optional[str] = None
    department: Optional[str] = None
    message: str


class RAGAgent:
    """面试向 Agent：工具路由 + 护栏 + 审计。"""

    INJECTION_PATTERNS = [
        r"忽略(以上|之前)?(所有)?(指令|规则)",
        r"system prompt",
        r"developer message",
        r"泄露.*(密钥|key|api)",
        r"输出.*OPENAI_API_KEY",
    ]

    SENSITIVE_PATTERNS = [
        r"sk-[A-Za-z0-9]{10,}",
        r"AKIA[0-9A-Z]{16}",
    ]

    def __init__(self, rag: EnterpriseRAGSystem):
        self.rag = rag

    def _check_guardrails(self, text: str) -> None:
        for pat in self.INJECTION_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                raise GuardrailViolation("检测到疑似提示注入/越权请求，已拒绝执行。")
        for pat in self.SENSITIVE_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                raise GuardrailViolation("检测到疑似敏感信息泄露风险，已拒绝执行。")

    def _route_intent(self, message: str) -> str:
        m = message.lower()
        if any(k in m for k in ["谁上传", "上传者", "哈希", "file_hash", "文件路径", "document_id", "版本", "更新时间", "last_modified"]):
            return "doc_info"
        if any(k in m for k in ["列出", "搜索", "找一下", "有哪些资料", "相关文档"]):
            return "search"
        return "rag_answer"

    @monitor_requests("agent")
    def chat(self, req: AgentRequest) -> Dict[str, Any]:
        rid = req.request_id or new_request_id()
        # req.user 由 API 层注入（JWT/身份系统），避免客户端伪造
        self.rag.set_current_user(getattr(req, "user", "anonymous"))

        try:
            self._check_guardrails(req.message)
        except GuardrailViolation as e:
            self.rag._log_audit("agent_blocked", "agent", {"request_id": rid, "message": req.message, "reason": str(e)})
            return fail("GUARDRAIL", str(e), request_id=rid)

        intent = self._route_intent(req.message)

        # 统一：任何工具调用都必须携带 department/user_roles（权限不绕过）
        if intent == "search":
            docs, scores = self.rag.search_documents(req.message, k=5, department=req.department, user_roles=getattr(req, "user_roles", ["public"]))
            results = []
            for d, s in zip(docs, scores):
                results.append(
                    {
                        "document_id": d.metadata.get("document_id"),
                        "source_file": d.metadata.get("source_file"),
                        "preview": d.page_content[:200],
                        "similarity_score": float(s),
                        "freshness_score": d.metadata.get("freshness_score"),
                    }
                )
            self.rag._log_audit("agent_search", "agent", {"request_id": rid, "count": len(results)})
            return ok({"action": "search", "results": results}, request_id=rid)

        if intent == "doc_info":
            m = re.search(r"([0-9a-fA-F-]{32,36})", req.message)
            if not m:
                return ok({"action": "doc_info", "answer": "请提供 document_id（UUID）后再查询元信息。"}, request_id=rid)
            info = self.rag.get_document_info(m.group(1))
            if not info:
                return ok({"action": "doc_info", "answer": "未找到该 document_id 对应的文档记录。"}, request_id=rid)
            self.rag._log_audit("agent_doc_info", "agent", {"request_id": rid, "document_id": m.group(1)})
            return ok({"action": "doc_info", "document": info}, request_id=rid)

        # rag_answer
        docs, _ = self.rag.search_documents(req.message, k=5, department=req.department, user_roles=getattr(req, "user_roles", ["public"]))
        if not docs:
            return ok({"action": "rag_answer", "answer": "根据您的权限与知识库内容，无法回答该问题。", "sources": []}, request_id=rid)

        answer = self.rag.rag_answer(req.message, docs)
        sources = [
            {
                "ref": i + 1,
                "document_id": d.metadata.get("document_id"),
                "source_file": d.metadata.get("source_file"),
                "last_modified": d.metadata.get("last_modified"),
                "freshness_score": d.metadata.get("freshness_score"),
            }
            for i, d in enumerate(docs)
        ]
        self.rag._log_audit("agent_rag_answer", "agent", {"request_id": rid, "sources": [s["document_id"] for s in sources]})
        return ok({"action": "rag_answer", "answer": answer, "sources": sources}, request_id=rid)


# =============================================================================
# FastAPI
# =============================================================================
try:
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import PlainTextResponse
except Exception:  # pragma: no cover
    FastAPI = None


# -----------------
# Auth helpers
# -----------------
class Identity(BaseModel):
    user: str
    roles: List[str] = Field(default_factory=lambda: ["public"])
    department: Optional[str] = None


def _parse_api_keys(api_keys: str) -> Dict[str, Identity]:
    """将 "alice:hr|public,bob:public" 解析为 Identity 映射。

    约定：X-API-Key 直接传用户名（演示用），真实生产可换成随机 key。
    """

    out: Dict[str, Identity] = {}
    if not api_keys:
        return out
    for item in api_keys.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            continue
        user, roles_str = item.split(":", 1)
        roles = [r.strip() for r in roles_str.split("|") if r.strip()]
        out[user.strip()] = Identity(user=user.strip(), roles=roles or ["public"])
    return out


def _jwt_encode(payload: Dict[str, Any], secret: str, algorithm: str) -> str:
    if pyjwt is None:
        raise ImportError("请先安装 PyJWT：pip install pyjwt")
    return pyjwt.encode(payload, secret, algorithm=algorithm)  # type: ignore


def _jwt_decode(token: str, secret: str, algorithms: List[str]) -> Dict[str, Any]:
    if pyjwt is None:
        raise ImportError("请先安装 PyJWT：pip install pyjwt")
    return pyjwt.decode(token, secret, algorithms=algorithms)  # type: ignore


def get_identity(
    cfg: RAGConfig,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Identity:
    """统一鉴权入口：默认 JWT（Bearer），可回退 API Key。

    注意：JWT 功能依赖 PyJWT；若未安装会返回明确错误提示。

    - JWT: Authorization: Bearer <token>
    - API Key: X-API-Key: <user>（演示用）
    """

    mode = (cfg.auth_mode or "jwt").lower()

    if mode == "jwt":
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <token>")
        token = authorization.split(" ", 1)[1].strip()
        try:
            payload = _jwt_decode(token, cfg.jwt_secret, algorithms=[cfg.jwt_algorithm])
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

        # exp 校验（JWT 标准是秒级时间戳）
        exp = payload.get("exp")
        if exp is not None:
            try:
                exp_ts = int(exp)
            except Exception:
                raise HTTPException(status_code=401, detail="Invalid exp in token")
            now_ts = int(datetime.now(tz=timezone.utc).timestamp())
            if now_ts >= exp_ts:
                raise HTTPException(status_code=401, detail="Token expired")

        user = payload.get("sub") or payload.get("user")
        roles = payload.get("roles") or ["public"]
        department = payload.get("department")

        if not user:
            raise HTTPException(status_code=401, detail="Token missing sub/user")
        if isinstance(roles, str):
            roles = [roles]
        if not isinstance(roles, list):
            roles = ["public"]

        return Identity(user=str(user), roles=[str(r) for r in roles], department=department)

    if mode == "api_key":
        mapping = _parse_api_keys(cfg.api_keys)
        if not x_api_key:
            raise HTTPException(status_code=401, detail="Missing X-API-Key")
        ident = mapping.get(x_api_key)
        if not ident:
            raise HTTPException(status_code=403, detail="Invalid API key")
        return ident

    raise HTTPException(status_code=500, detail=f"Unsupported AUTH_MODE: {mode}")


class IngestRequest(BaseModel):
    request_id: Optional[str] = None
    paths: List[str]
    department: Optional[str] = None
    access_control: List[str] = Field(default_factory=lambda: ["public"])
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    request_id: Optional[str] = None
    query: str
    k: int = 5
    department: Optional[str] = None
    freshness_weight: float = 0.3


class QueryRequest(BaseModel):
    request_id: Optional[str] = None
    question: str
    department: Optional[str] = None


def create_app() -> Any:
    if FastAPI is None:
        raise ImportError("未安装 fastapi。请运行：pip install fastapi uvicorn")

    cfg = RAGConfig()
    # env 兼容：允许使用 LLM_PROVIDER 覆盖
    env_llm_provider = os.environ.get("LLM_PROVIDER")
    if env_llm_provider:
        cfg.llm_provider = env_llm_provider

    rag = EnterpriseRAGSystem(cfg)
    agent = RAGAgent(rag)

    app = FastAPI(title="Enterprise RAG + Agent", version="1.0.0")

    # 统一身份依赖（默认 jwt）
    def _dep_identity(
        authorization: Optional[str] = Header(default=None, alias="Authorization"),
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> Identity:
        return get_identity(cfg, authorization=authorization, x_api_key=x_api_key)

    # ----------
    # Auth: dev_login（仅演示用）
    # ----------
    class DevLoginRequest(BaseModel):
        user: str
        roles: List[str] = Field(default_factory=lambda: ["public"])
        department: Optional[str] = None
        exp_minutes: Optional[int] = None

    @app.post("/auth/dev_login")
    def dev_login(req: DevLoginRequest):
        """开发/面试演示用：签发 JWT。

        生产环境应对接公司 SSO / OAuth2 / 统一网关。
        """
        minutes = int(req.exp_minutes or cfg.jwt_exp_minutes)
        exp_ts = int((datetime.now(tz=timezone.utc) + timedelta(minutes=minutes)).timestamp())
        payload = {
            "sub": req.user,
            "roles": req.roles,
            "department": req.department,
            "exp": exp_ts,
            "iat": int(datetime.now(tz=timezone.utc).timestamp()),
        }
        try:
            token = _jwt_encode(payload, cfg.jwt_secret, cfg.jwt_algorithm)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Token sign failed: {e}")
        return ok({"access_token": token, "token_type": "bearer", "expires_at": exp_ts})

    @app.get("/health")
    def health():
        return ok({"status": "ok"})

    @app.get("/metrics")
    def metrics():
        return PlainTextResponse(prom.generate_latest().decode("utf-8"))

    @app.post("/ingest")
    def ingest(req: IngestRequest, ident: Identity = Depends(_dep_identity)):
        rid = req.request_id or new_request_id()
        try:
            rag.set_current_user(ident.user)
            out = rag.ingest_documents(
                req.paths,
                department=req.department or ident.department,
                access_control=req.access_control,
                custom_metadata=req.custom_metadata,
            )
            return ok({"ingest": out}, request_id=rid)
        except Exception as e:
            return fail("INGEST_ERROR", str(e), request_id=rid)

    @app.post("/search")
    def search(req: SearchRequest, ident: Identity = Depends(_dep_identity)):
        rid = req.request_id or new_request_id()
        try:
            rag.set_current_user(ident.user)
            docs, scores = rag.search_documents(
                req.query,
                k=req.k,
                department=req.department or ident.department,
                user_roles=ident.roles,
                freshness_weight=req.freshness_weight,
            )
            results = [
                {
                    "ref": i + 1,
                    "content": d.page_content,
                    "metadata": d.metadata,
                    "similarity_score": float(s),
                }
                for i, (d, s) in enumerate(zip(docs, scores))
            ]
            return ok({"query": req.query, "results": results}, request_id=rid)
        except Exception as e:
            return fail("SEARCH_ERROR", str(e), request_id=rid)

    @app.post("/query")
    def query(req: QueryRequest, ident: Identity = Depends(_dep_identity)):
        rid = req.request_id or new_request_id()
        try:
            rag.set_current_user(ident.user)
            docs, _ = rag.search_documents(req.question, k=5, department=req.department or ident.department, user_roles=ident.roles)
            if not docs:
                return ok({"question": req.question, "answer": "根据您的权限与知识库内容，无法回答该问题。", "sources": []}, request_id=rid)
            answer = rag.rag_answer(req.question, docs)
            sources = [
                {
                    "ref": i + 1,
                    "document_id": d.metadata.get("document_id"),
                    "source_file": d.metadata.get("source_file"),
                }
                for i, d in enumerate(docs)
            ]
            return ok({"question": req.question, "answer": answer, "sources": sources}, request_id=rid)
        except Exception as e:
            return fail("QUERY_ERROR", str(e), request_id=rid)

    @app.post("/agent_chat")
    def agent_chat(req: AgentRequest, ident: Identity = Depends(_dep_identity)):
        # Agent 内部统一 ok/fail，但身份必须来自服务端（不可从 body 伪造）
        req.user = ident.user
        req.user_roles = ident.roles
        if ident.department is not None:
            req.department = req.department or ident.department
        return agent.chat(req)

    return app


if __name__ == "__main__":
    # 启动：python zhi_liao_knowledge_assistant/最终版3.py
    # 文档：curl -X POST http://127.0.0.1:8000/ingest ... 或直接用 /docs
    try:
        import uvicorn

        uvicorn.run(create_app(), host="127.0.0.1", port=8000, log_level="info")
    except Exception as e:
        logger.error(f"启动失败: {e}")
        logger.error("请确认已安装 fastapi 与 uvicorn：pip install fastapi uvicorn")
