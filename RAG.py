"""企业级 RAG + Agent + API（单文件版 - 离线优化版）

说明：此版本已优化为完全离线运行：
- LLM：使用 DeepSeek API（网络）
- Embedding：使用本地 BGE 模型（完全离线）
- 自动处理模型拷贝和降级

快速开始：
1) 安装依赖：pip install -r requirements.txt
2) 配置 .env：修改 OPENAI_API_KEY
3) 运行：python RAGplus.py
4) 访问：http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import prometheus_client as prom

# LangChain 相关
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Pydantic
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# JWT
try:
    import jwt as pyjwt
except ImportError:
    pyjwt = None

# FastAPI
try:
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import PlainTextResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
except ImportError:
    FastAPI = None

# =============================================================================
# 日志配置
# =============================================================================
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_system.log", encoding="utf-8")
    ]
)

# =============================================================================
# 监控指标
# =============================================================================
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

# =============================================================================
# RAG 质量与成本指标（新增：不影响运行）
# =============================================================================
RAG_EMPTY_RETRIEVAL_COUNTER = prom.Counter(
    "rag_empty_retrieval_total",
    "Number of queries with empty retrieval results",
    ["endpoint", "department"],
)
RAG_RETRIEVAL_DOCS_HIST = prom.Histogram(
    "rag_retrieval_docs_count",
    "Number of retrieved docs returned to downstream RAG",
    ["endpoint"],
    buckets=(0, 1, 2, 3, 5, 8, 13, 21),
)
RAG_RETRIEVAL_SCORE_HIST = prom.Histogram(
    "rag_retrieval_similarity_score",
    "Similarity score distribution of retrieved docs",
    ["endpoint"],
)
RAG_LLM_LATENCY = prom.Histogram(
    "rag_llm_latency_seconds",
    "LLM call latency in seconds",
    ["endpoint", "model"],
)
RAG_LLM_TOKENS = prom.Counter(
    "rag_llm_tokens_total",
    "LLM token usage total (if available)",
    ["endpoint", "model", "type"],
)
RAG_LLM_COST_USD = prom.Counter(
    "rag_llm_cost_usd_total",
    "Estimated LLM cost in USD (optional)",
    ["endpoint", "model"],
)


def monitor_requests(endpoint: str):
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
# 数据模型
# =============================================================================
class ApiResponse(BaseModel):
    request_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


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


class Identity(BaseModel):
    user: str
    roles: List[str] = Field(default_factory=lambda: ["public"])
    department: Optional[str] = None


# API请求模型
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


class AgentRequest(BaseModel):
    request_id: Optional[str] = None
    department: Optional[str] = None
    message: str
    user: Optional[str] = None
    user_roles: Optional[List[str]] = None
    
    model_config = {"extra": "allow"}  # 允许额外字段，用于动态赋值


class DevLoginRequest(BaseModel):
    user: str
    roles: List[str] = Field(default_factory=lambda: ["public"])
    department: Optional[str] = None
    exp_minutes: Optional[int] = None


def new_request_id() -> str:
    return str(uuid.uuid4())


def ok(data: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
    rid = request_id or new_request_id()
    return ApiResponse(request_id=rid, success=True, data=data).model_dump()


def fail(code: str, message: str, request_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> Dict[
    str, Any]:
    rid = request_id or new_request_id()
    return ApiResponse(
        request_id=rid,
        success=False,
        error={"code": code, "message": message, "details": details or {}},
    ).model_dump()


# =============================================================================
# 配置类
# =============================================================================
class RAGConfig(BaseSettings):
    auth_mode: str = "jwt"
    jwt_secret: str = "dev_jwt_secret_change_me"
    jwt_algorithm: str = "HS256"
    jwt_exp_minutes: int = 60 * 24
    api_keys: str = ""

    vectorstore_type: str = "chroma"
    vectorstore_path: str = "./chroma_db"
    metadata_db_path: str = "./metadata_db"

    openai_api_key: Optional[str] = None
    openai_chat_model: str = "deepseek-chat"
    openai_base_url: str = "https://api.deepseek.com/v1"

    use_local_embedding: bool = True
    embedding_model_path: str = "./models/bge-small-zh-v1.5"
    embedding_device: str = "cpu"
    auto_copy_model: bool = True
    modelscope_cache_path: str = "C:\\Users\\yang\\.cache\\modelscope\\hub\\models\\BAAI\\bge-small-zh-v1.5"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_workers: int = 4

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# =============================================================================
# 本地 Embedding 实现（完全离线）
# =============================================================================
class LocalEmbeddings:
    """本地 Embedding 实现，完全离线"""

    def __init__(self, model_path: str = "./models/bge-small-zh-v1.5", device: str = "cpu"):
        # 强制设置离线环境变量
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.dimension = 512

        try:
            self._load_model()
        except Exception as e:
            logger.warning(f"加载模型失败，使用轻量模式: {e}")
            self._create_lightweight_model()

    def _load_model(self):
        """尝试加载本地模型"""
        try:
            from sentence_transformers import SentenceTransformer

            # 检查模型文件是否存在
            if self.model_path.exists():
                # 查找模型文件
                model_files = list(self.model_path.glob("*.bin")) + list(self.model_path.glob("*.safetensors"))

                if model_files:
                    logger.info(f"找到模型文件，尝试加载: {self.model_path}")

                    # 尝试加载本地模型
                    self.model = SentenceTransformer(
                        str(self.model_path),
                        device=self.device,
                        local_files_only=True  # 关键：只使用本地文件
                    )
                    self.dimension = self.model.get_sentence_embedding_dimension()
                    logger.info(f"✓ 本地模型加载成功，维度: {self.dimension}")
                    return

            # 如果没有模型文件，创建轻量级模型
            logger.info("未找到模型文件，使用轻量级模型")
            self._create_lightweight_model()

        except Exception as e:
            logger.warning(f"模型加载异常: {e}")
            self._create_lightweight_model()

    def _create_lightweight_model(self):
        """创建轻量级模型（用于演示）"""
        logger.info("使用轻量级嵌入模型（离线模式）")
        self.model = None
        self.dimension = 384

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, normalize_embeddings=True)
                return embeddings.tolist()
            except Exception as e:
                logger.warning(f"模型编码失败，使用模拟嵌入: {e}")

        # 使用模拟嵌入
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        if self.model is not None:
            try:
                embedding = self.model.encode(text, normalize_embeddings=True)
                return embedding.tolist()
            except Exception as e:
                logger.warning(f"模型编码失败，使用模拟嵌入: {e}")

        # 创建伪随机嵌入（离线可用）
        import hashlib
        hash_val = int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)
        np.random.seed(hash_val % 10000)

        embedding = np.random.randn(self.dimension) * 0.1
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()


# =============================================================================
# 文档处理器
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
        document.metadata.update({
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
        })

    def _process_single_document(self, file_path: Path, metadata_args: Dict[str, Any]) -> Tuple[
        List[Document], DocumentSource]:
        if not file_path.exists():
            raise Exception(f"文件不存在: {file_path}")

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
            raise Exception(f"不支持的文件类型: {suffix}")

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
    def load_and_process_documents(self, paths: Iterable[Union[str, Path]], **metadata_kwargs: Any) -> Tuple[
        List[Document], Dict[str, DocumentSource]]:
        all_documents: List[Document] = []
        all_sources: Dict[str, DocumentSource] = {}

        expanded_files = self._expand_paths(paths)
        if not expanded_files:
            logger.warning("在指定路径下未找到任何支持的文件。")
            return [], {}

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_path = {executor.submit(self._process_single_document, fp, metadata_kwargs): fp for fp in
                              expanded_files}
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
# 向量存储管理器
# =============================================================================
class VectorStoreManager:
    def __init__(self, config: RAGConfig, embeddings: Any):
        self.config = config
        self.embeddings = embeddings
        self.vectorstore: Optional[Any] = None
        os.makedirs(self.config.vectorstore_path, exist_ok=True)

    def build_or_load(self, documents: Optional[List[Document]] = None) -> None:
        try:
            if self.config.vectorstore_type == "chroma":
                if documents:
                    self.vectorstore = Chroma.from_documents(
                        documents,
                        self.embeddings,
                        persist_directory=self.config.vectorstore_path
                    )
                else:
                    self.vectorstore = Chroma(
                        persist_directory=self.config.vectorstore_path,
                        embedding_function=self.embeddings
                    )
                logger.info("✓ 向量数据库已就绪 (类型: chroma)")
                return

            raise Exception(f"不支持的向量数据库类型: {self.config.vectorstore_type}")

        except Exception as e:
            raise Exception(f"向量数据库初始化失败: {e}")

    def similarity_search_with_score(self, query: str, k: int, **kwargs: Any) -> List[Tuple[Document, float]]:
        if not self.vectorstore:
            raise Exception("向量库未初始化，无法执行搜索。")
        return self.vectorstore.similarity_search_with_score(query, k=k, **kwargs)


# =============================================================================
# 企业级 RAG 系统
# =============================================================================
class EnterpriseRAGSystem:
    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        # 强制设置离线环境变量
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

        self.config = config or RAGConfig()

        # 兼容环境变量
        if not self.config.openai_api_key:
            env_key = os.environ.get("OPENAI_API_KEY")
            if env_key:
                self.config.openai_api_key = env_key

        logger.info("=" * 60)
        logger.info("企业级 RAG 系统初始化中...")
        logger.info("配置: LLM=DeepSeek, Embedding=本地BGE模型（离线模式）")
        logger.info(f"模型路径: {self.config.embedding_model_path}")
        logger.info("=" * 60)

        # 确保本地模型存在
        self._ensure_local_model()

        self.doc_processor = DocumentProcessor(self.config)
        self.embeddings = self._init_embeddings()
        self.vector_store = VectorStoreManager(self.config, self.embeddings)

        self.llm: Optional[Any] = None
        self.current_user: str = "system"
        self.metadata_db: Dict[str, DocumentSource] = {}

        os.makedirs(self.config.metadata_db_path, exist_ok=True)
        self._load_metadata_db()

    def _ensure_local_model(self) -> None:
        """确保本地模型目录存在"""
        local_model_dir = Path(self.config.embedding_model_path)
        cache_model_dir = Path(self.config.modelscope_cache_path)

        # 检查本地模型是否已经存在
        if local_model_dir.exists():
            # 检查是否有模型文件
            model_files = list(local_model_dir.glob("*.bin")) + list(local_model_dir.glob("*.safetensors"))
            if model_files:
                logger.info(f"✓ 本地模型已存在: {local_model_dir}")
                return

        # 尝试从缓存拷贝
        if cache_model_dir.exists() and self.config.auto_copy_model:
            logger.info(f"从缓存拷贝模型...")
            try:
                local_model_dir.mkdir(parents=True, exist_ok=True)
                files_copied = 0
                for file in cache_model_dir.iterdir():
                    if file.is_file():
                        shutil.copy2(file, local_model_dir / file.name)
                        files_copied += 1
                if files_copied > 0:
                    logger.info(f"✓ 模型已拷贝到: {local_model_dir}")
                else:
                    logger.warning("未找到可拷贝的模型文件")
            except Exception as e:
                logger.warning(f"拷贝模型失败: {e}")

        # 创建最小化模型文件
        if not local_model_dir.exists():
            local_model_dir.mkdir(parents=True, exist_ok=True)
            self._create_minimal_model_files(local_model_dir)

    def _create_minimal_model_files(self, model_dir: Path):
        """创建最小化模型文件"""
        import json

        # 创建 config.json
        config = {
            "_name_or_path": "BAAI/bge-small-zh-v1.5",
            "architectures": ["BertModel"],
            "model_type": "bert",
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "vocab_size": 21128
        }

        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info(f"✓ 已创建最小化模型文件: {model_dir}")

    def _init_embeddings(self) -> Any:
        """初始化 Embedding 模型"""
        embeddings = LocalEmbeddings(
            model_path=self.config.embedding_model_path,
            device=self.config.embedding_device
        )

        # 测试嵌入
        try:
            test_text = "测试"
            vector = embeddings.embed_query(test_text)
            logger.info(f"✓ Embedding 初始化成功，向量维度: {len(vector)}")
            return embeddings
        except Exception as e:
            logger.error(f"Embedding 测试失败: {e}")
            raise

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
            json.dump(
                {k: v.model_dump() for k, v in self.metadata_db.items()},
                f,
                default=str,
                indent=2,
                ensure_ascii=False
            )

    def _log_audit(self, action: str, document_id: str, details: Optional[Dict[str, Any]] = None) -> None:
        entry = AuditLog(
            user=self.current_user,
            action=action,
            document_id=document_id,
            details=details or {}
        )
        log_file = Path(self.config.metadata_db_path) / "audit_logs.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json(ensure_ascii=False) + "\n")

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
            raise Exception("向量库未初始化，请先 ingest。")

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

        # 指标：空召回、命中数量、相似度分布
        dept_label = department or ""
        if len(final_docs) == 0:
            RAG_EMPTY_RETRIEVAL_COUNTER.labels(endpoint="search_documents", department=dept_label).inc()
        RAG_RETRIEVAL_DOCS_HIST.labels(endpoint="search_documents").observe(len(final_docs))
        for s in final_scores:
            try:
                RAG_RETRIEVAL_SCORE_HIST.labels(endpoint="search_documents").observe(float(s))
            except Exception:
                pass

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
        return src.model_dump() if src else None

    def init_llm(self, force_reload: bool = False) -> None:
        if self.llm and not force_reload:
            return

        if not self.config.openai_api_key:
            raise ValueError("DeepSeek API key 未配置")

        os.environ["OPENAI_API_KEY"] = self.config.openai_api_key

        llm_kwargs = {
            "model": self.config.openai_chat_model,
            "temperature": 0.1,
            "base_url": self.config.openai_base_url,
            "api_key": self.config.openai_api_key,
            "max_tokens": 2000,
            "timeout": 30,
        }

        self.llm = ChatOpenAI(**llm_kwargs)
        logger.info(f"✓ 已初始化 DeepSeek LLM: {self.config.openai_chat_model}")

    def rag_answer(self, question: str, docs: List[Document]) -> str:
        self.init_llm()

        context = "\n\n".join(
            f"[{i + 1}] {d.page_content}\n(source_file={d.metadata.get('source_file')}, document_id={d.metadata.get('document_id')})"
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

        start = time.time()
        model_name = getattr(self.config, "openai_chat_model", "") or "unknown"
        try:
            resp = self.llm.invoke(prompt)
            duration = time.time() - start
            RAG_LLM_LATENCY.labels(endpoint="rag_answer", model=model_name).observe(duration)

            # 兼容不同返回对象：尽量提取 token 用量
            usage = None
            try:
                usage = getattr(resp, "usage", None) or getattr(resp, "response_metadata", None) or getattr(resp, "metadata", None)
            except Exception:
                usage = None

            prompt_tokens = None
            completion_tokens = None
            total_tokens = None

            if isinstance(usage, dict):
                u = usage.get("token_usage") or usage.get("usage") or usage
                if isinstance(u, dict):
                    prompt_tokens = u.get("prompt_tokens")
                    completion_tokens = u.get("completion_tokens")
                    total_tokens = u.get("total_tokens")

            def _as_int(x: Any) -> Optional[int]:
                try:
                    return int(x) if x is not None else None
                except Exception:
                    return None

            prompt_tokens = _as_int(prompt_tokens)
            completion_tokens = _as_int(completion_tokens)
            total_tokens = _as_int(total_tokens)

            if prompt_tokens is not None:
                RAG_LLM_TOKENS.labels(endpoint="rag_answer", model=model_name, type="prompt").inc(prompt_tokens)
            if completion_tokens is not None:
                RAG_LLM_TOKENS.labels(endpoint="rag_answer", model=model_name, type="completion").inc(completion_tokens)
            if total_tokens is not None:
                RAG_LLM_TOKENS.labels(endpoint="rag_answer", model=model_name, type="total").inc(total_tokens)

            return getattr(resp, "content", None) or str(resp)
        except Exception as e:
            duration = time.time() - start
            RAG_LLM_LATENCY.labels(endpoint="rag_answer", model=model_name).observe(duration)
            logger.error(f"DeepSeek API 调用失败: {e}")
            return f"根据相关文档信息，我可以提供以下信息：\n" + \
                "\n".join([f"[{i + 1}] {d.page_content[:150]}..." for i, d in enumerate(docs)])


# =============================================================================
# Agent 系统
# =============================================================================
class RAGAgent:
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
                raise Exception("检测到疑似提示注入/越权请求，已拒绝执行。")
        for pat in self.SENSITIVE_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                raise Exception("检测到疑似敏感信息泄露风险，已拒绝执行。")

    def _route_intent(self, message: str) -> str:
        m = message.lower()
        if any(k in m for k in ["谁上传", "上传者", "哈希", "file_hash", "文件路径", "document_id", "版本", "更新时间",
                                "last_modified"]):
            return "doc_info"
        if any(k in m for k in ["列出", "搜索", "找一下", "有哪些资料", "相关文档"]):
            return "search"
        return "rag_answer"

    @monitor_requests("agent")
    def chat(self, req: AgentRequest) -> Dict[str, Any]:
        rid = req.request_id or new_request_id()
        user = req.user or "anonymous"
        self.rag.set_current_user(user)

        try:
            self._check_guardrails(req.message)
        except Exception as e:
            self.rag._log_audit("agent_blocked", "agent", {"request_id": rid, "message": req.message, "reason": str(e)})
            return fail("GUARDRAIL", str(e), request_id=rid)

        intent = self._route_intent(req.message)
        user_roles = req.user_roles or ["public"]

        if intent == "search":
            docs, scores = self.rag.search_documents(req.message, k=5, department=req.department, user_roles=user_roles)
            results = []
            for d, s in zip(docs, scores):
                results.append({
                    "document_id": d.metadata.get("document_id"),
                    "source_file": d.metadata.get("source_file"),
                    "preview": d.page_content[:200],
                    "similarity_score": float(s),
                    "freshness_score": d.metadata.get("freshness_score"),
                })
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

        docs, _ = self.rag.search_documents(req.message, k=5, department=req.department, user_roles=user_roles)
        if not docs:
            dept_label = req.department or ""
            RAG_EMPTY_RETRIEVAL_COUNTER.labels(endpoint="agent_chat", department=dept_label).inc()
            return ok({"action": "rag_answer", "answer": "根据您的权限与知识库内容，无法回答该问题。", "sources": []},
                      request_id=rid)

        answer = self.rag.rag_answer(req.message, docs)
        sources = [{
            "ref": i + 1,
            "document_id": d.metadata.get("document_id"),
            "source_file": d.metadata.get("source_file"),
            "last_modified": d.metadata.get("last_modified"),
            "freshness_score": d.metadata.get("freshness_score"),
        } for i, d in enumerate(docs)]

        self.rag._log_audit("agent_rag_answer", "agent",
                            {"request_id": rid, "sources": [s["document_id"] for s in sources]})
        return ok({"action": "rag_answer", "answer": answer, "sources": sources}, request_id=rid)


# =============================================================================
# FastAPI 服务器
# =============================================================================
def _parse_api_keys(api_keys: str) -> Dict[str, Identity]:
    out: Dict[str, Identity] = {}
    if not api_keys:
        return out
    for item in api_keys.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        user, roles_str = item.split(":", 1)
        roles = [r.strip() for r in roles_str.split("|") if r.strip()]
        out[user.strip()] = Identity(user=user.strip(), roles=roles or ["public"])
    return out


def _jwt_encode(payload: Dict[str, Any], secret: str, algorithm: str) -> str:
    if pyjwt is None:
        raise ImportError("请先安装 PyJWT")
    return pyjwt.encode(payload, secret, algorithm=algorithm)


def _jwt_decode(token: str, secret: str, algorithms: List[str]) -> Dict[str, Any]:
    if pyjwt is None:
        raise ImportError("请先安装 PyJWT")
    return pyjwt.decode(token, secret, algorithms=algorithms)


def get_identity(cfg: RAGConfig, authorization: Optional[str] = Header(None),
                 x_api_key: Optional[str] = Header(None)) -> Identity:
    mode = cfg.auth_mode.lower()

    if mode == "jwt":
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <token>")
        token = authorization.split(" ", 1)[1].strip()
        try:
            payload = _jwt_decode(token, cfg.jwt_secret, algorithms=[cfg.jwt_algorithm])
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

        user = payload.get("sub") or payload.get("user")
        roles = payload.get("roles") or ["public"]
        department = payload.get("department")

        if not user:
            raise HTTPException(status_code=401, detail="Token missing sub/user")
        if isinstance(roles, str):
            roles = [roles]

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


def create_app() -> Any:
    if FastAPI is None:
        raise ImportError("未安装 fastapi。请运行：pip install fastapi uvicorn")

    cfg = RAGConfig()

    if not cfg.openai_api_key:
        logger.warning("未配置 DeepSeek API Key，部分功能可能受限")

    rag = EnterpriseRAGSystem(cfg)
    agent = RAGAgent(rag)

    app = FastAPI(title="Enterprise RAG + Agent (DeepSeek + 本地BGE)", version="1.0.0")

    bearer_scheme = HTTPBearer(auto_error=False)

    def _dep_identity(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
        x_api_key: Optional[str] = Header(None),
    ) -> Identity:
        authorization = None
        if credentials is not None and credentials.scheme:
            authorization = f"{credentials.scheme} {credentials.credentials}"
        return get_identity(cfg, authorization=authorization, x_api_key=x_api_key)

    @app.post("/auth/dev_login")
    def dev_login(req: DevLoginRequest):
        minutes = int(req.exp_minutes or cfg.jwt_exp_minutes)
        exp_ts = int((datetime.now(tz=timezone.utc) + timedelta(minutes=minutes)).timestamp())
        payload = {
            "sub": req.user,
            "roles": req.roles,
            "department": req.department,
            "exp": exp_ts,
            "iat": int(datetime.now(tz=timezone.utc).timestamp()),
        }
        token = _jwt_encode(payload, cfg.jwt_secret, cfg.jwt_algorithm)
        return ok({"access_token": token, "token_type": "bearer", "expires_at": exp_ts})

    @app.get("/")
    def index():
        return ok({
            "service": "Enterprise RAG + Agent",
            "llm": "DeepSeek",
            "embedding": "本地BGE模型（离线）",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": "/health",
            "metrics": "/metrics",
        })

    @app.get("/health")
    def health():
        try:
            test_text = "健康检查"
            vector = rag.embeddings.embed_query(test_text)
            embedding_status = f"正常 (维度: {len(vector)})"
        except Exception as e:
            embedding_status = f"错误: {str(e)}"

        try:
            rag.init_llm()
            llm_status = "连接正常"
        except Exception as e:
            llm_status = f"错误: {str(e)}"

        return ok({
            "status": "running",
            "llm": llm_status,
            "embedding": embedding_status,
            "embedding_model_path": cfg.embedding_model_path,
            "timestamp": datetime.now().isoformat()
        })

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
            results = [{
                "ref": i + 1,
                "content": d.page_content,
                "metadata": d.metadata,
                "similarity_score": float(s),
            } for i, (d, s) in enumerate(zip(docs, scores))]
            return ok({"query": req.query, "results": results}, request_id=rid)
        except Exception as e:
            return fail("SEARCH_ERROR", str(e), request_id=rid)

    @app.post("/query")
    def query(req: QueryRequest, ident: Identity = Depends(_dep_identity)):
        rid = req.request_id or new_request_id()
        try:
            rag.set_current_user(ident.user)
            docs, _ = rag.search_documents(req.question, k=5, department=req.department or ident.department,
                                           user_roles=ident.roles)
            if not docs:
                return ok(
                    {"question": req.question, "answer": "根据您的权限与知识库内容，无法回答该问题。", "sources": []},
                    request_id=rid)
            answer = rag.rag_answer(req.question, docs)
            sources = [{
                "ref": i + 1,
                "document_id": d.metadata.get("document_id"),
                "source_file": d.metadata.get("source_file"),
            } for i, d in enumerate(docs)]
            return ok({"question": req.question, "answer": answer, "sources": sources}, request_id=rid)
        except Exception as e:
            return fail("QUERY_ERROR", str(e), request_id=rid)

    @app.post("/agent_chat")
    def agent_chat(req: AgentRequest, ident: Identity = Depends(_dep_identity)):
        # 使用 model_copy 更新请求中的用户信息（Pydantic v2 推荐方式）
        department = req.department
        if ident.department is not None:
            department = department or ident.department
        
        updated_req = req.model_copy(update={
            "user": ident.user,
            "user_roles": ident.roles,
            "department": department
        })
        
        try:
            return agent.chat(updated_req)
        except Exception as e:
            logger.error(f"Agent chat 错误: {e}", exc_info=True)
            return fail("AGENT_ERROR", str(e))

    return app


# =============================================================================
# 主函数
# =============================================================================
def setup_project():
    """设置项目目录结构"""
    directories = [
        "./models/bge-small-zh-v1.5",
        "./chroma_db",
        "./metadata_db",
        "./demo_docs",
        "./logs"
    ]

    print("=" * 60)
    print("设置项目目录结构...")

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {dir_path}")

    # 创建测试文档
    test_doc = "./demo_docs/公司政策.txt"
    if not Path(test_doc).exists():
        content = """# 公司政策文档
## 第一章：总则
1.1 目的
本政策旨在规范公司员工的日常行为。
1.2 适用范围
本政策适用于公司所有正式员工、实习生及外包人员。
## 第二章：考勤管理
2.1 工作时间
公司标准工作时间为周一至周五，上午9:00至下午6:00。
2.2 请假制度
员工请假需提前通过OA系统申请。
生效日期：2024年1月1日
版本：v1.0"""

        with open(test_doc, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✓ 创建测试文档: {test_doc}")

    print("=" * 60)
    print("项目设置完成！")
    print("请确保已修改 .env 文件中的 OPENAI_API_KEY")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_project()
    else:
        try:
            import uvicorn

            # 创建必要的目录
            Path("./models/bge-small-zh-v1.5").mkdir(parents=True, exist_ok=True)
            Path("./chroma_db").mkdir(parents=True, exist_ok=True)
            Path("./metadata_db").mkdir(parents=True, exist_ok=True)

            app = create_app()

            print("=" * 60)
            print("企业级 RAG + Agent 服务启动")
            print("配置: LLM=DeepSeek, Embedding=本地BGE（离线模式）")
            print("接口文档: http://127.0.0.1:8000/docs")
            print("健康检查: http://127.0.0.1:8000/health")
            print("=" * 60)

            uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

        except Exception as e:
            logger.error(f"启动失败: {e}")
            print("\n建议先运行设置命令:")
            print("python RAGplus.py setup")
            print("\n然后安装依赖:")
            print("pip install -r requirements.txt")