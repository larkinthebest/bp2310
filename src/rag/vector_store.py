"""
Vector store layer — Pinecone index management, storage, retrieval, reranking.

Design decisions:
  - Deterministic vector IDs (hash of source + modality + timestamp/chunk_index).
  - Index provisioning is separated into `ensure_indexes()` — called explicitly.
  - Rerank fallback is explicit: returns `rerank_applied=False` on failure.
  - Retrieval returns scores when available.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import cfg

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────

@dataclass
class RerankResult:
    """Outcome of a reranking attempt."""
    items: List[Dict[str, Any]]
    rerank_applied: bool = True
    fallback_reason: str | None = None


# ── Helpers ───────────────────────────────────────────────────────

def _make_vector_id(source: str, modality: str, index: int = 0, timestamp: float | None = None) -> str:
    """Build a stable, deterministic ID for a Pinecone vector."""
    parts = [source, modality, str(index)]
    if timestamp is not None:
        parts.append(f"t{timestamp}")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ── VectorStore ───────────────────────────────────────────────────

class VectorStore:
    def __init__(self, *, auto_provision: bool = True):
        if not cfg.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not set")

        self.pc = Pinecone(api_key=cfg.pinecone_api_key)

        self.index_name_text = cfg.pinecone_index_text
        self.index_name_audio = cfg.pinecone_index_audio
        self.index_name_video = cfg.pinecone_index_video

        if auto_provision:
            self.ensure_indexes()

        # Select embeddings based on provider
        if cfg.is_google:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self.text_embeddings = GoogleGenerativeAIEmbeddings(
                model=cfg.embedding_model,
                google_api_key=cfg.google_api_key,
            )
        else:
            from langchain_openai import OpenAIEmbeddings
            self.text_embeddings = OpenAIEmbeddings(model=cfg.embedding_model)

        # LangChain wrappers for text/audio
        self.text_store = PineconeVectorStore(
            index_name=self.index_name_text,
            embedding=self.text_embeddings,
        )
        self.audio_store = PineconeVectorStore(
            index_name=self.index_name_audio,
            embedding=self.text_embeddings,
        )

        # Direct video index handle
        self.video_index = self.pc.Index(self.index_name_video)
        self._video_lock = threading.Lock()

    # ── Provisioning (separate from runtime) ──────────────────────

    def ensure_indexes(self) -> None:
        """Create Pinecone indexes if they do not exist. Safe to call multiple times."""
        emb_dim = cfg.embedding_dimension
        self._ensure_index(self.index_name_text, emb_dim)
        self._ensure_index(self.index_name_audio, emb_dim)
        self._ensure_index(self.index_name_video, 768)  # CLIP is always 768

    def _ensure_index(self, name: str, dimension: int) -> None:
        existing = self.pc.list_indexes().names()
        if name in existing:
            return
        logger.info("Creating Pinecone index: %s (dim=%d)", name, dimension)
        self.pc.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cfg.pinecone_cloud, region=cfg.pinecone_region),
        )
        while not self.pc.describe_index(name).status["ready"]:
            time.sleep(1)
        logger.info("Index %s ready", name)

    # ── Storage ───────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> None:
        text_docs: list[Document] = []
        audio_docs: list[Document] = []
        video_vectors: list[dict] = []

        frame_counter: dict[str, int] = {}  # per-source frame counter

        for doc in documents:
            doc_type = doc.metadata.get("type", "text")

            if "embedding" in doc.metadata:
                source = doc.metadata.get("source", "unknown")
                ts = doc.metadata.get("timestamp")
                idx = frame_counter.get(source, 0)
                frame_counter[source] = idx + 1

                vector_id = _make_vector_id(source, doc_type, idx, ts)
                video_vectors.append({
                    "id": vector_id,
                    "values": doc.metadata["embedding"],
                    "metadata": {
                        **{k: v for k, v in doc.metadata.items() if k != "embedding"},
                        "caption": doc.page_content,
                    },
                })
            elif doc_type in ("audio", "video_audio"):
                audio_docs.append(doc)
            else:
                text_docs.append(doc)

        if text_docs:
            logger.info("Adding %d text docs → %s", len(text_docs), self.index_name_text)
            self.text_store.add_documents(text_docs)

        if audio_docs:
            logger.info("Adding %d audio docs → %s", len(audio_docs), self.index_name_audio)
            self.audio_store.add_documents(audio_docs)

        if video_vectors:
            logger.info("Adding %d video vectors → %s", len(video_vectors), self.index_name_video)
            batch_size = 100
            with self._video_lock:
                for i in range(0, len(video_vectors), batch_size):
                    self.video_index.upsert(vectors=video_vectors[i : i + batch_size])

    # ── Retrieval ─────────────────────────────────────────────────

    def search(self, query: str, k: int = 3) -> Dict[str, List[Document]]:
        """Search text and audio indexes."""
        return {
            "text": self.text_store.similarity_search(query, k=k),
            "audio": self.audio_store.similarity_search(query, k=k),
        }

    def search_video(self, query_embedding: List[float], k: int = 5) -> dict:
        """Search the video index with a CLIP embedding."""
        return self.video_index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
        )

    # ── Reranking ─────────────────────────────────────────────────

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> RerankResult:
        """
        Rerank documents using Pinecone's reranker.
        Returns a RerankResult with explicit `rerank_applied` flag.
        """
        if not documents:
            return RerankResult(items=[], rerank_applied=False, fallback_reason="empty input")

        max_chars = 800
        truncated = [d[:max_chars] for d in documents]

        try:
            results = self.pc.inference.rerank(
                model="bge-reranker-v2-m3",
                query=query,
                documents=truncated,
                top_n=top_n,
                return_documents=True,
            )
            items = [{"index": r.index, "score": r.score} for r in results.data]
            return RerankResult(items=items, rerank_applied=True)
        except Exception as e:
            logger.warning("Reranking failed: %s — returning original order", e)
            items = [{"index": i, "score": 0.0} for i in range(min(top_n, len(documents)))]
            return RerankResult(
                items=items,
                rerank_applied=False,
                fallback_reason=str(e),
            )
