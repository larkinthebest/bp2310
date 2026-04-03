"""
RAG Pipeline — retrieval, context assembly, generation.

Architecture changes from the original:
  - Sources are derived from retrieval metadata, NOT regex-parsed from LLM output.
  - Context assembly is bounded by MAX_CONTEXT_CHARS to prevent prompt overflow.
  - Domain-specific prompting is loaded from config profiles, not hard-coded.
  - Returns a structured QueryResult dataclass.
  - Uses shared CLIPManager instead of loading its own CLIP.
  - LLM still appends "Sources Used:" for display, but backend ignores it.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.config import cfg
from src.models import clip_manager

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Structured output from the RAG pipeline."""
    answer: str
    sources: list[str]                          # from retrieval metadata
    contexts: list[str]                         # raw context strings shown to LLM
    video_matches: list[dict]                   # raw Pinecone match dicts
    rerank_applied: bool = False
    rerank_fallback_reason: str | None = None
    context_truncated: bool = False


# ── Pipeline ──────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self, vector_store):
        self.vector_store = vector_store

        # Select LLM based on provider
        if cfg.is_google:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model=cfg.llm_model,
                google_api_key=cfg.google_api_key,
            )
        else:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=cfg.llm_model)

        self._video_top_k = cfg.video_top_k
        self._text_top_k = cfg.text_top_k
        self._max_context_chars = cfg.max_context_chars

        # Langfuse (optional — fail gracefully if not configured)
        self._callbacks = []
        try:
            from langfuse.langchain import CallbackHandler
            self._callbacks.append(CallbackHandler())
        except Exception:
            logger.debug("Langfuse not configured — skipping callback")

        logger.info("RAG pipeline ready (model=%s)", cfg.llm_model)

    # ── Retrieval ─────────────────────────────────────────────────

    def hybrid_retrieval(
        self,
        question: str,
        top_k: int | None = None,
        use_reranking: bool = True,
        source_filter: list[str] | None = None,
    ) -> tuple[list, list, list, bool, str | None]:
        """
        Three-way retrieval: text, audio, video.
        Returns (text_docs, audio_docs, video_matches, rerank_applied, rerank_reason).
        """
        top_k = top_k or self._text_top_k
        candidate_k = top_k * 3 if use_reranking else top_k

        # 1. Text + Audio search
        search_results = self.vector_store.search(question, k=candidate_k)
        text_docs = search_results.get("text", [])
        audio_docs = search_results.get("audio", [])

        # Apply source filter
        if source_filter:
            names = set(source_filter)
            text_docs = [d for d in text_docs if os.path.basename(d.metadata.get("source", "")) in names]
            audio_docs = [d for d in audio_docs if os.path.basename(d.metadata.get("source", "")) in names]

        # 2. Rerank
        rerank_applied = False
        rerank_reason = None
        if use_reranking:
            all_docs = text_docs + audio_docs
            if all_docs:
                doc_texts = [doc.page_content for doc in all_docs]
                result = self.vector_store.rerank(question, doc_texts, top_n=top_k)
                rerank_applied = result.rerank_applied
                rerank_reason = result.fallback_reason

                reranked_docs = []
                for item in result.items:
                    idx = item["index"]
                    if idx < len(all_docs):
                        doc = all_docs[idx]
                        doc.metadata["rerank_score"] = item["score"]
                        reranked_docs.append(doc)

                text_docs = [d for d in reranked_docs if d.metadata.get("type") not in ("audio", "video_audio")]
                audio_docs = [d for d in reranked_docs if d.metadata.get("type") in ("audio", "video_audio")]

        # 3. Video search (CLIP)
        query_embedding = clip_manager.get_text_embedding(question)
        video_results = self.vector_store.search_video(query_embedding, k=self._video_top_k)
        video_matches = video_results.get("matches", [])

        if source_filter:
            names = set(source_filter)
            video_matches = [m for m in video_matches if os.path.basename(m["metadata"].get("source", "")) in names]

        return text_docs, audio_docs, video_matches, rerank_applied, rerank_reason

    # ── Context building (bounded) ────────────────────────────────

    def _build_context(self, text_docs, audio_docs, video_matches) -> tuple[str, list[str], bool]:
        """
        Build a bounded context string from retrieved documents.
        Returns (context_string, raw_context_parts, was_truncated).
        """
        parts: list[str] = []

        for d in text_docs:
            src = os.path.basename(d.metadata.get("source", "unknown"))
            parts.append(f"[Source: {src}]\n{d.page_content}")

        for d in audio_docs:
            src = os.path.basename(d.metadata.get("source", "unknown"))
            parts.append(f"[Source: {src}] [Audio Transcript]\n{d.page_content}")

        video_parts = self._group_temporal_context(video_matches)
        parts.extend(video_parts)

        # Enforce context budget
        truncated = False
        budget = self._max_context_chars
        selected: list[str] = []
        used = 0
        for part in parts:
            if used + len(part) > budget:
                truncated = True
                break
            selected.append(part)
            used += len(part)

        context_str = "\n\n".join(selected)
        return context_str, parts, truncated

    # ── Source extraction from metadata ────────────────────────────

    @staticmethod
    def _extract_sources(text_docs, audio_docs, video_matches) -> list[str]:
        """Build source list from retrieval metadata — NOT from LLM text."""
        seen: set[str] = set()
        sources: list[str] = []
        for d in text_docs + audio_docs:
            name = os.path.basename(d.metadata.get("source", ""))
            if name and name not in seen:
                seen.add(name)
                sources.append(name)
        for m in video_matches:
            name = os.path.basename(m["metadata"].get("source", ""))
            if name and name not in seen:
                seen.add(name)
                sources.append(name)
        return sources

    # ── Temporal scene grouping ───────────────────────────────────

    @staticmethod
    def _group_temporal_context(video_matches) -> list[str]:
        if not video_matches:
            return []

        source_frames: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for match in video_matches:
            meta = match["metadata"]
            src = meta.get("source", "unknown")
            caption = meta.get("caption", "Visual content")
            timestamp = meta.get("timestamp", 0)
            source_frames[src].append((timestamp, caption))

        context_strs: list[str] = []
        for src, frames in source_frames.items():
            basename = os.path.basename(src)
            frames.sort(key=lambda x: x[0])

            interval = cfg.video_frame_interval
            max_gap = interval * 2.5

            scenes: list[list[tuple[float, str]]] = []
            current: list[tuple[float, str]] = [frames[0]]
            for i in range(1, len(frames)):
                if frames[i][0] - frames[i - 1][0] <= max_gap:
                    current.append(frames[i])
                else:
                    scenes.append(current)
                    current = [frames[i]]
            scenes.append(current)

            for scene in scenes:
                start_t, end_t = scene[0][0], scene[-1][0]
                header = f"[Source: {basename}] [Scene: {start_t}s – {end_t}s]"
                body = "\n".join(f"  [{t}s] {cap}" for t, cap in scene)
                context_strs.append(f"{header}\n{body}")

        return context_strs

    # ── Main query method ─────────────────────────────────────────

    def query(
        self,
        question: str,
        feedback: str | None = None,
        use_reranking: bool = True,
        source_filter: list[str] | None = None,
        history: list[dict] | None = None,
    ) -> QueryResult:
        """Run the full RAG pipeline and return structured results."""

        # 1. Retrieve
        text_docs, audio_docs, video_matches, rerank_applied, rerank_reason = (
            self.hybrid_retrieval(question, use_reranking=use_reranking, source_filter=source_filter)
        )

        # 2. Build bounded context
        context_str, raw_contexts, truncated = self._build_context(text_docs, audio_docs, video_matches)

        # 3. Sources from metadata
        sources = self._extract_sources(text_docs, audio_docs, video_matches)

        # 4. History
        history_str = ""
        if history:
            lines = []
            for msg in history[-10:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                lines.append(f"{role}: {msg.get('content', '')}")
            history_str = "\n".join(lines)

        # 5. Question
        question_text = f"Question: {question}"
        if feedback:
            question_text += f"\nFeedback: {feedback}. Improve accordingly."

        # 6. Generate
        template = """{system_prompt}

Rules:
1. Be concise — keep answers to 2-4 sentences.
2. Base your answer on the context provided. If the context has relevant information, use it.
3. When referencing video frames, cite the timestamp: [Frame: 24s].
4. Only say you don't know if the context truly has nothing relevant.

{history}

CONTEXT:
{context}

{question_block}"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {
                "system_prompt": lambda _: cfg.system_prompt,
                "context": lambda _: context_str,
                "question_block": RunnablePassthrough(),
                "history": lambda _: history_str,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        answer_text = chain.invoke(question_text, config={"callbacks": self._callbacks})

        # 7. Strip any "Sources Used:" line the LLM may have appended
        # (We already have authoritative sources from metadata.)
        import re
        pattern = r"(?i)(\n|\r\n)\s*(?:\*\*|#+\s*)?Sources?\s*(?:Used)?\s*(?:\*\*|:)?\s*[:\-]?\s*(.*)"
        match = re.search(pattern, answer_text, re.DOTALL)
        if match:
            answer_text = answer_text[: match.start()].strip()

        return QueryResult(
            answer=answer_text,
            sources=sources,
            contexts=raw_contexts,
            video_matches=video_matches,
            rerank_applied=rerank_applied,
            rerank_fallback_reason=rerank_reason,
            context_truncated=truncated,
        )
