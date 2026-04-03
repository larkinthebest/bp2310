"""
Text / document loader with proper chunk metadata enrichment.
"""

from __future__ import annotations

import logging
import os
from typing import List

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import cfg

logger = logging.getLogger(__name__)

# Extension → LangChain loader class
_LOADER_MAP = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
}


class UniversalTextLoader:
    """Loads text-based files, splits into chunks, and enriches metadata."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self._chunk_size = chunk_size or cfg.text_chunk_size
        self._chunk_overlap = chunk_overlap or cfg.text_chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=len,
        )

    def load_file(self, file_path: str) -> List[Document]:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext not in _LOADER_MAP:
            raise ValueError(f"Unsupported text file extension: {ext}")

        loader_cls = _LOADER_MAP[ext]
        try:
            loader = loader_cls(file_path)
            docs = loader.load()
        except Exception as e:
            logger.error("Failed to load %s: %s", file_path, e)
            return []

        if not docs:
            return []

        # Split into chunks
        chunks = self._splitter.split_documents(docs)
        basename = os.path.basename(file_path)
        total_chunks = len(chunks)

        # Enrich metadata on every chunk
        for idx, chunk in enumerate(chunks):
            chunk.metadata.setdefault("source", file_path)
            chunk.metadata["source_basename"] = basename
            chunk.metadata["type"] = "text"
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_count"] = total_chunks

        logger.info(
            "  %s → %d chunks (size=%d, overlap=%d)",
            basename,
            total_chunks,
            self._chunk_size,
            self._chunk_overlap,
        )
        return chunks
