import os
from typing import List, Dict, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class UniversalTextLoader:
    def __init__(self):
        self.loaders = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.md': UnstructuredMarkdownLoader
        }
        chunk_size = int(os.getenv("TEXT_CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("TEXT_CHUNK_OVERLAP", "200"))
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_file(self, file_path: str) -> List[Document]:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext not in self.loaders:
            raise ValueError(f"Unsupported file extension: {ext}")
            
        loader_cls = self.loaders[ext]
        try:
            loader = loader_cls(file_path)
            docs = loader.load()
            # Split into smaller, overlapping chunks for better retrieval precision
            if docs:
                docs = self.splitter.split_documents(docs)
                print(f"  Split into {len(docs)} chunks (size={self.splitter._chunk_size}, overlap={self.splitter._chunk_overlap})")
            return docs
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def load_directory(self, directory_path: str) -> List[Document]:
        documents = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    docs = self.load_file(file_path)
                    documents.extend(docs)
                except ValueError:
                    continue # Skip unsupported files
        return documents
