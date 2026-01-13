import os
from typing import List, Dict, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document

class UniversalTextLoader:
    def __init__(self):
        self.loaders = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.md': UnstructuredMarkdownLoader
        }

    def load_file(self, file_path: str) -> List[Document]:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext not in self.loaders:
            raise ValueError(f"Unsupported file extension: {ext}")
            
        loader_cls = self.loaders[ext]
        try:
            loader = loader_cls(file_path)
            return loader.load()
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
