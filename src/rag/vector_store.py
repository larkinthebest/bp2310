import os
import time
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class VectorStore:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        self.cloud = os.getenv("PINECONE_CLOUD", "aws")
        self.region = os.getenv("PINECONE_REGION", "us-east-1")
            
        self.pc = Pinecone(api_key=self.api_key)
        
        # Load Index Names from Env
        self.index_name_text = os.getenv("PINECONE_INDEX_TEXT", "multimodal-rag-text")
        self.index_name_audio = os.getenv("PINECONE_INDEX_AUDIO", "multimodal-rag-audio")
        self.index_name_video = os.getenv("PINECONE_INDEX_VIDEO", "multimodal-rag-video")
        
        # Ensure Indexes Exist
        # Text & Audio use OpenAI Embeddings (1536)
        self._ensure_index(self.index_name_text, 1536)
        self._ensure_index(self.index_name_audio, 1536)
        
        # Video uses CLIP Embeddings (768)
        self._ensure_index(self.index_name_video, 768)
        
        # Initialize Embeddings
        self.text_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Initialize LangChain Wrappers
        self.text_store = PineconeVectorStore(
            index_name=self.index_name_text,
            embedding=self.text_embeddings
        )
        self.audio_store = PineconeVectorStore(
            index_name=self.index_name_audio,
            embedding=self.text_embeddings
        )
        
        # Video Index (Direct Access)
        self.video_index = self.pc.Index(self.index_name_video)

    def _ensure_index(self, name, dimension):
        if name not in self.pc.list_indexes().names():
            print(f"Creating Pinecone index: {name}")
            self.pc.create_index(
                name=name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region)
            )
            while not self.pc.describe_index(name).status['ready']:
                time.sleep(1)

    def add_documents(self, documents: List[Document]):
        text_docs = []
        audio_docs = []
        video_vectors = []
        
        for doc in documents:
            doc_type = doc.metadata.get("type", "text")
            
            if "embedding" in doc.metadata:
                # Video Frame / Image with pre-computed embedding
                vector_id = f"{doc.metadata.get('source', 'unknown')}_{time.time()}_{len(video_vectors)}"
                video_vectors.append({
                    "id": vector_id,
                    "values": doc.metadata["embedding"],
                    "metadata": {k: v for k, v in doc.metadata.items() if k != "embedding"}
                })
            elif doc_type == "audio" or doc_type == "video_audio":
                audio_docs.append(doc)
            else:
                text_docs.append(doc)
        
        if text_docs:
            print(f"Adding {len(text_docs)} text documents to {self.index_name_text}...")
            self.text_store.add_documents(text_docs)
            
        if audio_docs:
            print(f"Adding {len(audio_docs)} audio documents to {self.index_name_audio}...")
            self.audio_store.add_documents(audio_docs)
            
        if video_vectors:
            print(f"Adding {len(video_vectors)} video vectors to {self.index_name_video}...")
            batch_size = 100
            for i in range(0, len(video_vectors), batch_size):
                batch = video_vectors[i:i+batch_size]
                self.video_index.upsert(vectors=batch)

    def search(self, query: str, k: int = 3) -> Dict[str, List[Document]]:
        """Search across Text and Audio indexes using text query."""
        results = {}
        
        # Search Text
        results["text"] = self.text_store.similarity_search(query, k=k)
        
        # Search Audio
        results["audio"] = self.audio_store.similarity_search(query, k=k)
        
        return results

    def search_video(self, query_embedding: List[float], k: int = 3):
        """Search Video index with a separate embedding (CLIP)."""
        return self.video_index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
