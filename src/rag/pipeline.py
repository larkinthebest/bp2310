from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langfuse.langchain import CallbackHandler
import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

class RAGPipeline:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.langfuse_handler = CallbackHandler()
        
        # Initialize CLIP for query embedding
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)

    def get_clip_text_embedding(self, text: str):
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0].tolist()

    def hybrid_retrieval(self, question: str):
        # 1. Text & Audio Retrieval (OpenAI Embeddings)
        # Returns dict with 'text' and 'audio' keys
        search_results = self.vector_store.search(question, k=3)
        text_docs = search_results.get("text", [])
        audio_docs = search_results.get("audio", [])
        
        # 2. Video Retrieval (CLIP Embeddings)
        query_embedding = self.get_clip_text_embedding(question)
        video_results = self.vector_store.search_video(query_embedding, k=2)
        
        # Format Video Results
        video_context = []
        for match in video_results['matches']:
            meta = match['metadata']
            if meta.get('type') == 'video_frame':
                video_context.append(f"[Video Frame] Source: {meta.get('source')} at {meta.get('timestamp')}s")
            elif meta.get('type') == 'image':
                video_context.append(f"[Image] Source: {meta.get('source')}")
            else:
                 video_context.append(f"[Visual] Source: {meta.get('source')}")
                
        return text_docs, audio_docs, video_context

    def query(self, question: str):
        # Perform Hybrid Retrieval
        text_docs, audio_docs, video_context = self.hybrid_retrieval(question)
        
        # Combine Context
        text_context_str = "\n\n".join([d.page_content for d in text_docs])
        audio_context_str = "\n\n".join([f"[Audio Transcript]: {d.page_content}" for d in audio_docs])
        video_context_str = "\n".join(video_context)
        
        full_context = f"Text Context:\n{text_context_str}\n\nAudio Context:\n{audio_context_str}\n\nVisual Context:\n{video_context_str}"
        
        # Generate Answer
        template = """Answer the question based on the following context (which includes both text documents and descriptions of visual content):
        
        {context}
        
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": lambda x: full_context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        contexts = [d.page_content for d in text_docs] + \
                   [d.page_content for d in audio_docs] + \
                   video_context
                   
        answer = chain.invoke(question, config={"callbacks": [self.langfuse_handler]})
        
        return {
            "answer": answer,
            "contexts": contexts
        }
