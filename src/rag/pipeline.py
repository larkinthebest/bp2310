from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langfuse.langchain import CallbackHandler
import os
import re
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

    def hybrid_retrieval(self, question: str, top_k: int = 3, use_reranking: bool = True):
        # 1. Text & Audio Retrieval (OpenAI Embeddings)
        # Retrieve more candidates if reranking is enabled
        candidate_k = top_k * 3 if use_reranking else top_k
        
        search_results = self.vector_store.search(question, k=candidate_k)
        text_docs = search_results.get("text", [])
        audio_docs = search_results.get("audio", [])
        
        # 2. Rerank text and audio documents together (if enabled)
        if use_reranking:
            all_text_docs = text_docs + audio_docs
            if all_text_docs:
                # Prepare documents for reranking
                doc_texts = [doc.page_content for doc in all_text_docs]
                
                # Rerank using Pinecone
                print(f"Reranking {len(doc_texts)} text/audio documents...")
                reranked_results = self.vector_store.rerank(question, doc_texts, top_n=top_k)
                
                # Map reranked results back to original documents
                reranked_docs = []
                for result in reranked_results:
                    original_idx = result["index"]
                    if original_idx < len(all_text_docs):
                        doc = all_text_docs[original_idx]
                        doc.metadata["rerank_score"] = result["score"]
                        reranked_docs.append(doc)
                
                # Split back into text and audio based on type
                text_docs = [d for d in reranked_docs if d.metadata.get("type") != "audio" and d.metadata.get("type") != "video_audio"]
                audio_docs = [d for d in reranked_docs if d.metadata.get("type") in ["audio", "video_audio"]]
        
        # 3. Video Retrieval (CLIP Embeddings) - separate modality, no text reranking
        query_embedding = self.get_clip_text_embedding(question)
        video_results = self.vector_store.search_video(query_embedding, k=2)
        
        # Return raw matches for video
        video_matches = video_results.get('matches', [])
                
        return text_docs, audio_docs, video_matches



    def query(self, question: str, feedback: str = None, use_reranking: bool = True):
        # Perform Hybrid Retrieval
        text_docs, audio_docs, video_matches = self.hybrid_retrieval(question, use_reranking=use_reranking)
        
        # Format Context with Sources
        context_parts = []
        
        # Text Context
        for d in text_docs:
            src = os.path.basename(d.metadata.get("source", "unknown"))
            context_parts.append(f"[Source: {src}]\n{d.page_content}")
            
        # Audio Context
        for d in audio_docs:
            src = os.path.basename(d.metadata.get("source", "unknown"))
            context_parts.append(f"[Source: {src}] [Audio Transcript]\n{d.page_content}")
        
        # Video Context
        video_context_strs = []
        for match in video_matches:
            meta = match['metadata']
            src = os.path.basename(meta.get("source", "unknown"))
            
            if meta.get('type') == 'video_frame':
                video_context_strs.append(f"[Source: {src}] [Video Frame] at {meta.get('timestamp')}s")
            elif meta.get('type') == 'image':
                video_context_strs.append(f"[Source: {src}] [Image]")
            else:
                 video_context_strs.append(f"[Source: {src}] [Visual]")
        
        full_context = "\n\n".join(context_parts + video_context_strs)
        
        # Incorporate Feedback if present
        question_text = f"Question: {question}"
        if feedback:
            question_text += f"\n\nContext Note: A previous answer to this question was not relevant enough. Feedback: {feedback}\nPlease improve the answer based on this feedback."

        # Generate Answer
        template = """Answer the question based on the following context. 
        Each context chunk starts with [Source: filename].
        
        {context}
        
        {question_block}
        
        At the end of your answer, explicitly list the unique source filenames you used to derive the answer, in the format:
        Sources Used: filename1, filename2...
        If no context was used, do not list sources.
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": lambda x: full_context, "question_block": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        contexts = [d.page_content for d in text_docs] + \
                   [d.page_content for d in audio_docs] + \
                   video_context_strs
                   
        answer_text = chain.invoke(question_text, config={"callbacks": [self.langfuse_handler]})
        
        # Parse Sources from Answer using Regex
        final_answer = answer_text
        sources_list = []
        
        # Pattern to match "Sources Used:" or similar variations, case insensitive
        # Matches: Sources Used:, Source Used:, Sources:, Source:
        # capturing the rest of the line or text
        pattern = r"(?i)(\n|\r\n)\s*(?:\*\*|#+\s*)?Sources?\s*(?:Used)?\s*(?:\*\*|:)?\s*[:\-]?\s*(.*)"
        
        match = re.search(pattern, answer_text, re.DOTALL)
        if match:
            # Everything before the match is the answer
            final_answer = answer_text[:match.start()].strip()
            sources_str = match.group(2).strip()
            
            if sources_str and sources_str.lower() != "none":
                # Split by comma, strip whitespace, modify to simple basenames if needed
                sources_list = [s.strip() for s in sources_str.split(',') if s.strip()]

        return {
            "answer": final_answer,
            "contexts": contexts,
            "sources": sources_list
        }
