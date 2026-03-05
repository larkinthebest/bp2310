from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langfuse.langchain import CallbackHandler
import os
import re
from collections import defaultdict

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

class RAGPipeline:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        model_name = os.getenv("LLM_MODEL", "gpt-5-mini")
        print(f"Using LLM model: {model_name}")
        self.llm = ChatOpenAI(model=model_name)
        self.langfuse_handler = CallbackHandler()
        self.video_top_k = int(os.getenv("VIDEO_TOP_K", "30"))
        
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

    def _expand_query(self, question: str) -> str:
        """Use LLM to expand the user's query with sports-specific terms for better retrieval."""
        try:
            expansion_prompt = ChatPromptTemplate.from_template(
                "You are a sports search assistant. Given the user's question, generate a short list of "
                "additional sports-specific keywords and synonyms that would help find relevant content. "
                "Include terms for: player actions, match events, tactical terms, and common commentary phrases. "
                "Return ONLY the keywords separated by spaces, no explanations. Keep it under 30 words.\n\n"
                "Question: {question}\n\nKeywords:"
            )
            chain = expansion_prompt | self.llm | StrOutputParser()
            keywords = chain.invoke({"question": question})
            expanded = f"{question} {keywords.strip()}"
            print(f"  Query expanded: '{question}' → +[{keywords.strip()}]")
            return expanded
        except Exception as e:
            print(f"  Warning: query expansion failed: {e}")
            return question

    def _group_temporal_context(self, video_matches) -> list[str]:
        """
        Group video frames by source file, sort by timestamp, and produce
        scene-block strings so the LLM sees a coherent narrative per video.
        """
        # Bucket frames by source
        source_frames = defaultdict(list)
        for match in video_matches:
            meta = match['metadata']
            src = meta.get("source", "unknown")
            caption = meta.get("caption", "Visual content")
            timestamp = meta.get("timestamp", 0)
            source_frames[src].append((timestamp, caption))

        context_strs = []
        for src, frames in source_frames.items():
            basename = os.path.basename(src)
            # Sort frames chronologically
            frames.sort(key=lambda x: x[0])

            # Group consecutive frames into scenes (gap > 2× interval = new scene)
            interval = float(os.getenv("VIDEO_FRAME_INTERVAL_SECONDS", "3"))
            max_gap = interval * 2.5  # allow small gaps before splitting

            scenes = []
            current_scene = [frames[0]]
            for i in range(1, len(frames)):
                if frames[i][0] - frames[i - 1][0] <= max_gap:
                    current_scene.append(frames[i])
                else:
                    scenes.append(current_scene)
                    current_scene = [frames[i]]
            scenes.append(current_scene)

            for scene in scenes:
                start_t = scene[0][0]
                end_t = scene[-1][0]
                header = f"[Source: {basename}] [Scene: {start_t}s – {end_t}s]"
                body_lines = [f"  [{t}s] {cap}" for t, cap in scene]
                context_strs.append(header + "\n" + "\n".join(body_lines))

        return context_strs

    def hybrid_retrieval(self, question: str, top_k: int = 5, use_reranking: bool = True):
        # --- Query Expansion for text/audio retrieval ---
        expanded_question = self._expand_query(question)

        # 1. Text & Audio Retrieval (OpenAI Embeddings) — use expanded query
        candidate_k = top_k * 3 if use_reranking else top_k
        
        search_results = self.vector_store.search(expanded_question, k=candidate_k)
        text_docs = search_results.get("text", [])
        audio_docs = search_results.get("audio", [])
        
        # 2. Rerank text and audio documents together (if enabled)
        if use_reranking:
            all_text_docs = text_docs + audio_docs
            if all_text_docs:
                doc_texts = [doc.page_content for doc in all_text_docs]
                
                # Rerank against the ORIGINAL question (not expanded) for precision
                print(f"Reranking {len(doc_texts)} text/audio documents...")
                reranked_results = self.vector_store.rerank(question, doc_texts, top_n=top_k)
                
                reranked_docs = []
                for result in reranked_results:
                    original_idx = result["index"]
                    if original_idx < len(all_text_docs):
                        doc = all_text_docs[original_idx]
                        doc.metadata["rerank_score"] = result["score"]
                        reranked_docs.append(doc)
                
                text_docs = [d for d in reranked_docs if d.metadata.get("type") != "audio" and d.metadata.get("type") != "video_audio"]
                audio_docs = [d for d in reranked_docs if d.metadata.get("type") in ["audio", "video_audio"]]
        
        # 3. Video Retrieval (CLIP Embeddings) — use ORIGINAL question (CLIP prefers short queries)
        query_embedding = self.get_clip_text_embedding(question)
        video_results = self.vector_store.search_video(query_embedding, k=self.video_top_k)
        
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
        
        # Video Context — grouped into temporal scene blocks
        video_context_strs = self._group_temporal_context(video_matches)
        
        full_context = "\n\n".join(context_parts + video_context_strs)
        
        # Incorporate Feedback if present
        question_text = f"Question: {question}"
        if feedback:
            question_text += f"\n\nContext Note: A previous answer to this question was not relevant enough. Feedback: {feedback}\nPlease improve the answer based on this feedback."

        # Generate Answer
        template = """You are an expert sports commentator and analyst. Use the video frame descriptions, 
audio transcriptions, and documents provided to give vivid, detailed analysis of the 
moment the user is asking about. Reference specific timestamps when available.

Each context chunk starts with [Source: filename]. Video scenes show chronological 
frame descriptions grouped by time range — use the timeline to understand the flow of play.
        
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
        
        pattern = r"(?i)(\n|\r\n)\s*(?:\*\*|#+\s*)?Sources?\s*(?:Used)?\s*(?:\*\*|:)?\s*[:\-]?\s*(.*)"
        
        match = re.search(pattern, answer_text, re.DOTALL)
        if match:
            final_answer = answer_text[:match.start()].strip()
            sources_str = match.group(2).strip()
            
            if sources_str and sources_str.lower() != "none":
                sources_list = [s.strip() for s in sources_str.split(',') if s.strip()]

        return {
            "answer": final_answer,
            "contexts": contexts,
            "sources": sources_list
        }
