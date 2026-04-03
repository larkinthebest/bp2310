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
        self.video_top_k = int(os.getenv("VIDEO_TOP_K", "20"))
        
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
            frames.sort(key=lambda x: x[0])

            interval = float(os.getenv("VIDEO_FRAME_INTERVAL_SECONDS", "3"))
            max_gap = interval * 2.5

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

    def hybrid_retrieval(self, question: str, top_k: int = 5,
                         use_reranking: bool = True,
                         source_filter: list[str] | None = None):
        # 1. Text & Audio Retrieval
        candidate_k = top_k * 3 if use_reranking else top_k
        search_results = self.vector_store.search(question, k=candidate_k)
        text_docs = search_results.get("text", [])
        audio_docs = search_results.get("audio", [])

        # Filter by selected source files
        if source_filter:
            basenames = set(source_filter)
            text_docs = [d for d in text_docs if os.path.basename(d.metadata.get("source", "")) in basenames]
            audio_docs = [d for d in audio_docs if os.path.basename(d.metadata.get("source", "")) in basenames]

        # 2. Rerank
        if use_reranking:
            all_text_docs = text_docs + audio_docs
            if all_text_docs:
                doc_texts = [doc.page_content for doc in all_text_docs]
                print(f"Reranking {len(doc_texts)} text/audio documents...")
                reranked_results = self.vector_store.rerank(question, doc_texts, top_n=top_k)

                reranked_docs = []
                for result in reranked_results:
                    original_idx = result["index"]
                    if original_idx < len(all_text_docs):
                        doc = all_text_docs[original_idx]
                        doc.metadata["rerank_score"] = result["score"]
                        reranked_docs.append(doc)

                text_docs = [d for d in reranked_docs if d.metadata.get("type") not in ("audio", "video_audio")]
                audio_docs = [d for d in reranked_docs if d.metadata.get("type") in ("audio", "video_audio")]

        # 3. Video Retrieval (CLIP)
        query_embedding = self.get_clip_text_embedding(question)
        video_results = self.vector_store.search_video(query_embedding, k=self.video_top_k)
        video_matches = video_results.get('matches', [])

        # Filter video by source
        if source_filter:
            basenames = set(source_filter)
            video_matches = [m for m in video_matches if os.path.basename(m['metadata'].get('source', '')) in basenames]

        return text_docs, audio_docs, video_matches


    def query(self, question: str, feedback: str = None,
              use_reranking: bool = True,
              source_filter: list[str] | None = None,
              history: list[dict] | None = None):
        # Perform Hybrid Retrieval
        text_docs, audio_docs, video_matches = self.hybrid_retrieval(
            question, use_reranking=use_reranking, source_filter=source_filter
        )

        # Format Context with Sources
        context_parts = []
        for d in text_docs:
            src = os.path.basename(d.metadata.get("source", "unknown"))
            context_parts.append(f"[Source: {src}]\n{d.page_content}")
        for d in audio_docs:
            src = os.path.basename(d.metadata.get("source", "unknown"))
            context_parts.append(f"[Source: {src}] [Audio Transcript]\n{d.page_content}")

        video_context_strs = self._group_temporal_context(video_matches)
        full_context = "\n\n".join(context_parts + video_context_strs)

        # Build conversation history string (last 10 messages)
        history_str = ""
        if history:
            recent = history[-10:]
            lines = []
            for msg in recent:
                role = "User" if msg.get("role") == "user" else "Assistant"
                lines.append(f"{role}: {msg.get('content', '')}")
            history_str = "\n".join(lines)

        # Incorporate Feedback
        question_text = f"Question: {question}"
        if feedback:
            question_text += f"\nFeedback on previous answer: {feedback}. Improve accordingly."

        # Generate Answer
        template = """You are a sports commentator and analyst. Use the provided context to answer the question.

Rules:
1. Be concise — keep answers to 2-4 sentences.
2. Base your answer on the context provided. If the context has relevant information, use it.
3. When referencing video frames, cite the timestamp: [Frame: 24s].
4. Only say you don't know if the context truly has nothing relevant.
5. At the very end, on a new line, list sources: Sources Used: filename1, filename2...

{history}

CONTEXT:
{context}

{question_block}"""

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": lambda x: full_context, "question_block": RunnablePassthrough(), "history": lambda x: history_str}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        contexts = [d.page_content for d in text_docs] + \
                   [d.page_content for d in audio_docs] + \
                   video_context_strs

        answer_text = chain.invoke(question_text, config={"callbacks": [self.langfuse_handler]})

        # Parse Sources from Answer
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
            "sources": sources_list,
            "video_matches": video_matches
        }
