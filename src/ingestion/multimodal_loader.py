"""
Multimodal ingestion: video, audio, image, text/PDF.

Key design decisions:
  - Uses the shared CLIPManager singleton (src.models) — never loads CLIP itself.
  - Domain-specific captioning is driven by config.caption_prompt, not hard-coded.
  - Video processing uses try/finally for safe resource cleanup.
  - Captioning is optional (ENABLE_CAPTIONS=false skips Vision API calls).
"""

from __future__ import annotations

import base64
import io
import logging
import math
import os
from typing import List

from langchain_core.documents import Document
from moviepy.editor import VideoFileClip
from PIL import Image

from src.config import cfg
from src.models import clip_manager

logger = logging.getLogger(__name__)


class MultimodalLoader:
    """Loads and processes files of any supported modality."""

    def __init__(self, openai_api_key: str | None = None):
        self._caption_prompt = cfg.caption_prompt
        self._caption_model = cfg.caption_model
        self._caption_max_tokens = cfg.caption_max_tokens
        self._enable_captions = cfg.enable_captions
        self._frame_interval = cfg.video_frame_interval

        if cfg.is_google:
            import google.generativeai as genai
            genai.configure(api_key=cfg.google_api_key)
            self._genai = genai
            self._provider = "google"
        else:
            import openai as _openai
            key = openai_api_key or cfg.openai_api_key
            if not key:
                raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
            _openai.api_key = key
            self._openai = _openai
            self._provider = "openai"

    # ── Image helpers ─────────────────────────────────────────────

    @staticmethod
    def _encode_image_b64(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def caption_image(self, image: Image.Image) -> str:
        """Caption an image using the Vision API. Falls back gracefully."""
        if not self._enable_captions:
            return "Visual content"
        try:
            if self._provider == "google":
                return self._caption_google(image)
            else:
                return self._caption_openai(image)
        except Exception as e:
            logger.warning("Vision caption failed: %s", e)
            return "Visual content (caption unavailable)"

    def _caption_google(self, image: Image.Image) -> str:
        """Caption using Google Gemini Vision."""
        model = self._genai.GenerativeModel(self._caption_model)
        response = model.generate_content(
            [self._caption_prompt, image],
            generation_config=self._genai.types.GenerationConfig(
                max_output_tokens=self._caption_max_tokens,
            ),
        )
        text = (response.text or "").strip()
        return text or "Visual content"

    def _caption_openai(self, image: Image.Image) -> str:
        """Caption using OpenAI Vision."""
        b64 = self._encode_image_b64(image)
        resp = self._openai.chat.completions.create(
            model=self._caption_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": self._caption_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            max_completion_tokens=self._caption_max_tokens,
        )
        content = resp.choices[0].message.content
        if isinstance(content, list):
            text = "".join(p.get("text", "") for p in content if isinstance(p, dict)).strip()
        else:
            text = (content or "").strip()
        return text or "Visual content"

    # ── Audio ─────────────────────────────────────────────────────

    def process_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper (OpenAI) or Gemini."""
        if self._provider == "google":
            return self._transcribe_google(audio_path)
        else:
            return self._transcribe_openai(audio_path)

    def _transcribe_openai(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            result = self._openai.audio.transcriptions.create(model="whisper-1", file=f)
        return result.text

    def _transcribe_google(self, audio_path: str) -> str:
        """Transcribe audio using Gemini's audio understanding."""
        audio_file = self._genai.upload_file(audio_path)
        model = self._genai.GenerativeModel(self._caption_model)
        response = model.generate_content(
            ["Transcribe this audio accurately. Return only the transcribed text, nothing else.", audio_file],
        )
        return (response.text or "").strip()

    # ── Video ─────────────────────────────────────────────────────

    def process_video(self, video_path: str) -> List[Document]:
        """Extract audio transcript + frame embeddings/captions from a video."""
        video = VideoFileClip(video_path)
        documents: List[Document] = []
        try:
            # 1. Audio transcript
            documents.extend(self._extract_audio(video, video_path))

            # 2. Frame embeddings + captions
            documents.extend(self._extract_frames(video, video_path))
        finally:
            video.close()

        return documents

    def _extract_audio(self, video: VideoFileClip, video_path: str) -> List[Document]:
        audio_path = f"{video_path}.tmp.mp3"
        docs: List[Document] = []
        try:
            if video.audio is not None:
                video.audio.write_audiofile(audio_path, verbose=False, logger=None)
                text = self.process_audio(audio_path)
                docs.append(Document(
                    page_content=f"Video Transcription: {text}",
                    metadata={"source": video_path, "type": "video_audio"},
                ))
        except Exception as e:
            logger.error("Audio extraction failed for %s: %s", video_path, e)
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        return docs

    def _extract_frames(self, video: VideoFileClip, video_path: str) -> List[Document]:
        duration = video.duration
        interval = self._frame_interval
        total = int(math.floor(duration / interval)) + 1
        frame_times = [round(i * interval, 2) for i in range(total)]

        docs: List[Document] = []
        for i, t in enumerate(frame_times, 1):
            safe_t = min(t, duration - 0.1)
            try:
                raw_frame = video.get_frame(safe_t)
            except Exception as e:
                logger.warning("Frame read failed at %.1fs in %s: %s", t, video_path, e)
                continue

            image = Image.fromarray(raw_frame)
            embedding = clip_manager.get_image_embedding(image)
            caption = self.caption_image(image)

            docs.append(Document(
                page_content=f"{caption} (frame at {t}s)",
                metadata={
                    "source": video_path,
                    "type": "video_frame",
                    "timestamp": t,
                    "embedding": embedding,
                },
            ))
            if i % 5 == 0 or i == total:
                logger.info("Ingested %d/%d frames …", i, total)

        return docs

    # ── Dispatcher ────────────────────────────────────────────────

    def load_file(self, file_path: str) -> List[Document]:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower().strip()

        if ext in (".jpg", ".jpeg", ".png"):
            image = Image.open(file_path)
            return [Document(
                page_content=self.caption_image(image),
                metadata={
                    "source": file_path,
                    "type": "image",
                    "embedding": clip_manager.get_image_embedding(image),
                },
            )]

        if ext in (".mp3", ".wav", ".m4a"):
            text = self.process_audio(file_path)
            return [Document(
                page_content=text,
                metadata={"source": file_path, "type": "audio"},
            )]

        if ext in (".mp4", ".mov", ".avi", ".mkv"):
            return self.process_video(file_path)

        if ext in (".pdf", ".txt", ".docx", ".md"):
            from src.ingestion.text_loader import UniversalTextLoader
            return UniversalTextLoader().load_file(file_path)

        raise ValueError(f"Unsupported file extension: {ext}")