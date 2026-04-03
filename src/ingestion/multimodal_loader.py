import os
from typing import Any, List, Union, cast
from langchain_core.documents import Document
import openai
from moviepy.editor import VideoFileClip
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import io
import base64
import math


class MultimodalLoader:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        # Initialize CLIP
        print("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cast(Any, self.model).to(self.device)
        print(f"CLIP model loaded on {self.device}")

    def _encode_image(self, image: Image.Image) -> str:
        """Encode a PIL image to base64 jpeg string."""
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def caption_image(self, image: Image.Image) -> str:
        """
        Creates a detailed sports commentary description of the frame
        using OpenAI Vision.  Returns a plain-text caption string.
        """
        try:
            b64 = self._encode_image(image)
            resp = openai.chat.completions.create(
                model="gpt-5-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are an expert sports AI commentator."
                                "In case of controversial issues, always rely on the collection of rules that are in PDF files."
                                "Describe what is happening in this frame in 2-3 sentences. "
                                "Focus on: the action taking place, player positions and movements, "
                                "tactical situation, and any notable events (goals, fouls, saves, etc.). "
                                "Be specific and vivid, as if you are commentating live. "
                                "Respond with plain text only"
                            )
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }],
                max_completion_tokens=500,
            )
            message_content = resp.choices[0].message.content
            if isinstance(message_content, list):
                content = "".join(
                    part.get("text", "")
                    for part in message_content
                    if isinstance(part, dict)
                ).strip()
            else:
                content = (message_content or "").strip()
            return content or "Visual content"
        except Exception as e:
            print(f"Warning: vision caption failed: {e}")
            return "Visual content (caption unavailable)"

    def get_clip_embedding(self, image_input: Union[str, Image.Image]) -> List[float]:
        """Generates CLIP embedding for an image."""
        if isinstance(image_input, str):
            image = Image.open(image_input)
        else:
            image = image_input
            
        inputs = cast(Any, self.processor)(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = cast(Any, self.model).get_image_features(**inputs)
        
        # Normalize
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0].tolist()

    def process_audio(self, audio_path: str) -> str:
        """Transcribes audio using Whisper."""
        with open(audio_path, "rb") as audio_file:
            transcription = openai.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        return transcription.text

    def process_video(self, video_path: str) -> List[Document]:
        """Processes video by extracting audio and frames."""
        video = VideoFileClip(video_path)
        documents: List[Document] = []
        
        # 1. Extract and transcribe audio
        audio_path = f"{video_path}.mp3"
        try:
            audio = video.audio
            if audio is not None:
                audio.write_audiofile(audio_path, verbose=False, logger=None)
                audio_text = self.process_audio(audio_path)
                documents.append(Document(
                    page_content=f"Video Transcription: {audio_text}",
                    metadata={"source": video_path, "type": "video_audio"}
                ))
        except Exception as e:
            print(f"Error processing video audio: {e}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        
        # 2. Extract frames 
        duration = video.duration
        interval = float(os.getenv("VIDEO_FRAME_INTERVAL_SECONDS", 3))
        total_candidates = int(math.floor(duration / interval)) + 1
        frame_times = [round(t, 2) for t in [i * interval for i in range(total_candidates)]]

        for i, t in enumerate(frame_times, 1):
            frame = video.get_frame(t)
            image = Image.fromarray(frame)
            embedding = self.get_clip_embedding(image)
            caption = self.caption_image(image)

            documents.append(Document(
                page_content=f"{caption} (frame at {t}s)",
                metadata={
                    "source": video_path,
                    "type": "video_frame",
                    "timestamp": t,
                    "embedding": embedding,
                }
            ))
            if i % 5 == 0 or i == len(frame_times):
                print(f"Ingested {i}/{len(frame_times)} frames...")

        video.close()
        return documents

    def load_file(self, file_path: str) -> List[Document]:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower().strip()
        
        if ext in ['.jpg', '.jpeg', '.png']:
            image = Image.open(file_path)
            embedding = self.get_clip_embedding(image)
            caption = self.caption_image(image)
            return [Document(
                page_content=caption,
                metadata={
                    "source": file_path,
                    "type": "image",
                    "embedding": embedding,
                }
            )]
        elif ext in ['.mp3', '.wav', '.m4a']:
            transcription = self.process_audio(file_path)
            return [Document(page_content=transcription, metadata={"source": file_path, "type": "audio"})]
        elif ext in ['.mp4', '.mov', '.avi']:
            return self.process_video(file_path)
        elif ext in ['.pdf', '.txt', '.docx', '.md']:
            # Import here to avoid circular dependencies if any, or just standard import at top
            from src.ingestion.text_loader import UniversalTextLoader
            loader = UniversalTextLoader()
            return loader.load_file(file_path)
        else:
            raise ValueError(f"Unsupported multimodal file extension: {ext}")