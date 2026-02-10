import os
from typing import List, Union, Tuple, Optional, Dict
from langchain_core.documents import Document
import openai
from moviepy.editor import VideoFileClip
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import io
import base64
import math
import json

class MultimodalLoader:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        # Initialize CLIP
        print("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"CLIP model loaded on {self.device}")

    def _encode_image(self, image: Image.Image) -> str:
        """Encode a PIL image to base64 jpeg string."""
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def caption_image(self, image: Image.Image) -> Tuple[str, Optional[Dict[str, str]]]:
        """
        Creates a short, descriptive caption using OpenAI Vision.
        Additionally, if a scoreboard is visible, ask for structured fields.
        Returns (caption, scoreboard_dict|None).
        """
        try:
            b64 = self._encode_image(image)
            resp = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are assisting sports video RAG. "
                                "If a scoreboard is visible, return JSON with keys: "
                                "caption (one sentence), home_team, home_score, home_shots, "
                                "away_team, away_score, away_shots, period, clock. "
                                "Use null for missing fields. Respond with JSON only."
                            )
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}" }}
                    ]
                }],
                max_tokens=120,
            )
            content = resp.choices[0].message.content.strip()
            # Strip common markdown fences
            if content.startswith("```"):
                content = content.strip("`")
                # remove optional json label
                if content.lower().startswith("json"):
                    content = content[4:].strip()
            # Fallback: extract first JSON object if fences remain
            if not content.lstrip().startswith("{"):
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    content = content[start:end+1]
            try:
                data = json.loads(content)
                caption = data.get("caption", "").strip() or "Visual content"
                scoreboard = {k: v for k, v in data.items() if k != "caption"}
                # normalize empty strings to None
                scoreboard = {k: (v if v not in ("", None) else None) for k, v in scoreboard.items()}
                return caption, scoreboard
            except Exception as parse_err:
                print(f"Warning: vision JSON parse failed: {parse_err}; raw={content}")
                return content, None
        except Exception as e:
            print(f"Warning: vision caption failed: {e}")
            return "Visual content (caption unavailable)", None

    def get_clip_embedding(self, image_input: Union[str, Image.Image]) -> List[float]:
        """Generates CLIP embedding for an image."""
        if isinstance(image_input, str):
            image = Image.open(image_input)
        else:
            image = image_input
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
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
        documents = []
        
        # 1. Extract and transcribe audio
        audio_path = f"{video_path}.mp3"
        try:
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
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
        
        # 2. Extract frames (e.g., every 2 seconds)
        duration = video.duration
        interval = float(os.getenv("VIDEO_FRAME_INTERVAL_SECONDS", 2))
        total_candidates = int(math.floor(duration / interval)) + 1
        frame_times = [round(t, 2) for t in [i * interval for i in range(total_candidates)]]

        for i, t in enumerate(frame_times, 1):
            frame = video.get_frame(t)
            image = Image.fromarray(frame)
            embedding = self.get_clip_embedding(image)
            caption, scoreboard = self.caption_image(image)

            documents.append(Document(
                page_content=f"{caption} (frame at {t}s)", 
                metadata={
                    "source": video_path, 
                    "type": "video_frame", 
                    "timestamp": t,
                    "embedding": embedding, # Store embedding here to be handled by VectorStore
                    "scoreboard": json.dumps(scoreboard) if scoreboard else None
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
            caption, scoreboard = self.caption_image(image)
            return [Document(
                page_content=caption, 
                metadata={
                    "source": file_path, 
                    "type": "image",
                    "embedding": embedding,
                    "scoreboard": scoreboard
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
