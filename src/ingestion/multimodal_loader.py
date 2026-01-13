import os
from typing import List, Union
from langchain_core.documents import Document
import openai
from moviepy.editor import VideoFileClip
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

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
        
        # 2. Extract frames (e.g., every 5 seconds)
        duration = video.duration
        for t in range(0, int(duration), 5):
            frame = video.get_frame(t)
            image = Image.fromarray(frame)
            embedding = self.get_clip_embedding(image)
            
            # For Pinecone, we store the embedding directly. 
            # Since LangChain Document is text-based, we'll store a placeholder text 
            # and attach the embedding in metadata (or handle it in VectorStore).
            # Here we just return a Document with special metadata.
            documents.append(Document(
                page_content=f"Video Frame at {t}s", 
                metadata={
                    "source": video_path, 
                    "type": "video_frame", 
                    "timestamp": t,
                    "embedding": embedding # Store embedding here to be handled by VectorStore
                }
            ))
            
        return documents

    def load_file(self, file_path: str) -> List[Document]:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower().strip()
        
        if ext in ['.jpg', '.jpeg', '.png']:
            embedding = self.get_clip_embedding(file_path)
            return [Document(
                page_content="Image", 
                metadata={
                    "source": file_path, 
                    "type": "image",
                    "embedding": embedding
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
