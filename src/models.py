"""
Shared model registry — singleton CLIP model manager.

CLIP (~1.7 GB) must be loaded exactly once per process and shared
between ingestion (image embedding) and retrieval (text→image search).

Usage:
    from src.models import clip_manager
    embedding = clip_manager.get_image_embedding(pil_image)
    text_emb  = clip_manager.get_text_embedding("a goal")
"""

from __future__ import annotations

import logging
import threading
from typing import Union

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from src.config import cfg

logger = logging.getLogger(__name__)

_lock = threading.Lock()


class CLIPManager:
    """Process-wide singleton that holds the CLIP model and processor."""

    _instance: CLIPManager | None = None

    def __new__(cls) -> CLIPManager:
        if cls._instance is None:
            with _lock:
                if cls._instance is None:  # double-check
                    obj = super().__new__(cls)
                    obj._initialized = False
                    cls._instance = obj
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        with _lock:
            if self._initialized:
                return
            model_name = cfg.clip_model_name
            logger.info("Loading CLIP model '%s' …", model_name)
            self._model = CLIPModel.from_pretrained(model_name)
            self._processor = CLIPProcessor.from_pretrained(model_name)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)  # type: ignore[arg-type]
            logger.info("CLIP loaded on %s", self._device)
            self._initialized = True

    # ── Public API ────────────────────────────────────────────────

    def get_image_embedding(self, image_input: Union[str, Image.Image]) -> list[float]:
        """Return a normalised CLIP embedding for an image (path or PIL)."""
        if isinstance(image_input, str):
            image = Image.open(image_input)
        else:
            image = image_input

        inputs = self._processor(images=image, return_tensors="pt").to(self._device)  # type: ignore[operator]
        with torch.no_grad():
            features = self._model.get_image_features(**inputs)  # type: ignore[operator]
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()[0].tolist()

    def get_text_embedding(self, text: str) -> list[float]:
        """Return a normalised CLIP text embedding for video retrieval."""
        inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(self._device)  # type: ignore[operator]
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)  # type: ignore[operator]
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()[0].tolist()


# Module-level convenience reference
clip_manager = CLIPManager()
