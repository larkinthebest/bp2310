"""
FastAPI web application — Sports AI Commentator.

Architecture:
  - All heavy work (pipeline.query, ingestion) runs in threadpool via asyncio.to_thread.
  - CLIP model is loaded once at startup via the shared CLIPManager singleton.
  - MultimodalLoader is created once at startup and reused (not per-upload).
  - Frame extraction is cached in-memory.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from moviepy.editor import VideoFileClip
from PIL import Image

from src.config import cfg
from src.ingestion.multimodal_loader import MultimodalLoader
from src.rag.pipeline import RAGPipeline
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── Global State ──────────────────────────────────────────────────

DATA_DIR = cfg.data_dir
TRACKING_FILE = cfg.tracking_file
_tracking_lock = threading.Lock()

vector_store: VectorStore = None  # type: ignore
pipeline: RAGPipeline = None  # type: ignore
loader: MultimodalLoader = None  # type: ignore  # created once, reused


# ── Tracking helpers ──────────────────────────────────────────────

def load_processed_files() -> set:
    if os.path.exists(TRACKING_FILE):
        try:
            with open(TRACKING_FILE, "r") as f:
                return set(json.load(f))
        except (json.JSONDecodeError, IOError):
            return set()
    return set()


def save_processed_file(filename: str):
    processed = load_processed_files()
    processed.add(filename)
    with open(TRACKING_FILE, "w") as f:
        json.dump(list(processed), f)


def remove_processed_file(filename: str):
    processed = load_processed_files()
    processed.discard(filename)
    with open(TRACKING_FILE, "w") as f:
        json.dump(list(processed), f)


# ── Frame extraction (with bounded LRU cache) ────────────────────

_MAX_FRAME_CACHE = 200
_frame_cache: OrderedDict[str, str] = OrderedDict()


def extract_frame_base64(video_path: str, timestamp: float) -> str | None:
    cache_key = f"{video_path}:{timestamp}"
    if cache_key in _frame_cache:
        _frame_cache.move_to_end(cache_key)
        return _frame_cache[cache_key]

    video = None
    try:
        video = VideoFileClip(video_path)
        frame = video.get_frame(min(timestamp, video.duration - 0.1))
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=80)
        result = base64.b64encode(buf.getvalue()).decode("utf-8")
        _frame_cache[cache_key] = result
        if len(_frame_cache) > _MAX_FRAME_CACHE:
            _frame_cache.popitem(last=False)
        return result
    except Exception as e:
        logger.warning("Frame extraction failed at %.1fs in %s: %s", timestamp, video_path, e)
        return None
    finally:
        if video:
            video.close()


# ── Ingestion ─────────────────────────────────────────────────────

ingestion_status = {"running": False, "progress": "", "results": None}


def _process_single_file(ldr: MultimodalLoader, vs: VectorStore, data_dir: str, filename: str):
    file_path = os.path.join(data_dir, filename)
    try:
        documents = ldr.load_file(file_path)
        if documents:
            vs.add_documents(documents)
            with _tracking_lock:
                save_processed_file(filename)
            return filename, True, None
        return filename, True, "No documents extracted"
    except Exception as e:
        logger.error("Ingestion failed for %s: %s", filename, e)
        return filename, False, str(e)


def _run_ingestion_for_files(files_to_process: list[str]):
    global ingestion_status
    ingestion_status = {"running": True, "progress": "Starting…", "results": None}

    if not files_to_process:
        ingestion_status = {"running": False, "progress": "No new files", "results": {"total": 0, "succeeded": 0, "failed": 0}}
        return

    max_workers = cfg.max_ingest_workers
    ingestion_status["progress"] = f"Processing {len(files_to_process)} files…"

    start = time.time()
    succeeded = failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_single_file, loader, vector_store, DATA_DIR, fn): fn
            for fn in files_to_process
        }
        for future in as_completed(futures):
            _, ok, _ = future.result()
            if ok:
                succeeded += 1
            else:
                failed += 1
            ingestion_status["progress"] = f"Processed {succeeded + failed}/{len(files_to_process)}"

    elapsed = time.time() - start
    ingestion_status = {
        "running": False,
        "progress": f"Done in {elapsed:.1f}s",
        "results": {"total": len(files_to_process), "succeeded": succeeded, "failed": failed},
    }


def run_ingestion():
    os.makedirs(DATA_DIR, exist_ok=True)
    all_files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    processed = load_processed_files()
    _run_ingestion_for_files([f for f in all_files if f not in processed])


# ── FastAPI app ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, pipeline, loader
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    logger.info("Initializing Vector Store…")
    vector_store = VectorStore()
    logger.info("Initializing RAG Pipeline…")
    pipeline = RAGPipeline(vector_store)
    logger.info("Initializing Multimodal Loader…")
    loader = MultimodalLoader()
    logger.info("Server ready!")
    yield


app = FastAPI(title="Multimodal RAG AI", lifespan=lifespan)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request / Response models ─────────────────────────────────────

class HistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    selected_files: list[str] | None = None
    history: list[HistoryMessage] | None = None


class FrameInfo(BaseModel):
    timestamp: float
    source: str
    caption: str
    image_base64: str | None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    frames: list[FrameInfo]


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    history = None
    if req.history:
        history = [{"role": m.role, "content": m.content} for m in req.history]

    source_filter = req.selected_files if req.selected_files else None

    # Run blocking pipeline in threadpool
    result = await asyncio.to_thread(
        pipeline.query,
        req.question,
        None,   # feedback
        True,   # use_reranking
        source_filter,
        history,
    )

    # Build frame thumbnails
    frames: list[FrameInfo] = []
    seen: set[str] = set()
    for match in result.video_matches:
        meta = match["metadata"]
        source = meta.get("source", "")
        timestamp = meta.get("timestamp", 0)
        caption = meta.get("caption", "")
        key = f"{source}:{timestamp}"
        if key in seen:
            continue
        seen.add(key)

        video_path = source
        if not os.path.isabs(video_path):
            video_path = os.path.join(DATA_DIR, os.path.basename(video_path))

        img_b64 = extract_frame_base64(video_path, timestamp) if os.path.exists(video_path) else None
        frames.append(FrameInfo(
            timestamp=timestamp,
            source=os.path.basename(source),
            caption=caption,
            image_base64=img_b64,
        ))

    frames = frames[:8]
    return ChatResponse(answer=result.answer, sources=result.sources, frames=frames)


@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    if ingestion_status["running"]:
        return JSONResponse({"status": "already_running", "progress": ingestion_status["progress"]}, status_code=409)

    os.makedirs(DATA_DIR, exist_ok=True)
    saved_names = []
    for f in files:
        dest = os.path.join(DATA_DIR, f.filename)
        with open(dest, "wb") as out:
            content = await f.read()
            out.write(content)
        saved_names.append(f.filename)

    processed = load_processed_files()
    to_ingest = [n for n in saved_names if n not in processed]
    if to_ingest:
        thread = threading.Thread(target=_run_ingestion_for_files, args=(to_ingest,), daemon=True)
        thread.start()

    return JSONResponse({"status": "started", "uploaded": saved_names, "ingesting": to_ingest})


@app.post("/api/ingest")
async def ingest():
    if ingestion_status["running"]:
        return JSONResponse({"status": "already_running", "progress": ingestion_status["progress"]})
    thread = threading.Thread(target=run_ingestion, daemon=True)
    thread.start()
    return JSONResponse({"status": "started"})


@app.get("/api/ingest/status")
async def ingest_status_ep():
    return JSONResponse(ingestion_status)


@app.get("/api/files")
async def list_files():
    if not os.path.exists(DATA_DIR):
        return JSONResponse({"files": []})
    processed = load_processed_files()
    files = []
    for f in os.listdir(DATA_DIR):
        if os.path.isfile(os.path.join(DATA_DIR, f)):
            files.append({"name": f, "processed": f in processed})
    return JSONResponse({"files": files})


@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    with _tracking_lock:
        remove_processed_file(filename)
    return JSONResponse({"status": "deleted", "filename": filename})


@app.get("/api/video/{filename}")
async def serve_video(filename: str):
    video_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(video_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(video_path, media_type="video/mp4", filename=filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
