import io
import asyncio
import uuid
import json
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pyannote.audio import Pipeline
import whisper
import numpy as np
import soundfile as sf
import gc
import os
import shutil
import torch
import torch.nn.functional as F
from typing import Optional, Literal
from pydantic import BaseModel
from pyannote.audio.core.task import Specifications, Problem, Resolution
import redis.asyncio as redis

torch.serialization.add_safe_globals([Specifications, Problem, Resolution])

# ----- Configuration -----
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
JOBS_DIR = os.getenv("JOBS_DIR", os.path.join(os.getcwd(), "jobs"))
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "86400"))  # 24 hours default
REDIS_QUEUE_KEY = "transcription:queue"
REDIS_JOB_PREFIX = "transcription:job:"
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "3"))

# ----- Pydantic Models -----


class TranscriptionSegment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str


class TranscriptionResponse(BaseModel):
    segments: list[TranscriptionSegment]
    full_text: str


class JobCreateResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    id: str
    status: Literal["queued", "diarizing", "transcribing", "completed", "error"]
    progress: float
    stage_detail: Optional[str] = None
    result: Optional[TranscriptionResponse] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# ----- Globals (ML models and Redis) -----
pipeline: Optional[Pipeline] = None
whisper_model = None
device = torch.device("cpu")
redis_client: Optional[redis.Redis] = None
worker_tasks: list[asyncio.Task] = []


# ----- Helpers -----


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_audio_bytes(data: bytes, target_sr: int = 16000) -> tuple[torch.Tensor, int, np.ndarray]:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, n_samples)

    if sr != target_sr:
        target_len = int(round(waveform.shape[-1] * target_sr / sr))
        waveform = F.interpolate(
            waveform.unsqueeze(0),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).squeeze(0)
        sr = target_sr

    waveform_device = waveform.to(device)
    audio_np = waveform.cpu().squeeze(0).numpy()
    return waveform_device, sr, audio_np


# ----- Redis Job Management -----


def _serialize_job(job: dict) -> str:
    """Serialize job dict to JSON, handling datetime and Pydantic models."""
    def default(o):
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, BaseModel):
            return o.model_dump()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")
    return json.dumps(job, default=default)


def _deserialize_job(data: str) -> dict:
    """Deserialize JSON back to job dict."""
    job = json.loads(data)
    job["created_at"] = datetime.fromisoformat(job["created_at"])
    job["updated_at"] = datetime.fromisoformat(job["updated_at"])
    if job.get("result"):
        job["result"] = TranscriptionResponse(**job["result"])
    return job


async def save_job(job: dict):
    """Save job to Redis with TTL."""
    if redis_client is None:
        raise RuntimeError("Redis not connected")
    key = f"{REDIS_JOB_PREFIX}{job['id']}"
    await redis_client.setex(key, JOB_TTL_SECONDS, _serialize_job(job))


async def get_job(job_id: str) -> Optional[dict]:
    """Get job from Redis."""
    if redis_client is None:
        raise RuntimeError("Redis not connected")
    key = f"{REDIS_JOB_PREFIX}{job_id}"
    data = await redis_client.get(key)
    if data is None:
        return None
    return _deserialize_job(data)


async def enqueue_job(file: UploadFile) -> str:
    """Save file to disk and enqueue job in Redis."""
    if redis_client is None:
        raise RuntimeError("Redis not connected")

    job_id = uuid.uuid4().hex

    # Create job-specific directory
    job_subdir = os.path.join(JOBS_DIR, f"job_{job_id}")
    os.makedirs(job_subdir, exist_ok=True)
    filename = file.filename or f"audio_{job_id}"
    dest_path = os.path.join(job_subdir, filename)

    # Write file to disk
    data = await file.read()
    with open(dest_path, "wb") as f:
        f.write(data)

    now = datetime.utcnow()
    job = {
        "id": job_id,
        "status": "queued",
        "progress": 0.0,
        "stage_detail": "waiting in queue",
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "file_path": dest_path,
    }

    await save_job(job)
    await redis_client.rpush(REDIS_QUEUE_KEY, job_id)
    return job_id


def cleanup_job_file(file_path: Optional[str]):
    """Remove temporary job files."""
    if file_path and os.path.exists(file_path):
        parent = os.path.dirname(file_path)
        try:
            shutil.rmtree(parent)
        except Exception:
            try:
                os.remove(file_path)
            except Exception:
                pass


# ----- Background Worker -----


async def job_worker(worker_id: int = 0):
    """Background worker that polls Redis queue and processes jobs."""
    assert redis_client is not None
    print(f"Job worker #{worker_id} started, waiting for jobs...")

    while True:
        try:
            # Blocking pop with 5 second timeout
            result = await redis_client.blpop(REDIS_QUEUE_KEY, timeout=5)
            if result is None:
                continue

            _, job_id_bytes = result
            job_id = job_id_bytes.decode() if isinstance(job_id_bytes, bytes) else job_id_bytes

            job = await get_job(job_id)
            if job is None:
                print(f"[worker {worker_id}] Job {job_id} not found in Redis, skipping")
                continue

            try:
                await process_job(job)
            except Exception as e:
                job["status"] = "error"
                job["error"] = str(e)
                job["updated_at"] = datetime.utcnow()
                await save_job(job)
                print(f"[worker {worker_id}] Job {job_id} failed: {e}")

        except asyncio.CancelledError:
            print(f"Job worker #{worker_id} cancelled")
            break
        except Exception as e:
            print(f"Worker #{worker_id} error: {e}")
            await asyncio.sleep(1)


async def process_job(job: dict):
    """Process a single transcription job."""
    if pipeline is None or whisper_model is None:
        raise RuntimeError("Models not loaded yet")

    job_id = job["id"]
    print(f"Processing job {job_id}")

    job["status"] = "diarizing"
    job["stage_detail"] = "running speaker diarization"
    job["updated_at"] = datetime.utcnow()
    await save_job(job)

    # Load audio from saved file
    file_path = job["file_path"]
    with open(file_path, "rb") as f:
        data = f.read()

    waveform, sample_rate, audio_np = load_audio_bytes(data, target_sr=16000)

    # Run diarization
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # Get segments list
    segments_iter = getattr(diarization, "speaker_diarization", None)
    if segments_iter is None:
        segments_list = [(turn, speaker) for (turn, _, speaker) in diarization.itertracks(yield_label=True)]
    else:
        segments_list = list(segments_iter)

    total = max(len(segments_list), 1)
    job["status"] = "transcribing"
    job["stage_detail"] = f"transcribing {total} segments"
    job["progress"] = 0.0
    job["updated_at"] = datetime.utcnow()
    await save_job(job)

    segments: list[TranscriptionSegment] = []
    full_text_parts: list[str] = []

    for idx, (turn, speaker) in enumerate(segments_list, start=1):
        start_sample = int(turn.start * sample_rate)
        end_sample = int(turn.end * sample_rate)
        segment_audio = audio_np[start_sample:end_sample]
        result = whisper_model.transcribe(segment_audio, fp16=False, language="pt")
        text = result["text"].strip()

        seg = TranscriptionSegment(
            start=float(turn.start),
            end=float(turn.end),
            speaker=str(speaker),
            text=text,
        )
        segments.append(seg)
        full_text_parts.append(f"[{turn.start:.3f} -- {turn.end:.3f}] {speaker}: {text}")

        # Update progress
        job["progress"] = round(100.0 * idx / total, 2)
        job["updated_at"] = datetime.utcnow()
        await save_job(job)
        await asyncio.sleep(0)  # yield control

        del result
        del segment_audio
        gc.collect()

    job["result"] = TranscriptionResponse(
        segments=segments,
        full_text="\n".join(full_text_parts),
    )
    job["status"] = "completed"
    job["stage_detail"] = "done"
    job["updated_at"] = datetime.utcnow()
    await save_job(job)

    # Cleanup temp files
    cleanup_job_file(job.get("file_path"))
    print(f"Job {job_id} completed")


# ----- Lifespan -----


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global pipeline, whisper_model, device, redis_client, worker_tasks

    # Connect to Redis
    redis_client = redis.from_url(REDIS_URL, decode_responses=False)
    await redis_client.ping()
    print(f"Connected to Redis at {REDIS_URL}")

    # Setup jobs directory
    os.makedirs(JOBS_DIR, exist_ok=True)

    # Load ML models
    device = detect_device()

    hf_token = os.getenv("HF_TOKEN", "-")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    ).to(device)

    whisper_model_name = os.getenv("WHISPER_MODEL", "small")
    whisper_model = whisper.load_model(whisper_model_name, device=str(device))
    print(f"Models loaded successfully on {device}!")

    # Start background workers (configurable concurrency)
    for i in range(max(WORKER_CONCURRENCY, 1)):
        worker_tasks.append(asyncio.create_task(job_worker(i)))

    yield

    # Shutdown
    for t in worker_tasks:
        t.cancel()
    for t in worker_tasks:
        try:
            await t
        except asyncio.CancelledError:
            pass
    worker_tasks.clear()

    if redis_client:
        await redis_client.close()
    print("Shutdown complete")


# ----- FastAPI App -----

app = FastAPI(
    title="Transcription & Diarization API",
    description="Production-ready API for audio transcription with speaker diarization using Whisper and Pyannote. Uses Redis for job persistence.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Transcription & Diarization API is running"}


@app.get("/health")
async def health_check():
    """Check if models and Redis are ready."""
    redis_ok = False
    if redis_client:
        try:
            await redis_client.ping()
            redis_ok = True
        except Exception:
            pass

    return {
        "status": "healthy" if (pipeline and whisper_model and redis_ok) else "degraded",
        "pipeline_loaded": pipeline is not None,
        "whisper_loaded": whisper_model is not None,
        "redis_connected": redis_ok,
    }


@app.post("/jobs", response_model=JobCreateResponse)
async def create_job(file: UploadFile = File(...)):
    """Submit a long audio transcription job. Returns a job id for polling."""
    if pipeline is None or whisper_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    try:
        job_id = await enqueue_job(file)
        return JobCreateResponse(job_id=job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll job status. When completed, includes the transcription result."""
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        id=job["id"],
        status=job["status"],
        progress=float(job["progress"]),
        stage_detail=job.get("stage_detail"),
        result=job.get("result"),
        error=job.get("error"),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis not connected")

    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Cleanup files
    cleanup_job_file(job.get("file_path"))

    # Remove from Redis
    key = f"{REDIS_JOB_PREFIX}{job_id}"
    await redis_client.delete(key)

    return {"status": "deleted", "job_id": job_id}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """Synchronous transcription endpoint for shorter audio files."""
    if pipeline is None or whisper_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    try:
        data = await file.read()
        waveform, sample_rate, audio_np = load_audio_bytes(data, target_sr=16000)
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

        segments: list[TranscriptionSegment] = []
        full_text_parts: list[str] = []

        segments_iter = getattr(diarization, "speaker_diarization", None)
        if segments_iter is None:
            iterable = [(turn, speaker) for (turn, _, speaker) in diarization.itertracks(yield_label=True)]
        else:
            iterable = segments_iter

        for turn, speaker in iterable:
            start_sample = int(turn.start * sample_rate)
            end_sample = int(turn.end * sample_rate)
            segment_audio = audio_np[start_sample:end_sample]
            result = whisper_model.transcribe(segment_audio, fp16=False, language="id")
            text = result["text"].strip()
            segment = TranscriptionSegment(
                start=float(turn.start),
                end=float(turn.end),
                speaker=str(speaker),
                text=text,
            )
            segments.append(segment)
            full_text_parts.append(f"[{turn.start:.3f} -- {turn.end:.3f}] {speaker}: {text}")
            del result
            del segment_audio
            gc.collect()

        return TranscriptionResponse(
            segments=segments,
            full_text="\n".join(full_text_parts),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)