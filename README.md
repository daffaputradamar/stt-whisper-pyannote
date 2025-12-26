# Transcription & Diarization API

Production-ready FastAPI service for speech transcription using Whisper with speaker diarization via Pyannote. Uses **Redis** for job persistence and supports background processing for long audio files.

## Features

- 🎙️ **Speaker Diarization** — Identifies who spoke when using Pyannote
- 📝 **Transcription** — Transcribes speech to text using OpenAI Whisper
- ⏳ **Background Jobs** — Queue long audio for async processing with progress polling
- 🔄 **Redis Persistence** — Jobs survive restarts, no in-memory state
- 🐳 **Docker Ready** — Production deployment with Docker Compose
- 🚀 **GPU Support** — Automatic CUDA/MPS detection

## Quick Start with Docker

```bash
# Clone and enter the project
cd stt-whisper-pyannote

# Copy and edit environment variables
cp .env.example .env
# Edit .env and set your HF_TOKEN

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f api
```

The API will be available at `http://localhost:8000`.

## Local Development

```bash
# Install dependencies (requires Python 3.11+)
pip install -e .

# Start Redis (required)
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Run the API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `HF_TOKEN` | — | Hugging Face token for Pyannote models |
| `WHISPER_MODEL` | `small` | Whisper model size (tiny/base/small/medium/large) |
| `JOBS_DIR` | `./jobs` | Directory for temporary audio files |
| `JOB_TTL_SECONDS` | `86400` | Job expiry time (24h default) |

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

### Synchronous Transcription (short audio)

```bash
curl -X POST -F "file=@audio.wav" http://localhost:8000/transcribe
```

### Background Job (long audio)

**Submit a job:**
```bash
curl -X POST -F "file=@long_audio.wav" http://localhost:8000/jobs
# Returns: {"job_id": "abc123..."}
```

**Poll status:**
```bash
curl http://localhost:8000/jobs/<job_id>
```

**Response:**
```json
{
  "id": "abc123...",
  "status": "transcribing",
  "progress": 45.5,
  "stage_detail": "transcribing 20 segments",
  "result": null,
  "error": null,
  "created_at": "2025-12-26T10:00:00",
  "updated_at": "2025-12-26T10:01:30"
}
```

When `status` is `completed`, the `result` field contains the full transcription.

**Delete a job:**
```bash
curl -X DELETE http://localhost:8000/jobs/<job_id>
```

## GPU Support

For NVIDIA GPU acceleration with Docker:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Uncomment the GPU section in `docker-compose.yml`
3. Rebuild: `docker-compose up -d --build`

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│  FastAPI    │────▶│   Redis     │
│             │◀────│   + Worker  │◀────│   Queue     │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │   Whisper   │
                    │  + Pyannote │
                    └─────────────┘
```

- Jobs are stored in Redis with configurable TTL
- Audio files are saved to disk (configurable `JOBS_DIR`)
- Single background worker processes jobs sequentially to avoid GPU contention
- Progress updates are persisted to Redis for polling

## License

MIT
