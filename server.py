"""
Aris Voice GPU Server on Modal
Combines Whisper STT + Fish Speech S2 Pro TTS on a single GPU.

Deploy:  modal deploy server.py
STT:     curl -X POST https://your-org--aris-voice-server.modal.run/v1/transcribe \
           -F "audio=@test.wav" -F "language=en"
TTS:     curl -X POST https://your-org--aris-voice-server.modal.run/v1/tts \
           -H "Content-Type: application/json" \
           -d '{"text": "Hello!", "format": "wav"}' --output test.wav
Health:  curl https://your-org--aris-voice-server.modal.run/health
"""

import subprocess
import modal

# ─── Image ──────────────────────────────────────────────────────
# Base: CUDA + Python
# Then layer: Whisper (pip) + Fish Speech (from source)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "ffmpeg", "build-essential")
    # Whisper STT
    .pip_install("faster-whisper>=1.1.0", "python-multipart>=0.0.12")
    # Fish Speech TTS
    .run_commands(
        "git clone --depth 1 https://github.com/fishaudio/fish-speech.git /app/fish-speech"
    )
    .workdir("/app/fish-speech")
    .pip_install(
        "sglang[all]>=0.5.0",
        "torch>=2.6.0",
        "torchaudio>=2.6.0",
        "transformers>=4.45.0",
        "huggingface_hub>=0.24.0",
        "accelerate",
        "einops",
        "ormsgpack",
        "websockets",
    )
    .run_commands("pip install -e '.[server]'")
    .workdir("/app")
    .pip_install("fastapi>=0.115.0", "uvicorn")
)

# ─── App ────────────────────────────────────────────────────────
app = modal.App("aris-voice", image=image)
model_volume = modal.Volume.from_name("aris-voice-models", create_if_missing=True)

MODEL_DIR = "/models"
FISH_CHECKPOINT = f"{MODEL_DIR}/s2-pro"
PORT = 8080

# ─── Globals ────────────────────────────────────────────────────
whisper_model = None


def load_whisper():
    global whisper_model
    if whisper_model is None:
        from faster_whisper import WhisperModel
        print("Loading Whisper large-v3...")
        whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        print("Whisper loaded.")
    return whisper_model


# ─── Single GPU Server ──────────────────────────────────────────
@app.function(
    gpu="A10G",
    timeout=3600,
    scaledown_window=15,
    volumes={MODEL_DIR: model_volume},
    memory=16384,
)
@modal.web_server(port=PORT, startup_timeout=180)
def server():
    """Combined Whisper STT + Fish Speech TTS on one GPU."""
    import os
    import tempfile
    from fastapi import FastAPI, UploadFile, Form, HTTPException
    from fastapi.responses import JSONResponse, Response
    import uvicorn

    web_app = FastAPI(title="Aris Voice Server")

    @web_app.on_event("startup")
    def startup():
        # Pre-load Whisper
        load_whisper()
        # Check Fish Speech models
        if not os.path.exists(FISH_CHECKPOINT):
            print("Downloading S2 Pro weights (~8GB, first run)...")
            subprocess.run(
                ["huggingface-cli", "download", "fishaudio/s2-pro",
                 "--local-dir", FISH_CHECKPOINT],
                check=True,
            )
            model_volume.commit()
            print("S2 Pro cached.")

    # ─── STT: Whisper ───────────────────────────────────────────
    @web_app.post("/v1/transcribe")
    async def transcribe(audio: UploadFile, language: str = Form(default="")):
        """Transcribe audio to text via Whisper large-v3."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(await audio.read())
                tmp_path = f.name

            m = load_whisper()
            segments, info = m.transcribe(
                tmp_path,
                language=language or None,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200,
                ),
            )

            text = " ".join([s.text for s in segments]).strip()
            os.unlink(tmp_path)

            return JSONResponse({
                "text": text,
                "language": info.language,
                "duration": info.duration,
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ─── TTS: Fish Speech ──────────────────────────────────────
    @web_app.post("/v1/tts")
    async def tts(request_data: dict):
        """Generate speech via Fish Speech S2 Pro.

        Accepts: {text, format?, reference_id?, temperature?, top_p?}
        Returns: WAV/PCM audio bytes
        """
        try:
            import io
            text = request_data.get("text", "")
            if not text:
                raise HTTPException(status_code=400, detail="text required")

            # Build request to Fish Speech internal API
            import ormsgpack
            from websockets.asyncio.client import connect as ws_connect

            # Use the Fish Speech server's WebSocket API
            payload = {
                "text": text,
                "format": request_data.get("format", "wav"),
                "normalize": request_data.get("normalize", True),
                "temperature": request_data.get("temperature", 0.7),
                "top_p": request_data.get("top_p", 0.7),
                "repetition_penalty": request_data.get("repetition_penalty", 1.2),
                "chunk_length": request_data.get("chunk_length", 200),
            }
            if request_data.get("reference_id"):
                payload["reference_id"] = request_data["reference_id"]

            # Call Fish Speech via HTTP POST (local server started by tools/api_server.py)
            # The Fish Speech API server runs as a background process
            fish_url = "http://127.0.0.1:8081/v1/tts"

            import httpx
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(fish_url, json=payload)
                if resp.status_code != 200:
                    raise HTTPException(
                        status_code=resp.status_code,
                        detail=f"Fish Speech error: {resp.text}",
                    )
                return Response(
                    content=resp.content,
                    media_type="audio/wav",
                    headers={"Content-Disposition": "inline; filename=speech.wav"},
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.get("/health")
    def health():
        return {"status": "ok", "services": ["whisper-large-v3", "fish-speech-s2-pro"]}

    # Start Fish Speech API server in background (port 8081, separate from our 8080)
    fish_cmd = [
        "python", "/app/fish-speech/tools/api_server.py",
        "--llama-checkpoint-path", FISH_CHECKPOINT,
        "--decoder-checkpoint-path", f"{FISH_CHECKPOINT}/codec.pth",
        "--listen", "127.0.0.1:8081",
        "--half",
    ]
    print("Starting Fish Speech API server on :8081")
    subprocess.Popen(fish_cmd)

    uvicorn.run(web_app, host="0.0.0.0", port=PORT)


# ─── Health endpoint (outside GPU) ──────────────────────────────
@app.function(image=modal.Image.debian_slim(), timeout=10)
@modal.web_endpoint(method="GET", label="health")
def health():
    return {"status": "ok", "services": ["whisper-large-v3", "fish-speech-s2-pro"]}
