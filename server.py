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
    .pip_install("fastapi>=0.115.0", "uvicorn", "httpx>=0.27.0")
)

# ─── App ────────────────────────────────────────────────────────
app = modal.App("aris-voice", image=image)
model_volume = modal.Volume.from_name("aris-voice-models", create_if_missing=True)

MODEL_DIR = "/models"
FISH_CHECKPOINT = f"{MODEL_DIR}/s2-pro"
PORT = 8080

# ─── Globals ────────────────────────────────────────────────────
whisper_model = None
server_ready = False


def load_whisper():
    global whisper_model
    if whisper_model is None:
        from faster_whisper import WhisperModel
        print("Loading Whisper large-v3...")
        whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        print("Whisper loaded.")
    return whisper_model


def download_fish_models():
    """Download Fish Speech model weights if not cached."""
    import os
    if os.path.exists(FISH_CHECKPOINT):
        print(f"S2 Pro models found at {FISH_CHECKPOINT}")
        return
    print("Downloading S2 Pro weights (~8GB, first run)...")
    subprocess.run(
        ["huggingface-cli", "download", "fishaudio/s2-pro",
         "--local-dir", FISH_CHECKPOINT],
        check=True,
    )
    model_volume.commit()
    print("S2 Pro cached.")


def start_fish_server():
    """Start Fish Speech API server as background process."""
    cmd = [
        "python", "/app/fish-speech/tools/api_server.py",
        "--llama-checkpoint-path", FISH_CHECKPOINT,
        "--decoder-checkpoint-path", f"{FISH_CHECKPOINT}/codec.pth",
        "--listen", "127.0.0.1:8081",
        "--half",
    ]
    print("Starting Fish Speech API server on :8081")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def wait_for_fish_server(timeout=120):
    """Wait for Fish Speech server to be ready."""
    import time
    import httpx
    print("Waiting for Fish Speech server to be ready...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get("http://127.0.0.1:8081/health", timeout=2.0)
            if resp.status_code == 200:
                print("Fish Speech server ready.")
                return True
        except Exception:
            pass
        time.sleep(2)
    print(f"WARNING: Fish Speech server not ready after {timeout}s")
    return False


# ─── Single GPU Server ──────────────────────────────────────────
@app.function(
    gpu="A10G",
    timeout=3600,
    scaledown_window=15,
    volumes={MODEL_DIR: model_volume},
    memory=16384,
)
@modal.web_server(port=PORT, startup_timeout=300)
def server():
    """Combined Whisper STT + Fish Speech TTS on one GPU."""
    import os
    import tempfile
    import threading
    from fastapi import FastAPI, UploadFile, Form, HTTPException
    from fastapi.responses import JSONResponse, Response
    import httpx
    import uvicorn

    web_app = FastAPI(title="Aris Voice Server")

    # ─── Background model loading (non-blocking) ──────────────────
    def _load_models():
        global server_ready
        try:
            # 1. Download Fish Speech models if needed
            download_fish_models()

            # 2. Start Fish Speech server
            start_fish_server()

            # 3. Wait for Fish Speech to be ready
            wait_for_fish_server()

            # 4. Pre-load Whisper
            load_whisper()

            server_ready = True
            print("=== All models loaded. Server ready. ===")
        except Exception as e:
            print(f"Model loading failed: {e}")

    # Start model loading in background thread so uvicorn starts immediately
    loader_thread = threading.Thread(target=_load_models, daemon=True)
    loader_thread.start()

    # ─── STT: Whisper ─────────────────────────────────────────────
    @web_app.post("/v1/transcribe")
    async def transcribe(audio: UploadFile, language: str = Form(default="")):
        if not server_ready:
            raise HTTPException(status_code=503, detail="Server still loading models")

        tmp_path = None
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

            return JSONResponse({
                "text": text,
                "language": info.language,
                "duration": info.duration,
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ─── TTS: Fish Speech ────────────────────────────────────────
    @web_app.post("/v1/tts")
    async def tts(request_data: dict):
        if not server_ready:
            raise HTTPException(status_code=503, detail="Server still loading models")

        try:
            text = request_data.get("text", "")
            if not text:
                raise HTTPException(status_code=400, detail="text required")

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

            fish_url = "http://127.0.0.1:8081/v1/tts"

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
        return {
            "status": "ready" if server_ready else "loading",
            "services": ["whisper-large-v3", "fish-speech-s2-pro"],
        }

    uvicorn.run(web_app, host="0.0.0.0", port=PORT)


# ─── Health endpoint (outside GPU) ──────────────────────────────
@app.function(image=modal.Image.debian_slim(), timeout=10)
@modal.web_endpoint(method="GET", label="health")
def health():
    return {"status": "ok", "services": ["whisper-large-v3", "fish-speech-s2-pro"]}
