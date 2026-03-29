"""
CLAWRION GPU Server on Modal
Orpheus TTS + Whisper STT on a single A10G GPU.

Deploy:  modal deploy server.py
STT:     POST /v1/transcribe (multipart audio file)
TTS:     POST /v1/tts (JSON body with "text", optional "voice")
Health:  GET /health
"""

import modal

# ─── Image ──────────────────────────────────────────────────────
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .uv_pip_install("faster-whisper==1.1.1")
    .uv_pip_install("orpheus-speech==0.1.0", "vllm==0.7.3")
    .uv_pip_install("fastapi", "uvicorn", "httpx")
)

# ─── App ────────────────────────────────────────────────────────
app = modal.App("aris-voice", image=image)
PORT = 8080


# ─── GPU Server ─────────────────────────────────────────────────
@app.function(
    gpu="A10G",
    timeout=3600,
    scaledown_window=60,
    min_containers=0,
    max_containers=1,
    memory=16384,
)
@modal.web_server(port=PORT, startup_timeout=300)
def server():
    """Orpheus TTS + Whisper STT on one GPU."""
    import io
    import os
    import tempfile
    import threading
    import wave
    from fastapi import FastAPI, UploadFile, Form, HTTPException
    from fastapi.responses import JSONResponse, Response
    import uvicorn

    web_app = FastAPI(title="CLAWRION Voice Server")

    whisper_model = None
    tts_model = None
    server_ready = False
    load_stage = "waiting"

    def _load_models():
        nonlocal whisper_model, tts_model, server_ready, load_stage

        try:
            # 1. Load Orpheus TTS
            load_stage = "loading_tts"
            print("Loading Orpheus TTS model...")
            from orpheus_tts import OrpheusModel
            tts_model = OrpheusModel(
                model_name="canopylabs/orpheus-tts-0.1-finetune-prod",
                max_model_len=2048,
            )
            print("Orpheus TTS loaded.")

            # 2. Load Whisper
            load_stage = "loading_whisper"
            print("Loading Whisper medium...")
            from faster_whisper import WhisperModel
            whisper_model = WhisperModel("medium", device="cuda", compute_type="float16")
            print("Whisper loaded.")

            server_ready = True
            load_stage = "ready"
            print("=== Server ready ===")

        except Exception as e:
            print(f"Model loading failed: {e}")
            load_stage = f"failed: {e}"

    threading.Thread(target=_load_models, daemon=True).start()

    # ─── STT ───────────────────────────────────────────────────
    @web_app.post("/v1/transcribe")
    async def transcribe(audio: UploadFile, language: str = Form(default="")):
        if not server_ready:
            raise HTTPException(503, "Server still loading models")

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(await audio.read())
                tmp_path = f.name

            segments, info = whisper_model.transcribe(
                tmp_path,
                language=language or None,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
            )
            text = " ".join([s.text for s in segments]).strip()
            return JSONResponse({"text": text, "language": info.language, "duration": info.duration})
        except Exception as e:
            raise HTTPException(500, str(e))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ─── TTS ───────────────────────────────────────────────────
    @web_app.post("/v1/tts")
    async def tts(request_data: dict):
        if not server_ready:
            raise HTTPException(503, "Server still loading models")

        text = request_data.get("text", "")
        if not text:
            raise HTTPException(400, "text required")

        voice = request_data.get("voice", "tara")

        try:
            syn_tokens = tts_model.generate_speech(
                prompt=text,
                voice=voice,
            )

            # Write tokens to WAV
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                for chunk in syn_tokens:
                    wf.writeframes(chunk)

            return Response(
                content=buf.getvalue(),
                media_type="audio/wav",
                headers={"Content-Disposition": "inline; filename=speech.wav"},
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, str(e))

    @web_app.get("/health")
    def health():
        return {
            "status": "ready" if server_ready else "loading",
            "stage": load_stage,
            "services": ["whisper-medium", "orpheus-tts"],
        }

    uvicorn.run(web_app, host="0.0.0.0", port=PORT)


# ─── Lightweight health check (no GPU) ─────────────────────────
@app.function(image=modal.Image.debian_slim().uv_pip_install("fastapi"), timeout=10)
@modal.fastapi_endpoint(method="GET", label="health")
def health():
    return {"status": "ok", "services": ["whisper-medium", "orpheus-tts"]}
