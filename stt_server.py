"""
Whisper STT Server on Modal

Deploy:  modal deploy stt_server.py
Test:    curl -X POST https://your-org--whisper-stt-server.modal.run/v1/transcribe \
           -F "audio=@test.wav" \
           -F "language=en"
"""

import io
import modal

# ─── Image ──────────────────────────────────────────────────────
stt_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "faster-whisper>=1.1.0",
        "fastapi>=0.115.0",
        "python-multipart>=0.0.12",
    )
)

app = modal.App("whisper-stt", image=stt_image)

# ─── Globals (loaded once per container) ────────────────────────
model = None


def load_model():
    global model
    if model is None:
        from faster_whisper import WhisperModel
        print("Loading Whisper large-v3 model...")
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        print("Whisper model loaded.")
    return model


@app.function(
    gpu="A10G",
    timeout=300,
    scaledown_window=15,
    memory=8192,
)
@modal.web_server(port=8000, startup_timeout=60)
def server():
    import json
    import tempfile
    import uvicorn
    from fastapi import FastAPI, UploadFile, Form, HTTPException
    from fastapi.responses import JSONResponse

    web_app = FastAPI(title="Whisper STT")

    @web_app.on_event("startup")
    def startup():
        load_model()

    @web_app.post("/v1/transcribe")
    async def transcribe(
        audio: UploadFile,
        language: str = Form(default=""),
    ):
        """Transcribe audio file to text."""
        try:
            # Save uploaded audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                content = await audio.read()
                f.write(content)
                tmp_path = f.name

            # Transcribe
            m = load_model()
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

    @web_app.get("/health")
    def health():
        return {"status": "ok", "model": "whisper-large-v3"}

    uvicorn.run(web_app, host="0.0.0.0", port=8000)


@app.function(image=modal.Image.debian_slim(), timeout=10)
@modal.web_endpoint(method="GET", label="whisper-health")
def health():
    return {"status": "ok", "model": "whisper-large-v3"}
