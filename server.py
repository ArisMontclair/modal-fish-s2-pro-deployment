"""
Aris Voice GPU Server on Modal
Combines Whisper STT + Fish Speech S2 Pro TTS on a single GPU.

Deploy:  modal deploy server.py
STT:     POST /v1/transcribe (multipart audio file)
TTS:     POST /v1/tts (JSON body with "text")
Health:  GET /health
"""

import subprocess
import modal

# ─── Image ──────────────────────────────────────────────────────
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "ffmpeg", "clang", "libportaudio2", "portaudio19-dev")
    .uv_pip_install("faster-whisper==1.1.1")
    .run_commands("git clone --depth 1 --branch v2.0.0 https://github.com/fishaudio/fish-speech.git /app/fish-speech")
    .workdir("/app/fish-speech")
    .run_commands("pip install -e '.[server]'")
    # Patch: fish-speech torchaudio circular import bug with torch 2.8
    # list_audio_backends() was removed in torchaudio 2.8, the except block
    # re-imports torchaudio.io which causes a circular import → UnboundLocalError
    .run_commands(
        """python3 -c "
import pathlib
p = pathlib.Path('/app/fish-speech/fish_speech/inference_engine/reference_loader.py')
src = p.read_text()
old = '''        try:
            backends = torchaudio.list_audio_backends()
            if \\\"ffmpeg\\\" in backends:
                self.backend = \\\"ffmpeg\\\"
            else:
                self.backend = \\\"soundfile\\\"
        except AttributeError:
            # torchaudio 2.9+ removed list_audio_backends()
            # Try ffmpeg first, fallback to soundfile
            try:
                import torchaudio.io._load_audio_fileobj  # noqa: F401

                self.backend = \\\"ffmpeg\\\"
            except (ImportError, ModuleNotFoundError):
                self.backend = \\\"soundfile\\\"'''
new = '''        try:
            backends = torchaudio.list_audio_backends()
            self.backend = \\\"ffmpeg\\\" if \\\"ffmpeg\\\" in backends else \\\"soundfile\\\"
        except (AttributeError, UnboundLocalError):
            self.backend = \\\"soundfile\\\"'''
p.write_text(src.replace(old, new))
print('Patched reference_loader.py')
"
"""
    )
    .workdir("/app")
    .uv_pip_install("fastapi", "uvicorn", "httpx")
    .run_commands("huggingface-cli download fishaudio/s2-pro --local-dir /models/s2-pro")
)

# ─── App ────────────────────────────────────────────────────────
app = modal.App("aris-voice", image=image)

FISH_CHECKPOINT = "/models/s2-pro"
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
    """Combined Whisper STT + Fish Speech TTS on one GPU."""
    import os
    import tempfile
    import threading
    import time
    from fastapi import FastAPI, UploadFile, Form, HTTPException
    from fastapi.responses import JSONResponse, Response
    import httpx
    import uvicorn

    web_app = FastAPI(title="Aris Voice Server")

    whisper_model = None
    fish_proc = None
    server_ready = False
    load_stage = "waiting"  # waiting → preflight → loading_fish → loading_whisper → ready / failed

    def _load_models():
        nonlocal whisper_model, fish_proc, server_ready, load_stage

        # ─── Preflight checks (no GPU yet) ────────────────────
        load_stage = "preflight"
        # 1. Verify model weights exist
        if not os.path.exists(FISH_CHECKPOINT):
            print(f"FATAL: Fish Speech weights dir not found at {FISH_CHECKPOINT}")
            load_stage = "failed: missing weights"
            return
        if not os.path.exists(os.path.join(FISH_CHECKPOINT, "codec.pth")):
            print(f"FATAL: Fish Speech codec not found at {FISH_CHECKPOINT}/codec.pth")
            load_stage = "failed: missing codec"
            return
        if not os.path.exists(os.path.join(FISH_CHECKPOINT, "model.safetensors.index.json")):
            print(f"FATAL: Fish Speech model index not found at {FISH_CHECKPOINT}/model.safetensors.index.json")
            load_stage = "failed: missing model index"
            return
        print(f"Model weights verified: {FISH_CHECKPOINT}")

        # 2. Verify fish-speech imports cleanly
        try:
            import fish_speech
            print("fish-speech imported OK")
        except Exception as e:
            print(f"FATAL: fish-speech import failed: {e}")
            load_stage = f"failed: {e}"
            return

        # 3. Verify torchaudio works (was previous blocker)
        try:
            import torchaudio
            print(f"torchaudio {torchaudio.__version__} imported OK")
        except Exception as e:
            print(f"FATAL: torchaudio import failed: {e}")
            load_stage = f"failed: {e}"
            return

        print("All preflight checks passed. Loading models onto GPU...")

        try:
            # 1. Start Fish Speech server
            load_stage = "loading_fish"
            fish_cmd = [
                "python", "-m", "tools.api_server",
                "--llama-checkpoint-path", FISH_CHECKPOINT,
                "--decoder-checkpoint-path", f"{FISH_CHECKPOINT}/codec.pth",
                "--listen", "127.0.0.1:8081",
                "--half",
            ]
            print(f"Starting Fish Speech: {' '.join(fish_cmd)}")
            fish_proc = subprocess.Popen(
                fish_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd="/app/fish-speech",
            )

            def _log_stderr():
                for line in fish_proc.stderr:
                    print(f"[FishSpeech] {line.decode(errors='replace').rstrip()}")
            threading.Thread(target=_log_stderr, daemon=True).start()

            # 2. Wait for Fish Speech
            fish_ok = False
            for i in range(60):
                try:
                    resp = httpx.get("http://127.0.0.1:8081/health", timeout=2.0)
                    if resp.status_code == 200:
                        fish_ok = True
                        print("Fish Speech ready.")
                        break
                except Exception:
                    pass
                time.sleep(2)

            if not fish_ok:
                print("Fish Speech failed to start after 120s")

            # 3. Load Whisper
            load_stage = "loading_whisper"
            from faster_whisper import WhisperModel
            print("Loading Whisper large-v3...")
            whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
            print("Whisper loaded.")

            server_ready = True
            load_stage = "ready"
            print(f"=== Server ready (Fish: {'OK' if fish_ok else 'FAILED'}) ===")

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

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post("http://127.0.0.1:8081/v1/tts", json=payload)
                if resp.status_code != 200:
                    raise HTTPException(resp.status_code, f"Fish Speech error: {resp.text}")
                return Response(
                    content=resp.content,
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
            "services": ["whisper-large-v3", "fish-speech-s2-pro"],
        }

    uvicorn.run(web_app, host="0.0.0.0", port=PORT)


# ─── Lightweight health check (no GPU) ─────────────────────────
@app.function(image=modal.Image.debian_slim().pip_install("fastapi"), timeout=10)
@modal.fastapi_endpoint(method="GET", label="health")
def health():
    return {"status": "ok", "services": ["whisper-large-v3", "fish-speech-s2-pro"]}
