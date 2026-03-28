"""
Fish Speech S2 Pro TTS Server on Modal

Deploy:  modal deploy tts_server.py
Test:    curl -X POST https://your-org--fish-tts-server.modal.run/v1/tts \
           -H "Content-Type: application/json" \
           -d '{"text": "Hello!", "format": "wav"}' \
           --output test.wav
"""

import subprocess
import modal

# ─── Image ──────────────────────────────────────────────────────
tts_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "ffmpeg", "build-essential")
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
        "uvicorn",
        "fastapi",
    )
    .run_commands("pip install -e '.[server]'")
)

app = modal.App("fish-tts", image=tts_image)
model_volume = modal.Volume.from_name("fish-speech-models", create_if_missing=True)

MODEL_DIR = "/models"
CHECKPOINT = f"{MODEL_DIR}/s2-pro"
PORT = 8080


@app.function(
    gpu="A10G",
    timeout=3600,
    scaledown_window=15,
    volumes={MODEL_DIR: model_volume},
    memory=16384,
)
@modal.web_server(port=PORT, startup_timeout=180)
def server():
    """Fish Speech S2 Pro HTTP API server."""
    import os

    if not os.path.exists(CHECKPOINT):
        print("Downloading S2 Pro model weights (~8GB, first run)...")
        subprocess.run(
            [
                "huggingface-cli",
                "download",
                "fishaudio/s2-pro",
                "--local-dir",
                CHECKPOINT,
            ],
            check=True,
        )
        model_volume.commit()
        print("Model cached.")

    cmd = [
        "python",
        "tools/api_server.py",
        "--llama-checkpoint-path",
        CHECKPOINT,
        "--decoder-checkpoint-path",
        f"{CHECKPOINT}/codec.pth",
        "--listen",
        f"0.0.0.0:{PORT}",
        "--half",
    ]

    print(f"Fish Speech S2 Pro starting on :{PORT}")
    subprocess.Popen(cmd)


@app.function(image=modal.Image.debian_slim(), timeout=10)
@modal.web_endpoint(method="GET", label="fish-health")
def health():
    return {"status": "ok", "model": "fish-speech-s2-pro"}
