"""
Aris Voice Agent — FastAPI + Pipecat Pipeline
Browser-based voice interface for Aris (OpenClaw).

Run:  python bot.py
Open: http://localhost:7860
"""

import os
import io
import struct
import wave
import asyncio
import json
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from loguru import logger
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

load_dotenv(override=True)

# ─── Config ─────────────────────────────────────────────────────
VOICE_SERVER_URL = os.getenv("VOICE_SERVER_URL", "http://localhost:8080")
MODAL_HEALTH_URL = os.getenv("MODAL_HEALTH_URL", "")  # Lightweight health check (no GPU)
OPENCLAW_GATEWAY_URL = os.getenv("OPENCLAW_GATEWAY_URL", "")
OPENCLAW_GATEWAY_TOKEN = os.getenv("OPENCLAW_GATEWAY_TOKEN", "")
BOT_PORT = int(os.getenv("BOT_PORT", "7860"))
LLM_MODEL = os.getenv("LLM_MODEL", "xiaomi/mimo-v2-pro")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ─── Shared state ───────────────────────────────────────────────
http_session: aiohttp.ClientSession | None = None
_active_connections: list = []  # SmallWebRTCTransport instances
_connections_lock = asyncio.Lock()


# ─── OpenClaw Bridge ────────────────────────────────────────────
async def query_openclaw(text: str) -> str:
    """Send text to OpenClaw (Aris) and get a voice-optimized response."""
    if not OPENCLAW_GATEWAY_URL:
        return ""

    try:
        voice_text = (
            f"[VOICE MODE: Respond in 1-3 sentences max. "
            f"Concise, conversational, like speaking to someone in person. "
            f"No bullet points, no lists, no explanations unless asked.]\n\n"
            f"{text}"
        )

        result = await asyncio.create_subprocess_exec(
            "openclaw", "agent",
            "--to", "+31681299666",
            "--message", voice_text,
            "--json",
            "--timeout", "30",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                **os.environ,
                "OPENCLAW_GATEWAY_URL": OPENCLAW_GATEWAY_URL,
                "OPENCLAW_GATEWAY_TOKEN": OPENCLAW_GATEWAY_TOKEN,
            },
        )
        stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=35)

        if result.returncode != 0:
            logger.warning(f"OpenClaw agent failed: {stderr.decode()}")
            return ""

        data = json.loads(stdout)
        response = data.get("response", data.get("text", ""))
        return response.strip()

    except asyncio.TimeoutError:
        logger.warning("OpenClaw agent timed out")
        return ""
    except Exception as e:
        logger.warning(f"OpenClaw bridge error: {e}")
        return ""


# ─── Bot Pipeline ───────────────────────────────────────────────
async def run_bot(webrtc_connection):
    """Run the Pipecat pipeline for a single WebRTC connection."""
    from pipecat.frames.frames import (
        Frame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        TextFrame,
        TranscriptionFrame,
    )
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport, TransportParams

    from orpheus_tts import OrpheusTTS
    from whisper_stt import WhisperRemoteSTT

    global http_session
    if http_session is None:
        http_session = aiohttp.ClientSession()

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )

    stt = WhisperRemoteSTT(
        base_url=VOICE_SERVER_URL,
        aiohttp_session=http_session,
        language="",
    )

    tts = OrpheusTTS(
        base_url=VOICE_SERVER_URL,
        aiohttp_session=http_session,
        voice=os.getenv("TTS_VOICE", "tara"),
    )

    # ─── OpenClaw Bridge Processor ──────────────────────────────
    class OpenClawBridge(FrameProcessor):
        def __init__(self):
            super().__init__()
            self._use_openclaw = bool(OPENCLAW_GATEWAY_URL)

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, TranscriptionFrame) and self._use_openclaw:
                text = frame.text.strip()
                if not text:
                    return

                logger.info(f"→ OpenClaw: [{text}]")
                response = await query_openclaw(text)

                if response:
                    logger.info(f"← OpenClaw: [{response}]")
                    await self.push_frame(LLMFullResponseStartFrame())
                    await self.push_frame(TextFrame(text=response))
                    await self.push_frame(LLMFullResponseEndFrame())
                else:
                    logger.warning("OpenClaw unavailable")
                    await self.push_frame(LLMFullResponseStartFrame())
                    await self.push_frame(
                        TextFrame(text="Sorry, I couldn't reach my brain. Try again.")
                    )
                    await self.push_frame(LLMFullResponseEndFrame())
            else:
                await self.push_frame(frame, direction)

    bridge = OpenClawBridge()

    # ─── Build Pipeline ─────────────────────────────────────────
    if OPENCLAW_GATEWAY_URL:
        pipeline = Pipeline([
            transport.input(),
            stt,
            bridge,
            tts,
            transport.output(),
        ])
    else:
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (
            LocalSmartTurnAnalyzerV3,
        )
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMContextAggregatorPair,
            LLMUserAggregatorParams,
        )
        from pipecat.services.openrouter.llm import OpenRouterLLMService
        from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
        from pipecat.turns.user_turn_strategies import UserTurnStrategies

        llm = OpenRouterLLMService(
            api_key=OPENROUTER_API_KEY,
            model=LLM_MODEL,
            system_instruction=(
                "You are Aris, a voice assistant built by John. "
                "You are concise, direct, and warm. "
                "1-3 sentences max. No filler. Be opinionated. "
                "Use emotion tags: [excited] [whisper] [pause] [emphasis] [sigh]. "
                "Match the user's language."
            ),
        )

        context = LLMContext()
        user_agg, assistant_agg = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[
                        TurnAnalyzerUserTurnStopStrategy(
                            turn_analyzer=LocalSmartTurnAnalyzerV3()
                        )
                    ]
                ),
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        pipeline = Pipeline([
            transport.input(),
            stt,
            user_agg,
            llm,
            tts,
            transport.output(),
            assistant_agg,
        ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Track this transport for /speak broadcasts
    async with _connections_lock:
        _active_connections.append(transport)
    logger.info(f"Transport registered ({len(_active_connections)} active)")

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, connection):
        logger.info("Client connected — ready to talk")

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, connection):
        logger.info("Client disconnected")
        async with _connections_lock:
            if transport in _active_connections:
                _active_connections.remove(transport)
        logger.info(f"Transport removed ({len(_active_connections)} active)")
        await task.cancel()

    runner = PipelineRunner()
    await runner.run(task)


# ─── FastAPI Server ─────────────────────────────────────────────
webrtc_handler = SmallWebRTCRequestHandler()


@asynccontextmanager
async def lifespan(app):
    global http_session
    http_session = aiohttp.ClientSession()
    yield
    await http_session.close()
    await webrtc_handler.close()


app = FastAPI(title="Aris Voice Agent", lifespan=lifespan)


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    async def callback(connection):
        background_tasks.add_task(run_bot, connection)

    return await webrtc_handler.handle_web_request(request, callback)


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    await webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


# ─── Health Checks ──────────────────────────────────────────────
@app.get("/api/health/modal")
async def health_modal():
    # Use lightweight health endpoint if configured (no GPU cost)
    # Falls back to GPU server health (which wakes the GPU — avoid polling this)
    url = MODAL_HEALTH_URL or f"{VOICE_SERVER_URL}/health"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                url,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                data = await resp.json()
                return {"status": data.get("status", "unknown"), "stage": data.get("stage", "")}
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


@app.get("/api/health/openclaw")
async def health_openclaw():
    if not OPENCLAW_GATEWAY_URL:
        return {"status": "not_configured"}
    try:
        result = await asyncio.create_subprocess_exec(
            "openclaw", "health",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                **os.environ,
                "OPENCLAW_GATEWAY_URL": OPENCLAW_GATEWAY_URL,
                "OPENCLAW_GATEWAY_TOKEN": OPENCLAW_GATEWAY_TOKEN,
            },
        )
        stdout, _ = await asyncio.wait_for(result.communicate(), timeout=10)
        return {"status": "connected" if result.returncode == 0 else "error"}
    except Exception:
        return {"status": "unreachable"}


# ─── Speak Endpoint (push voice to connected browsers) ──────────
TTS_SAMPLE_RATE = 24000  # Fish Speech outputs 24kHz


def _strip_wav_header(wav_bytes: bytes) -> bytes:
    """Extract raw PCM data from a WAV file, skipping the header."""
    # WAV header is at least 44 bytes; find the 'data' chunk
    idx = wav_bytes.find(b"data")
    if idx == -1:
        # No data chunk found — assume 44-byte standard header
        return wav_bytes[44:]
    # data chunk: 4 bytes 'data' + 4 bytes chunk size + audio data
    data_size = struct.unpack_from("<I", wav_bytes, idx + 4)[0]
    pcm_start = idx + 8
    return wav_bytes[pcm_start : pcm_start + data_size]


def _pad_to_10ms(pcm: bytes, sample_rate: int = TTS_SAMPLE_RATE) -> bytes:
    """Pad PCM bytes to a multiple of 10ms chunks (required by RawAudioTrack)."""
    bytes_per_10ms = sample_rate * 2 * 10 // 1000  # 16-bit mono
    remainder = len(pcm) % bytes_per_10ms
    if remainder != 0:
        pcm += b"\x00" * (bytes_per_10ms - remainder)
    return pcm


async def _generate_tts_audio(text: str) -> bytes | None:
    """Call Modal TTS server and return raw PCM bytes (24kHz, 16-bit mono)."""
    global http_session
    if http_session is None:
        http_session = aiohttp.ClientSession()

    try:
        payload = {"text": text, "voice": os.getenv("TTS_VOICE", "tara")}

        url = f"{VOICE_SERVER_URL}/v1/tts"
        async with http_session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                logger.error(f"TTS error ({resp.status}): {error}")
                return None
            wav_bytes = await resp.read()

        pcm = _strip_wav_header(wav_bytes)
        pcm = _pad_to_10ms(pcm)
        return pcm

    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return None


async def _broadcast_audio(pcm: bytes):
    """Push raw PCM audio to all connected WebRTC transports."""
    from pipecat.frames.frames import OutputAudioRawFrame

    frame = OutputAudioRawFrame(
        audio=pcm,
        sample_rate=TTS_SAMPLE_RATE,
        num_channels=1,
    )

    async with _connections_lock:
        transports = list(_active_connections)

    for t in transports:
        try:
            output = t.output()
            if hasattr(output, "_client"):
                await output._client.write_audio_frame(frame)
            else:
                logger.warning("Transport output has no _client attribute")
        except Exception as e:
            logger.error(f"Failed to push audio to transport: {e}")


@app.post("/speak")
async def speak(request: Request):
    """Push voice to all connected browsers. No auth required.

    POST /speak with JSON: {"text": "Hello from Aris"}
    or query param: /speak?text=Hello+from+Aris
    """
    # Parse text from body or query param
    text = ""
    try:
        body = await request.json()
        text = body.get("text", "")
    except Exception:
        pass
    if not text:
        text = request.query_params.get("text", "")

    if not text:
        return JSONResponse({"error": "text required"}, status_code=400)

    async with _connections_lock:
        count = len(_active_connections)

    if count == 0:
        return JSONResponse({"error": "no connected browsers"}, status_code=409)

    pcm = await _generate_tts_audio(text)
    if pcm is None:
        return JSONResponse({"error": "TTS generation failed"}, status_code=502)

    await _broadcast_audio(pcm)
    logger.info(f"/speak: pushed [{text[:80]}] to {count} connection(s)")
    return {"status": "ok", "connections": count}


@app.get("/speak")
async def speak_get(text: str = ""):
    """GET /speak?text=Hello — convenience endpoint for testing."""
    if not text:
        return JSONResponse({"error": "text query param required"}, status_code=400)

    async with _connections_lock:
        count = len(_active_connections)

    if count == 0:
        return JSONResponse({"error": "no connected browsers"}, status_code=409)

    pcm = await _generate_tts_audio(text)
    if pcm is None:
        return JSONResponse({"error": "TTS generation failed"}, status_code=502)

    await _broadcast_audio(pcm)
    logger.info(f"/speak: pushed [{text[:80]}] to {count} connection(s)")
    return {"status": "ok", "connections": count}


# ─── Dashboard ──────────────────────────────────────────────────
@app.get("/")
async def dashboard():
    return FileResponse("dashboard.html")


# ─── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")

    logger.info(f"Modal server: {VOICE_SERVER_URL}")
    logger.info(f"OpenClaw: {OPENCLAW_GATEWAY_URL or 'not configured'}")
    logger.info(f"Starting Aris Voice Agent on port {BOT_PORT}...")

    uvicorn.run(app, host="0.0.0.0", port=BOT_PORT)
