"""
Aris Voice Agent — FastAPI + Pipecat Pipeline
Browser-based voice interface for Aris (OpenClaw).

Run:  python bot.py
Open: http://localhost:7860
"""

import os
import asyncio
import json
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

load_dotenv(override=True)

# ─── Config ─────────────────────────────────────────────────────
VOICE_SERVER_URL = os.getenv("VOICE_SERVER_URL", "http://localhost:8080")
OPENCLAW_GATEWAY_URL = os.getenv("OPENCLAW_GATEWAY_URL", "")
OPENCLAW_GATEWAY_TOKEN = os.getenv("OPENCLAW_GATEWAY_TOKEN", "")
BOT_SECRET = os.getenv("BOT_SECRET", "")
BOT_PORT = int(os.getenv("BOT_PORT", "7860"))
LLM_MODEL = os.getenv("LLM_MODEL", "xiaomi/mimo-v2-pro")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ─── Shared HTTP session ────────────────────────────────────────
http_session: aiohttp.ClientSession | None = None


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
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (
        LocalSmartTurnAnalyzerV3,
    )
    from pipecat.audio.vad.silero import SileroVADAnalyzer
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
    from pipecat.services.openrouter.llm import OpenRouterLLMService
    from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
    from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
    from pipecat.turns.user_turn_strategies import UserTurnStrategies

    from fish_speech_tts import FishSpeechSelfHostedTTS
    from whisper_stt import WhisperRemoteSTT

    global http_session
    if http_session is None:
        http_session = aiohttp.ClientSession()

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=SmallWebRTCTransport.Params(
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

    tts = FishSpeechSelfHostedTTS(
        base_url=VOICE_SERVER_URL,
        aiohttp_session=http_session,
        reference_id=os.getenv("FISH_VOICE_ID", ""),
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
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMContextAggregatorPair,
            LLMUserAggregatorParams,
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

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, connection):
        logger.info("Client connected — ready to talk")

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, connection):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner()
    await runner.run(task)


# ─── FastAPI Server ─────────────────────────────────────────────
webrtc_handler = SmallWebRTCRequestHandler()


app = FastAPI(title="Aris Voice Agent")


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
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"{VOICE_SERVER_URL}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                data = await resp.json()
                return {"status": data.get("status", "unknown")}
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


# ─── Dashboard ──────────────────────────────────────────────────
@app.get("/")
async def dashboard():
    return FileResponse("dashboard.html")


@asynccontextmanager
async def lifespan(app):
    global http_session
    http_session = aiohttp.ClientSession()
    yield
    await http_session.close()
    await webrtc_handler.close()


app.router.lifespan_context = lifespan


# ─── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")

    logger.info(f"Modal server: {VOICE_SERVER_URL}")
    logger.info(f"OpenClaw: {OPENCLAW_GATEWAY_URL or 'not configured'}")
    logger.info(f"Starting Aris Voice Agent on port {BOT_PORT}...")

    uvicorn.run(app, host="0.0.0.0", port=BOT_PORT)
