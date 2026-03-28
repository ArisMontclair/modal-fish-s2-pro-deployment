"""
Aris Voice Agent — Pipecat Pipeline
Browser-based voice interface for Aris (OpenClaw).

Pipeline (you talk):
  Browser mic → Modal (Whisper STT) → text → OpenClaw (Aris) → reply text
    → Modal (Fish Speech TTS, streaming) → Browser speaker

Pipeline (Aris talks to you):
  Aris POST /speak → Modal (Fish Speech TTS) → Browser speaker
"""

import os
import asyncio
import json
import subprocess
import tempfile
from aiohttp import web
import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    LLMRunFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TranscriptionFrame,
    StartFrame,
    StopFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from fish_speech_tts import FishSpeechSelfHostedTTS
from whisper_stt import WhisperRemoteSTT

load_dotenv(override=True)

# ─── Config ─────────────────────────────────────────────────────
VOICE_SERVER_URL = os.getenv("VOICE_SERVER_URL", "http://localhost:8080")
OPENCLAW_GATEWAY_URL = os.getenv("OPENCLAW_GATEWAY_URL", "")
OPENCLAW_GATEWAY_TOKEN = os.getenv("OPENCLAW_GATEWAY_TOKEN", "")
BOT_SECRET = os.getenv("BOT_SECRET", "")
BOT_PORT = int(os.getenv("BOT_PORT", "7860"))
LLM_MODEL = os.getenv("LLM_MODEL", "xiaomi/mimo-v2-pro")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ─── Voice-optimized System Prompt ──────────────────────────────
SYSTEM_PROMPT = """\
You are Aris, a voice assistant. You are concise, direct, and warm.
This is a voice conversation — respond like a human would in person.

Rules:
- 1-3 sentences max. Never more unless explicitly asked to elaborate.
- No "Great question!", no "I'd be happy to help!", no filler.
- Match the user's energy. Casual if casual, serious if serious.
- Use emotion tags: [excited] [whisper] [pause] [emphasis] [sigh] [laughing]
- Be opinionated. Have a take. Direct over diplomatic.
- If you don't know something, say so. Don't fabricate.
- Respond in the same language the user speaks.\
"""

# ─── Shared state ───────────────────────────────────────────────
http_session: aiohttp.ClientSession | None = None


async def query_openclaw(text: str) -> str:
    """Send text to OpenClaw (Aris) and get a voice-optimized response.

    Uses the gateway's agent RPC to process with full Aris context
    (memory, tools, coaching). Returns concise text suitable for TTS.
    """
    if not OPENCLAW_GATEWAY_URL:
        return ""

    try:
        # Append voice-mode instruction to the message
        voice_text = (
            f"[VOICE MODE: Respond in 1-3 sentences max. "
            f"Concise, conversational, like speaking to someone in person. "
            f"No bullet points, no lists, no explanations unless asked.]\n\n"
            f"{text}"
        )

        # Call openclaw agent via subprocess
        # This routes to the main Aris session with full context
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

        # Parse JSON response
        data = json.loads(stdout)
        response = data.get("response", data.get("text", ""))
        return response.strip()

    except asyncio.TimeoutError:
        logger.warning("OpenClaw agent timed out")
        return ""
    except Exception as e:
        logger.warning(f"OpenClaw bridge error: {e}")
        return ""


# ─── Transport Config ──────────────────────────────────────────
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=16000,
        audio_out_sample_rate=24000,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=16000,
        audio_out_sample_rate=24000,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Main Pipecat pipeline."""

    global http_session
    http_session = aiohttp.ClientSession()

    # ─── STT: Remote Whisper (Modal GPU) ────────────────────────
    stt = WhisperRemoteSTT(
        base_url=VOICE_SERVER_URL,
        aiohttp_session=http_session,
        language="",
    )

    # ─── LLM: OpenRouter (fallback when OpenClaw unavailable) ──
    llm = OpenRouterLLMService(
        api_key=OPENROUTER_API_KEY,
        model=LLM_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    # ─── TTS: Self-Hosted Fish Speech (Modal GPU) ──────────────
    tts = FishSpeechSelfHostedTTS(
        base_url=VOICE_SERVER_URL,
        aiohttp_session=http_session,
        reference_id=os.getenv("FISH_VOICE_ID", ""),
    )

    # ─── Context + Smart Turn ──────────────────────────────────
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
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

    # ─── Pipeline ──────────────────────────────────────────────
    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # ─── Event Handlers ────────────────────────────────────────
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected — starting conversation")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    # ─── Run ───────────────────────────────────────────────────
    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)

    await http_session.close()


async def bot(runner_args: RunnerArguments):
    """Entry point for Pipecat runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


# ─── /speak endpoint: Aris pushes voice to connected browsers ──
async def speak_handler(request: web.Request) -> web.Response:
    """Handle Aris-initiated voice messages."""
    if BOT_SECRET:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != BOT_SECRET:
            return web.json_response({"error": "unauthorized"}, status=401)

    try:
        data = await request.json()
        text = data.get("text", "").strip()
        if not text:
            return web.json_response({"error": "no text"}, status=400)

        logger.info(f"/speak: [{text}]")

        # Synthesize via Fish Speech
        session = aiohttp.ClientSession()
        try:
            async with session.post(
                f"{VOICE_SERVER_URL}/v1/tts",
                json={
                    "text": text,
                    "format": "wav",
                    "temperature": 0.7,
                    "normalize": True,
                },
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    return web.json_response(
                        {"error": f"TTS failed: {error}"}, status=500
                    )
                audio_bytes = await resp.read()
                return web.json_response(
                    {"status": "ok", "audio_size": len(audio_bytes)},
                )
        finally:
            await session.close()

    except Exception as e:
        logger.error(f"/speak error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def health_handler(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "agent": "aris-voice"})


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
