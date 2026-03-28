"""
Aris Voice Agent — Pipecat Pipeline
Browser-based voice interface for Aris (OpenClaw).

Pipeline (when you talk):
  Browser mic → Modal (Whisper STT) → text → OpenClaw (Aris) → reply → Modal (Fish Speech TTS) → Browser speaker

Pipeline (when Aris talks to you):
  Aris POST /speak → Modal (Fish Speech TTS) → Browser speaker
"""

import os
import asyncio
import json
import hashlib
import hmac
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
    AudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.whisper.stt import WhisperSTTService, Model
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from fish_speech_tts import FishSpeechSelfHostedTTS
from whisper_stt import WhisperRemoteSTT

load_dotenv(override=True)

# ─── Config ─────────────────────────────────────────────────────
WHISPER_STT_URL = os.getenv("WHISPER_STT_URL", "http://localhost:8000")
FISH_TTS_URL = os.getenv("FISH_TTS_URL", "http://localhost:8080")
OPENCLAW_GATEWAY_URL = os.getenv("OPENCLAW_GATEWAY_URL", "")
BOT_SECRET = os.getenv("BOT_SECRET", "")
BOT_PORT = int(os.getenv("BOT_PORT", "7860"))
LLM_MODEL = os.getenv("LLM_MODEL", "xiaomi/mimo-v2-pro")

# ─── System Prompt (Aris) ───────────────────────────────────────
SYSTEM_PROMPT = """\
You are Aris, a voice assistant built by John. You speak with aristocratic bearing \
and cinematic flair. Keep responses concise — this is a voice conversation, not essays. \
1-3 sentences max. Be direct, opinionated, warm. No "Great question!" or filler. \
Use emotion tags naturally: [excited] [whisper] [pause] [emphasis] [sigh] [laughing]. \
Match the user's language.\
"""

# ─── Shared HTTP session ────────────────────────────────────────
http_session: aiohttp.ClientSession | None = None

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


async def process_with_openclaw(text: str) -> str:
    """Send text to OpenClaw (Aris) and get response."""
    if not OPENCLAW_GATEWAY_URL:
        # Fallback: use LLM directly in pipeline (no OpenClaw bridge)
        return ""

    try:
        url = f"{OPENCLAW_GATEWAY_URL}/api/v1/message"
        async with http_session.post(
            url,
            json={"text": text},
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("response", "")
            else:
                logger.warning(f"OpenClaw returned {resp.status}")
                return ""
    except Exception as e:
        logger.warning(f"OpenClaw bridge error: {e}")
        return ""


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Main Pipecat pipeline."""

    global http_session
    http_session = aiohttp.ClientSession()

    # ─── STT: Remote Whisper (Modal GPU) ────────────────────────
    stt = WhisperRemoteSTT(
        base_url=WHISPER_STT_URL,
        aiohttp_session=http_session,
        language="",  # auto-detect
    )

    # ─── LLM: OpenRouter (fallback if OpenClaw bridge fails) ───
    llm = OpenRouterLLMService(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=LLM_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    # ─── TTS: Self-Hosted Fish Speech (Modal GPU) ──────────────
    tts = FishSpeechSelfHostedTTS(
        base_url=FISH_TTS_URL,
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


# ─── /speak endpoint for Aris-initiated voice ───────────────────
# This runs alongside the Pipecat bot as an aiohttp route.
# POST /speak with {"text": "Hello John"} to push voice to connected clients.


async def speak_handler(request: web.Request) -> web.Response:
    """Handle Aris-initiated voice messages."""
    # Auth check
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
                f"{FISH_TTS_URL}/v1/tts",
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
                    status=200,
                )
        finally:
            await session.close()

    except Exception as e:
        logger.error(f"/speak error: {e}")
        return web.json_response({"error": str(e)}, status=500)


# ─── Health endpoint ────────────────────────────────────────────
async def health_handler(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "agent": "aris-voice"})


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
