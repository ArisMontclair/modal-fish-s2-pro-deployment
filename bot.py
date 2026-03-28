"""
Aris Voice Agent — Pipecat Pipeline
Browser-based voice interface for Aris (OpenClaw).

Pipeline (you talk):
  Browser mic → Modal (Whisper STT) → text → OpenClaw (Aris) → reply text
    → Modal (Fish Speech TTS) → Browser speaker
"""

import os
import asyncio
import json
from dotenv import load_dotenv
from loguru import logger

import aiohttp
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
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
You are Aris, a voice assistant built by John. You are concise, direct, and warm.
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


async def query_openclaw(text: str) -> str:
    """Send text to OpenClaw (Aris) and get a voice-optimized response.

    Returns concise text suitable for TTS, or empty string on failure.
    Falls back to direct LLM if OpenClaw is unreachable.
    """
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


class OpenClawBridge(FrameProcessor):
    """Intercepts transcription frames, routes to OpenClaw, emits TTS-ready frames.

    Pipeline: STT → [TranscriptionFrame] → OpenClawBridge → [TextFrame] → TTS
    """

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
                # These frames trigger TTS in the pipeline
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(TextFrame(text=response))
                await self.push_frame(LLMFullResponseEndFrame())
            else:
                # OpenClaw failed — speak an error rather than silence
                logger.warning("OpenClaw unavailable, speaking fallback")
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(TextFrame(text="Sorry, I couldn't reach my brain. Try again."))
                await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)


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

    session = aiohttp.ClientSession()

    try:
        # ─── STT: Remote Whisper (Modal GPU) ────────────────────────
        stt = WhisperRemoteSTT(
            base_url=VOICE_SERVER_URL,
            aiohttp_session=session,
            language="",
        )

        # ─── LLM: OpenRouter (fallback when OpenClaw unavailable) ──
        llm = OpenRouterLLMService(
            api_key=OPENROUTER_API_KEY,
            model=LLM_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )

        # ─── OpenClaw Bridge ───────────────────────────────────────
        bridge = OpenClawBridge()

        # ─── TTS: Self-Hosted Fish Speech (Modal GPU) ──────────────
        tts = FishSpeechSelfHostedTTS(
            base_url=VOICE_SERVER_URL,
            aiohttp_session=session,
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
        # When OpenClaw is configured:
        #   STT → bridge (intercepts, calls OpenClaw) → TTS → output
        # When OpenClaw is NOT configured:
        #   STT → aggregator → LLM (OpenRouter) → TTS → output
        if OPENCLAW_GATEWAY_URL:
            pipeline = Pipeline([
                transport.input(),
                stt,
                bridge,          # OpenClaw intercepts here
                tts,
                transport.output(),
            ])
        else:
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
            logger.info("Client connected — ready to talk")

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await task.cancel()

        # ─── Run ───────────────────────────────────────────────────
        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
        await runner.run(task)

    finally:
        await session.close()


async def bot(runner_args: RunnerArguments):
    """Entry point for Pipecat runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
