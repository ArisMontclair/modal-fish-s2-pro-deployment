"""
Fish Speech Self-Hosted TTS Service for Pipecat
Connects to a self-hosted Fish Speech S2 Pro HTTP server (Modal).
"""

from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


@dataclass
class FishSpeechTTSSettings(TTSSettings):
    reference_id: Optional[str] = None
    format: str = "wav"
    temperature: float = 0.7
    top_p: float = 0.7
    repetition_penalty: float = 1.2
    chunk_length: int = 200
    normalize: bool = True


class FishSpeechSelfHostedTTS(TTSService):
    """Pipecat TTS service for self-hosted Fish Speech S2 Pro.

    Connects to a Fish Speech HTTP API server (POST /v1/tts).
    """

    Settings = FishSpeechTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        base_url: str,
        aiohttp_session: aiohttp.ClientSession,
        reference_id: Optional[str] = None,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        default_settings = self.Settings(
            model=None, voice=None, language=None, reference_id=None
        )
        if reference_id is not None:
            default_settings.reference_id = reference_id
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        if base_url.endswith("/"):
            base_url = base_url[:-1]
        self._base_url = base_url
        self._session = aiohttp_session
        logger.info(f"FishSpeechSelfHostedTTS initialized: {self._base_url}/v1/tts")

    def can_generate_metrics(self) -> bool:
        return True

    @traced_tts
    async def run_tts(
        self, text: str, context_id: str
    ) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        url = f"{self._base_url}/v1/tts"
        payload = {
            "text": text,
            "format": self._settings.format,
            "temperature": self._settings.temperature,
            "top_p": self._settings.top_p,
            "repetition_penalty": self._settings.repetition_penalty,
            "chunk_length": self._settings.chunk_length,
            "normalize": self._settings.normalize,
        }
        if self._settings.reference_id:
            payload["reference_id"] = self._settings.reference_id

        try:
            async with self._session.post(
                url, json=payload, headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"Fish Speech TTS error ({response.status}): {error}")
                    yield ErrorFrame(error=f"TTS error: {response.status} {error}")
                    yield TTSStoppedFrame(context_id=context_id)
                    return

                await self.start_tts_usage_metrics(text)

                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(self.chunk_size),
                    strip_wav_header=True,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame

        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"TTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()
