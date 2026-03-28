"""
Whisper Remote STT Service for Pipecat
Calls a self-hosted Whisper server on Modal via HTTP.
No GPU needed on the bot machine.
"""

import tempfile
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.utils.tracing.service_decorators import traced_stt


class WhisperRemoteSTT(STTService):
    """Pipecat STT service for remote Whisper server (HTTP).

    Sends audio chunks to a Whisper server running on Modal
    and returns transcriptions.
    """

    def __init__(
        self,
        *,
        base_url: str,
        aiohttp_session: aiohttp.ClientSession,
        language: str = "",
        sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        self._base_url = base_url
        self._session = aiohttp_session
        self._language = language
        self._sample_rate = sample_rate
        self._audio_buffer = bytearray()
        self._buffer_duration_sec = 2.0  # process every 2 seconds
        self._buffer_max_bytes = int(sample_rate * 2 * self._buffer_duration_sec)  # 16-bit PCM

        logger.info(f"WhisperRemoteSTT initialized: {self._base_url}/v1/transcribe")

    def can_generate_metrics(self) -> bool:
        return True

    @traced_stt
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Buffer audio and transcribe when buffer is full.

        Args:
            audio: Raw PCM audio bytes (16-bit, mono).
        """
        self._audio_buffer.extend(audio)

        if len(self._audio_buffer) < self._buffer_max_bytes:
            return

        # Get buffer and clear it
        buf = bytes(self._audio_buffer)
        self._audio_buffer.clear()

        await self.start_processing_metrics()

        try:
            # Convert PCM to WAV in memory
            import struct
            import wave
            import io

            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self._sample_rate)
                wf.writeframes(buf)
            wav_buffer.seek(0)

            # POST to Whisper server
            data = aiohttp.FormData()
            data.add_field(
                "audio",
                wav_buffer,
                filename="audio.wav",
                content_type="audio/wav",
            )
            if self._language:
                data.add_field("language", self._language)

            url = f"{self._base_url}/v1/transcribe"
            async with self._session.post(url, data=data) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"Whisper STT error ({response.status}): {error}")
                    yield ErrorFrame(error=f"Whisper STT error: {error}")
                    return

                result = await response.json()
                text = result.get("text", "").strip()

                if text:
                    logger.debug(f"Whisper STT: [{text}]")
                    yield TranscriptionFrame(
                        text=text,
                        user_id="",
                        timestamp="",
                    )

        except Exception as e:
            logger.error(f"WhisperRemoteSTT error: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            await self.stop_processing_metrics()
