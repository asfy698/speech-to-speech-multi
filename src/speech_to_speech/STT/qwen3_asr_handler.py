from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Iterator

import numpy as np
import torch
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)
console = Console()

PIPELINE_SR = 16000


class Qwen3ASRSTTHandler(BaseSTTHandler):
    """Speech-to-text handler for Qwen3 ASR models, defaulting to Tamil ASR."""

    def setup(
        self,
        model_name: str = "osmapi/tamil-asr-qwen3",
        device: str = "auto",
        dtype: str | torch.dtype = "auto",
        language_code: str = "auto",
        **_: Any,
    ) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.language_code = language_code
        self.transcribe_language = self._resolve_language(language_code)

        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as e:
            raise ImportError("Qwen3 ASR requires qwen-asr. Install with: pip install qwen-asr") from e

        logger.info("Loading Qwen3 ASR model: %s on %s", self.model_name, self.device)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "cuda:0" if self.device == "cuda" else self.device,
        }
        if self.dtype is not None:
            model_kwargs["dtype"] = self.dtype
        self.asr = Qwen3ASRModel.from_pretrained(self.model_name, **model_kwargs)
        logger.info("Qwen3 ASR model loaded")
        self.warmup()

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _resolve_dtype(self, dtype: str | torch.dtype) -> torch.dtype | None:
        if isinstance(dtype, torch.dtype):
            return dtype
        if dtype == "auto":
            if torch.cuda.is_available():
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return None
        return getattr(torch, dtype)

    def _resolve_language(self, language_code: str | None) -> str | None:
        if language_code is None:
            return None
        normalized = language_code.strip()
        if normalized.lower() in {"", "auto", "none", "null"}:
            return None
        return normalized

    def warmup(self) -> None:
        try:
            dummy_audio = np.zeros(PIPELINE_SR, dtype=np.float32)
            self._transcribe(dummy_audio)
            logger.info("Qwen3 ASR warmed up")
        except Exception as e:
            logger.warning("Qwen3 ASR warmup failed: %s", e)

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        if vad_audio.mode == "progressive":
            # Keep Tamil mode simple and cheap: final ASR only.
            yield PartialTranscription(
                text="",
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
            return

        start_s = perf_counter()
        audio = vad_audio.audio
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        else:
            audio = audio.astype(np.float32, copy=False)

        logger.info(
            "Qwen3 ASR final STT start turn=%s rev=%s audio=%.3fs",
            vad_audio.turn_id,
            vad_audio.turn_revision,
            len(audio) / PIPELINE_SR,
        )
        try:
            text = self._transcribe(audio).strip()
        except Exception as e:
            logger.error("Qwen3 ASR inference failed: %s", e, exc_info=True)
            text = ""

        logger.info(
            "Qwen3 ASR final STT done turn=%s rev=%s total=%.3fs chars=%d",
            vad_audio.turn_id,
            vad_audio.turn_revision,
            perf_counter() - start_s,
            len(text),
        )
        if text:
            console.print(f"[yellow]USER: {text}")

        yield Transcription(
            text=text,
            language_code=self.language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )

    def _transcribe(self, audio: np.ndarray) -> str:
        result = self.asr.transcribe((audio, PIPELINE_SR), language=self.transcribe_language)
        if isinstance(result, list) and result:
            return str(getattr(result[0], "text", result[0]))
        return ""

    @property
    def timing_log_level(self) -> int:
        return logging.INFO
