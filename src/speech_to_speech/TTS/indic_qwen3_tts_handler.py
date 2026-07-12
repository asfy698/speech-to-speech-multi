from __future__ import annotations

import logging
from pathlib import Path
from queue import Empty
from threading import Event
from time import perf_counter
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)
console = Console()

PIPELINE_SR = 16000


class IndicQwen3TTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Indic Qwen3-TTS LoRA handler. Defaults to the Tamil adapter."""

    def setup(
        self,
        should_listen: Event,
        model_name: str = "aguken-ai/Qwen3-TTS-0.6B-LoRA-Finetuned-Indic-Multilingual",
        adapter: str = "tamil_female",
        language: str = "auto",
        device: str = "cuda",
        dtype: str | torch.dtype = "auto",
        attn_implementation: str = "eager",
        lora_scale: float = 1.0,
        output_speed: float = 1.0,
        speculative_turns: SpeculativeTurnTracker | None = None,
        **_: Any,
    ) -> None:
        self.should_listen = should_listen
        self.model_name = model_name
        self.adapter = adapter
        self.language = self._resolve_language(language)
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.attn_implementation = attn_implementation
        self.lora_scale = lora_scale
        self.output_speed = output_speed
        self.speculative_turns = speculative_turns

        try:
            from huggingface_hub import snapshot_download
            from peft import PeftModel
            from qwen_tts import Qwen3TTSModel
        except ImportError as e:
            raise ImportError(
                "Indic Qwen3-TTS requires qwen-tts, peft, and huggingface_hub. "
                "Install with: pip install qwen-tts peft huggingface_hub"
            ) from e

        logger.info("Downloading/loading Indic Qwen3-TTS repo: %s", self.model_name)
        self.repo_path = Path(snapshot_download(self.model_name))
        adapter_path = self.repo_path / "adapters" / self.adapter
        if not adapter_path.exists():
            raise FileNotFoundError(f"Indic Qwen3-TTS adapter not found: {adapter_path}")

        logger.info("Loading Indic Qwen3-TTS base model on %s", self.device)
        self.model = Qwen3TTSModel.from_pretrained(
            str(self.repo_path),
            device_map=self.device,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
        )
        logger.info("Loading Indic Qwen3-TTS adapter: %s", adapter_path)
        self.model.model = PeftModel.from_pretrained(self.model.model, str(adapter_path))
        if hasattr(self.model.model, "set_adapter_scale"):
            self.model.model.set_adapter_scale(self.lora_scale)
        self.model.model.eval()

        self.ref_audio = self._find_first_file(adapter_path, ("*.wav", "*.mp3", "*.flac"))
        self.ref_text = self._read_first_text(adapter_path)
        if not self.ref_audio or not self.ref_text:
            raise FileNotFoundError(
                f"Indic Qwen3-TTS adapter {adapter_path} must include reference audio and reference text."
            )

        logger.info("Indic Qwen3-TTS loaded with adapter=%s language=%s", self.adapter, self.language)
        self.warmup()

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            return "cuda:0"
        return device

    def _resolve_dtype(self, dtype: str | torch.dtype) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        if dtype == "auto":
            return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        return getattr(torch, dtype)

    def _resolve_language(self, language: str | None) -> str | None:
        if language is None:
            return None
        normalized = language.strip().lower()
        if normalized in {"", "none", "null"}:
            return None
        if normalized == "tamil":
            logger.warning("Qwen3-TTS does not accept language='Tamil'; using language='auto' with the Tamil LoRA adapter.")
            return "auto"
        return normalized

    def _find_first_file(self, folder: Path, patterns: tuple[str, ...]) -> str | None:
        for pattern in patterns:
            matches = sorted(folder.glob(pattern))
            if matches:
                return str(matches[0])
        return None

    def _read_first_text(self, folder: Path) -> str | None:
        for name in ("ref_text.txt", "reference_text.txt", "prompt.txt", "transcript.txt"):
            path = folder / name
            if path.exists():
                return path.read_text(encoding="utf-8").strip()
        for path in sorted(folder.glob("*.txt")):
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
        return None

    def warmup(self) -> None:
        try:
            self._generate("வணக்கம்.")
            logger.info("Indic Qwen3-TTS warmed up")
        except Exception as e:
            logger.warning("Indic Qwen3-TTS warmup failed: %s", e)

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        if isinstance(tts_input, EndOfResponse):
            yield AUDIO_RESPONSE_DONE
            return

        speculative_turns = getattr(self, "speculative_turns", None)
        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug("Dropping stale Indic TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        text = self._coalesce_text(tts_input).strip() or "வணக்கம்."
        console.print(f"[green]ASSISTANT: {text}")

        start_s = perf_counter()
        try:
            audio = self._generate(text)
            if tts_input.speech_stopped_at_s is not None:
                logger.info(
                    "Last speech detected to Indic first speech out: %.3fs (turn=%s rev=%s)",
                    perf_counter() - tts_input.speech_stopped_at_s,
                    tts_input.turn_id,
                    tts_input.turn_revision,
                )
            logger.info("Indic Qwen3-TTS generated %.3fs audio in %.3fs", len(audio) / PIPELINE_SR, perf_counter() - start_s)
            yield audio
        except Exception as e:
            logger.error("Indic Qwen3-TTS generation failed: %s", e, exc_info=True)

    def _coalesce_text(self, first_input: TTSInput) -> str:
        parts = [first_input.text.strip()] if first_input.text.strip() else []
        if not hasattr(self.queue_in, "mutex") or not hasattr(self.queue_in, "queue"):
            return " ".join(parts)

        with self.queue_in.mutex:
            while self.queue_in.queue:
                next_item = self.queue_in.queue[0]
                if isinstance(next_item, EndOfResponse):
                    break
                if not isinstance(next_item, TTSInput):
                    break
                if first_input.turn_id != next_item.turn_id or first_input.turn_revision != next_item.turn_revision:
                    break
                try:
                    self.queue_in.queue.popleft()
                except Empty:
                    break
                if next_item.text.strip():
                    parts.append(next_item.text.strip())
        return " ".join(parts)

    def _generate(self, text: str) -> bytes:
        with torch.inference_mode():
            generated = self.model.generate_voice_clone(
                text=text,
                language=self.language,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
            )
        audio, sample_rate = self._extract_audio(generated)
        audio = self._to_float32_mono(audio)
        audio = self._resample(audio, sample_rate, PIPELINE_SR)
        if self.output_speed and self.output_speed != 1.0:
            audio = self._speed(audio, self.output_speed)
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767.0).astype(np.int16).tobytes()

    def _extract_audio(self, generated: Any) -> tuple[Any, int]:
        if isinstance(generated, dict):
            audio = None
            for key in ("audio", "waveform", "wav"):
                if key in generated and generated[key] is not None:
                    audio = generated[key]
                    break
            sample_rate = int(generated.get("sampling_rate") or generated.get("sample_rate") or 24000)
            return audio, sample_rate
        if isinstance(generated, tuple) and len(generated) >= 2:
            return generated[0], int(generated[1])
        return generated, 24000

    def _to_float32_mono(self, audio: Any) -> np.ndarray:
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().float().cpu().numpy()
        audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=0)
        if audio_np.size and np.max(np.abs(audio_np)) > 1.5:
            audio_np = audio_np / 32768.0
        return audio_np

    def _resample(self, audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
        if source_sr == target_sr:
            return audio
        tensor = torch.from_numpy(audio).float().view(1, 1, -1)
        target_len = max(1, round(audio.shape[-1] * target_sr / source_sr))
        resampled = F.interpolate(tensor, size=target_len, mode="linear", align_corners=False)
        return resampled.view(-1).numpy().astype(np.float32)

    def _speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        if speed <= 0:
            return audio
        tensor = torch.from_numpy(audio).float().view(1, 1, -1)
        target_len = max(1, round(audio.shape[-1] / speed))
        stretched = F.interpolate(tensor, size=target_len, mode="linear", align_corners=False)
        return stretched.view(-1).numpy().astype(np.float32)
