import csv
import logging
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from queue import Queue
from threading import Event as ThreadingEvent
from typing import Any, Callable, Literal, Optional, TypeVar, Union

from pathlib import Path

from openai.types.realtime import (
    ConversationItem,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeError,
    RealtimeErrorEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCancelEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    SessionCreatedEvent,
    SessionUpdateEvent,
)
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from pydantic import BaseModel, ConfigDict, Field, ValidationError

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a runtime dependency in the lockfile
    psutil = None

_PSUTIL_PROCESS = psutil.Process() if psutil is not None else None

from speech_to_speech.api.openai_realtime.handlers import (
    AudioHandler,
    ConversationHandler,
    ResponseHandler,
    SessionHandler,
)
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.pipeline.events import (
    AssistantTextEvent,
    PartialTranscriptionEvent,
    PipelineEvent,
    ResponseFailedEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TokenUsageEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.messages import GenerateResponseRequest
from speech_to_speech.pipeline.queue_types import TextPromptItem
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.utils import _generate_id

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512
BYTES_PER_SAMPLE = 2
CHUNK_SIZE_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE
LOOP_LOG_CSV_NAME = "realtime_loop_log.csv"
LOOP_LOG_MD_NAME = "realtime_loop_log.md"

_ResponseStatus = Literal["completed", "cancelled", "failed", "incomplete", "in_progress"]
_StatusReason = Literal["turn_detected", "client_cancelled", "max_output_tokens", "content_filter"]

_EVENT_TYPE_TO_MODEL: dict[str, type[BaseModel]] = {
    "input_audio_buffer.append": InputAudioBufferAppendEvent,
    "session.update": SessionUpdateEvent,
    "conversation.item.create": ConversationItemCreateEvent,
    "response.create": ResponseCreateEvent,
    "response.cancel": ResponseCancelEvent,
}

ClientEvent = Union[
    InputAudioBufferAppendEvent,
    SessionUpdateEvent,
    ConversationItemCreateEvent,
    ResponseCreateEvent,
    ResponseCancelEvent,
]

ServerEvent = Union[
    SessionCreatedEvent,
    RealtimeErrorEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    ConversationItemCreatedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
]

RealtimeEvent = Union[ClientEvent, ServerEvent]


_UsageMetricsT = TypeVar("_UsageMetricsT", bound="UsageMetrics")


class UsageMetrics(BaseModel):
    """Per-response usage counters.

    Supports ``+=`` for rolling per-response metrics into a global total
    and ``reset()`` for clearing per-response state after rollup.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    audio_duration_s: float = 0.0
    responses_completed: int = 0
    responses_cancelled: int = 0
    tool_calls: int = 0
    turns: int = 0

    def __iadd__(self: _UsageMetricsT, other: "UsageMetrics") -> _UsageMetricsT:
        for field in UsageMetrics.model_fields:
            setattr(self, field, getattr(self, field) + getattr(other, field))
        return self

    def reset(self) -> None:
        for field, info in UsageMetrics.model_fields.items():
            setattr(self, field, info.default)


class GlobalUsageMetrics(UsageMetrics):
    """Server-wide metrics that extend per-response counters with
    connection and error tracking."""

    connections: int = 0
    # connection duration in seconds.
    # latency tts, llm, vad, stt (mean, max, p90)
    errors_by_type: dict[str, int] = Field(default_factory=dict)

    def record_error(self, error_type: str) -> None:
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

    @property
    def total_errors(self) -> int:
        return sum(self.errors_by_type.values())


class ResourceSample(BaseModel):
    """Instantaneous resource snapshot collected during a loop stage."""

    at_s: float = Field(default_factory=time.perf_counter)
    cpu_percent: float | None = None
    ram_percent: float | None = None
    gpu_percent: float | None = None
    vram_percent: float | None = None
    vram_bytes: int | None = None


class LoopStageMetrics(BaseModel):
    """Per-stage metadata and raw samples for a single loop."""

    model_name: str | None = None
    status: str | None = None
    started_at_s: float | None = None
    finished_at_s: float | None = None
    elapsed_s: float | None = None
    bytes: int = 0
    first_sentence_latency_s: float | None = None
    samples: list[ResourceSample] = Field(default_factory=list)


class LoopRecord(BaseModel):
    """One user-speech-to-TTS cycle tracked on the server."""

    loop_index: int = 0
    speech_started_at_s: float | None = None
    audio_sent_to_stt_at_s: float | None = None
    tts_completed_at_s: float | None = None
    listening_time_s: float | None = None
    total_loop_time_s: float | None = None
    stt: LoopStageMetrics = Field(default_factory=LoopStageMetrics)
    llm: LoopStageMetrics = Field(default_factory=LoopStageMetrics)
    tts: LoopStageMetrics = Field(default_factory=LoopStageMetrics)

    def _summary_for_samples(self, samples: list[ResourceSample], field: str) -> dict[str, float | None]:
        values = [float(getattr(sample, field)) for sample in samples if getattr(sample, field) is not None]
        if not values:
            return {"average": None, "minimum": None, "maximum": None}
        return {
            "average": round(sum(values) / len(values), 3),
            "minimum": round(min(values), 3),
            "maximum": round(max(values), 3),
        }

    def _resource_block(self, stage: LoopStageMetrics) -> dict[str, Any]:
        return {
            "cpu_percent": self._summary_for_samples(stage.samples, "cpu_percent"),
            "ram_percent": self._summary_for_samples(stage.samples, "ram_percent"),
            "gpu_percent": self._summary_for_samples(stage.samples, "gpu_percent"),
            "vram_percent": self._summary_for_samples(stage.samples, "vram_percent"),
            "vram_bytes": self._summary_for_samples(stage.samples, "vram_bytes"),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "logged_at_utc": datetime.now(timezone.utc).isoformat(),
            "loop_index": self.loop_index,
            "speech_started_at_s": self.speech_started_at_s,
            "audio_sent_to_stt_at_s": self.audio_sent_to_stt_at_s,
            "listening_time_s": self.listening_time_s,
            "stt": {**self.stt.model_dump(exclude={"samples"}), "resources": self._resource_block(self.stt)},
            "llm": {**self.llm.model_dump(exclude={"samples"}), "resources": self._resource_block(self.llm)},
            "tts": {**self.tts.model_dump(exclude={"samples"}), "resources": self._resource_block(self.tts)},
            "tts_completed_at_s": self.tts_completed_at_s,
            "total_loop_time_s": self.total_loop_time_s,
        }

    def to_live_dict(self, now: float) -> dict[str, Any]:
        data = self.to_dict()
        if data["listening_time_s"] is None and self.speech_started_at_s is not None:
            data["listening_time_s"] = round(max(0.0, now - self.speech_started_at_s), 3)
        return data

    @staticmethod
    def _summary_from_values(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"average": None, "minimum": None, "maximum": None}
        return {
            "average": round(sum(values) / len(values), 3),
            "minimum": round(min(values), 3),
            "maximum": round(max(values), 3),
        }

    @classmethod
    def summarize_loop_history(cls, history: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "completed_loops": len(history),
            "listening_time_s": cls._summary_from_values(
                [float(loop["listening_time_s"]) for loop in history if loop.get("listening_time_s") is not None]
            ),
            "total_loop_time_s": cls._summary_from_values(
                [float(loop["total_loop_time_s"]) for loop in history if loop.get("total_loop_time_s") is not None]
            ),
            "stt_elapsed_s": cls._summary_from_values(
                [
                    float(loop.get("stt", {}).get("elapsed_s"))
                    for loop in history
                    if loop.get("stt", {}).get("elapsed_s") is not None
                ]
            ),
            "llm_elapsed_s": cls._summary_from_values(
                [
                    float(loop.get("llm", {}).get("elapsed_s"))
                    for loop in history
                    if loop.get("llm", {}).get("elapsed_s") is not None
                ]
            ),
            "tts_elapsed_s": cls._summary_from_values(
                [
                    float(loop.get("tts", {}).get("elapsed_s"))
                    for loop in history
                    if loop.get("tts", {}).get("elapsed_s") is not None
                ]
            ),
        }

    @staticmethod
    def _fmt_value(value: float | int | None, suffix: str = "") -> str:
        if value is None:
            return "n/a"
        if isinstance(value, int):
            return f"{value}{suffix}"
        return f"{value:.3f}{suffix}"

    def _fmt_resource_line(self, label: str, resource: dict[str, dict[str, float | None]]) -> str:
        return (
            f"{label}: "
            f"CPU {self._fmt_value(resource['cpu_percent']['average'], '%')} "
            f"(min {self._fmt_value(resource['cpu_percent']['minimum'], '%')}, max {self._fmt_value(resource['cpu_percent']['maximum'], '%')}), "
            f"RAM {self._fmt_value(resource['ram_percent']['average'], '%')} "
            f"(min {self._fmt_value(resource['ram_percent']['minimum'], '%')}, max {self._fmt_value(resource['ram_percent']['maximum'], '%')}), "
            f"GPU {self._fmt_value(resource['gpu_percent']['average'], '%')} "
            f"(min {self._fmt_value(resource['gpu_percent']['minimum'], '%')}, max {self._fmt_value(resource['gpu_percent']['maximum'], '%')}), "
            f"VRAM {self._fmt_value(resource['vram_percent']['average'], '%')} "
            f"(min {self._fmt_value(resource['vram_percent']['minimum'], '%')}, max {self._fmt_value(resource['vram_percent']['maximum'], '%')}), "
            f"VRAM bytes {self._fmt_value(resource['vram_bytes']['average'])} "
            f"(min {self._fmt_value(resource['vram_bytes']['minimum'])}, max {self._fmt_value(resource['vram_bytes']['maximum'])})"
        )

    def to_human_readable(self) -> str:
        stt_resources = self._resource_block(self.stt)
        llm_resources = self._resource_block(self.llm)
        tts_resources = self._resource_block(self.tts)
        parts = [
            f"Loop #{self.loop_index}",
            f"  listening_time_s={self._fmt_value(self.listening_time_s, 's')}, "
            f"audio_sent_to_stt_at_s={self._fmt_value(self.audio_sent_to_stt_at_s, 's')}",
            (
                f"  STT: model={self.stt.model_name or 'n/a'}, bytes={self.stt.bytes}, "
                f"elapsed={self._fmt_value(self.stt.elapsed_s, 's')}, "
                f"{self._fmt_resource_line('', stt_resources).lstrip(': ')}"
            ),
            (
                f"  LLM: model={self.llm.model_name or 'n/a'}, status={self.llm.status or 'n/a'}, "
                f"first_sentence_latency={self._fmt_value(self.llm.first_sentence_latency_s, 's')}, "
                f"elapsed={self._fmt_value(self.llm.elapsed_s, 's')}, "
                f"{self._fmt_resource_line('', llm_resources).lstrip(': ')}"
            ),
            (
                f"  TTS: model={self.tts.model_name or 'n/a'}, "
                f"elapsed={self._fmt_value(self.tts.elapsed_s, 's')}, "
                f"{self._fmt_resource_line('', tts_resources).lstrip(': ')}"
            ),
            f"  total_loop_time_s={self._fmt_value(self.total_loop_time_s, 's')}",
        ]
        return "\n".join(parts)


class ConnState(BaseModel):
    """Per-connection mutable state, including all protocol-level IDs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str = Field(default_factory=lambda: _generate_id("session"))
    conversation_id: str = Field(default_factory=lambda: _generate_id("conv"))
    runtime_config: RuntimeConfig = Field(default_factory=RuntimeConfig)
    in_response: bool = False
    response_pending: bool = False
    audio_buffer_has_data: bool = False
    audio_remainder: bytes = b""
    current_response_id: Optional[str] = None
    current_item_id: Optional[str] = None
    content_index: int = 0
    input_content_index: int = 0
    input_audio_duration_s: float = 0.0
    last_item_id: Optional[str] = None
    current_response_params: RealtimeResponseCreateParams | None = None
    pending_output_text_parts: list[str] = Field(default_factory=list)
    response_usage: UsageMetrics = Field(default_factory=UsageMetrics)
    speculative_turn_id: Optional[str] = None
    speculative_turn_revision: Optional[int] = None
    speculative_user_turn_id: Optional[str] = None
    speculative_user_turn_revision: Optional[int] = None
    speculative_user_speech_stopped_at_s: Optional[float] = None
    speculative_user_item_id: Optional[str] = None
    speculative_input_item_id: Optional[str] = None
    speculative_audio_duration_s: float = 0.0
    current_phase: str = "idle"
    current_phase_note: str = ""
    phase_started_at_s: float = Field(default_factory=time.perf_counter)
    phase_timings_s: dict[str, float] = Field(default_factory=dict)
    phase_history: list[dict[str, Any]] = Field(default_factory=list)
    loop_counter: int = 0
    current_loop: LoopRecord | None = None
    loop_history: list[dict[str, Any]] = Field(default_factory=list)
    llm_model_name: str | None = None
    stt_model_name: str | None = None
    tts_model_name: str | None = None
    # Client conversation.item.create items that arrived while a response was
    # generating. Applying them mid-generation races the LLM handler's chat
    # write-back (cross-thread), so they are buffered here and flushed in order
    # once the response completes. See ConversationHandler.flush_deferred_items.
    deferred_items: list[ConversationItem] = Field(default_factory=list)


class RealtimeService:
    """Translates between OpenAI Realtime protocol events and internal pipeline messages.

    One instance is shared across all WebSocket connections.  Per-connection
    state (response lifecycle, audio buffer) is tracked internally by
    connection id.
    """

    def __init__(
        self,
        text_prompt_queue: Queue[TextPromptItem] | None = None,
        should_listen: ThreadingEvent | None = None,
        chat_size: int = 10,
        speculative_turns: SpeculativeTurnTracker | None = None,
        log_dir: str | Path | None = None,
    ) -> None:
        self.text_prompt_queue = text_prompt_queue
        self.should_listen = should_listen
        self._chat_size = chat_size
        self.speculative_turns = speculative_turns
        self._conns: dict[str, ConnState] = {}
        self.total_usage = GlobalUsageMetrics()

        self.audio = AudioHandler(self)
        self.session = SessionHandler(self)
        self.response = ResponseHandler(self)
        self.conversation = ConversationHandler(self)
        self.log_dir = Path(log_dir) if log_dir is not None else self._default_log_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.loop_csv_path = self.log_dir / LOOP_LOG_CSV_NAME
        self.loop_md_path = self.log_dir / LOOP_LOG_MD_NAME

        self._pipeline_dispatch: dict[type[PipelineEvent], Callable[..., list[ServerEvent]]] = {
            SpeechStartedEvent: self.audio.on_speech_started,
            SpeechStoppedEvent: self.audio.on_speech_stopped,
            TokenUsageEvent: self._on_token_usage,
            PartialTranscriptionEvent: self.conversation.on_partial_transcription,
            TranscriptionCompletedEvent: self._on_transcription_completed,
            ResponseFailedEvent: self._on_response_failed,
        }

    @staticmethod
    def _default_log_dir() -> Path:
        here = Path(__file__).resolve()
        for candidate in here.parents:
            if (candidate / "pyproject.toml").exists():
                return candidate / "logs"
        return here.parents[5] / "logs"

    # ── Connection lifecycle ─────────────────────

    def register(self) -> str:
        """Register a new connection and return its session_id."""
        if self.speculative_turns:
            self.speculative_turns.reset()
        state = ConnState(runtime_config=RuntimeConfig(chat=Chat(self._chat_size)))
        state.phase_started_at_s = time.perf_counter()
        self._conns[state.session_id] = state
        self.total_usage.connections += 1
        return state.session_id

    def register_pipeline_models(
        self,
        conn_id: str,
        *,
        llm_model_name: str | None = None,
        stt_model_name: str | None = None,
        tts_model_name: str | None = None,
    ) -> None:
        """Cache the concrete STT/TTS model names used by this connection."""
        st = self._state(conn_id)
        if llm_model_name:
            st.llm_model_name = llm_model_name
        if stt_model_name:
            st.stt_model_name = stt_model_name
        if tts_model_name:
            st.tts_model_name = tts_model_name

    def unregister(self, conn_id: str) -> None:
        st = self._conns.pop(conn_id, None)
        if st is not None:
            # Suppress any in-flight compaction splice so a daemon worker can't
            # mutate a Chat tied to a closed session, and don't make further
            # billable LLM calls on its behalf once the splice is suppressed.
            st.runtime_config.chat.close()
            self.total_usage += st.response_usage
            logger.info(
                "Session %s unregistered — cumulative: input_tokens=%d, output_tokens=%d, audio=%.2fs",
                conn_id,
                self.total_usage.input_tokens,
                self.total_usage.output_tokens,
                self.total_usage.audio_duration_s,
            )

    def _state(self, conn_id: str) -> ConnState:
        return self._conns[conn_id]

    def _set_phase(self, conn_id: str, phase: str, note: str = "") -> None:
        st = self._state(conn_id)
        now = time.perf_counter()
        if st.current_phase == phase and st.current_phase_note == note:
            return
        elapsed = max(0.0, now - st.phase_started_at_s)
        if st.current_phase:
            st.phase_timings_s[st.current_phase] = st.phase_timings_s.get(st.current_phase, 0.0) + elapsed
            st.phase_history.append(
                {
                    "phase": st.current_phase,
                    "note": st.current_phase_note,
                    "elapsed_s": round(elapsed, 3),
                }
            )
        st.current_phase = phase
        st.current_phase_note = note
        st.phase_started_at_s = now

    @staticmethod
    def _format_model_name(model_name: Any) -> str | None:
        if model_name is None:
            return None
        value = str(model_name).strip()
        return value or None

    def _resource_snapshot(self) -> ResourceSample:
        cpu_percent: float | None = None
        ram_percent: float | None = None
        gpu_percent: float | None = None
        vram_percent: float | None = None
        vram_bytes: int | None = None

        if psutil is not None and _PSUTIL_PROCESS is not None:
            try:
                cpu_percent = float(_PSUTIL_PROCESS.cpu_percent(None))
                ram_percent = float(psutil.virtual_memory().percent)
            except Exception:
                pass

        try:
            import torch

            if torch.cuda.is_available():
                try:
                    query = [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ]
                    result = subprocess.run(query, capture_output=True, text=True, timeout=1.0, check=False)
                    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
                    if line:
                        parts = [part.strip() for part in line.split(",")]
                        if len(parts) >= 3:
                            gpu_percent = float(parts[0])
                            used_mib = float(parts[1])
                            total_mib = float(parts[2])
                            if total_mib > 0:
                                vram_percent = (used_mib / total_mib) * 100.0
                            vram_bytes = int(used_mib * 1024 * 1024)
                except Exception:
                    try:
                        device_index = torch.cuda.current_device()
                        props = torch.cuda.get_device_properties(device_index)
                        total_vram = float(props.total_memory)
                        used_vram = float(torch.cuda.memory_allocated(device_index))
                        vram_bytes = int(used_vram)
                        if total_vram > 0:
                            vram_percent = (used_vram / total_vram) * 100.0
                    except Exception:
                        pass
        except Exception:
            pass

        return ResourceSample(
            cpu_percent=cpu_percent,
            ram_percent=ram_percent,
            gpu_percent=gpu_percent,
            vram_percent=vram_percent,
            vram_bytes=vram_bytes,
        )

    def _current_loop(self, conn_id: str) -> LoopRecord | None:
        return self._state(conn_id).current_loop

    def _ensure_loop(self, conn_id: str) -> LoopRecord | None:
        st = self._state(conn_id)
        if st.current_loop is None:
            return None
        return st.current_loop

    def _capture_loop_sample(self, conn_id: str, stage: Literal["stt", "llm", "tts"]) -> None:
        loop = self._current_loop(conn_id)
        if loop is None:
            return
        getattr(loop, stage).samples.append(self._resource_snapshot())

    def start_loop(self, conn_id: str, *, speech_started_at_s: float | None = None) -> None:
        st = self._state(conn_id)
        st.loop_counter += 1
        st.current_loop = LoopRecord(
            loop_index=st.loop_counter,
            speech_started_at_s=speech_started_at_s if speech_started_at_s is not None else time.perf_counter(),
        )
        self._capture_loop_sample(conn_id, "stt")

    def mark_audio_sent_to_stt(self, conn_id: str) -> None:
        loop = self._ensure_loop(conn_id)
        if loop is None or loop.speech_started_at_s is None:
            return
        now = time.perf_counter()
        loop.audio_sent_to_stt_at_s = now
        loop.listening_time_s = max(0.0, now - loop.speech_started_at_s)
        loop.stt.started_at_s = now
        self._capture_loop_sample(conn_id, "stt")

    def add_stt_bytes(self, conn_id: str, byte_count: int) -> None:
        loop = self._ensure_loop(conn_id)
        if loop is None:
            return
        loop.stt.bytes += max(0, int(byte_count))

    def finish_stt(self, conn_id: str) -> None:
        loop = self._ensure_loop(conn_id)
        if loop is None:
            return
        now = time.perf_counter()
        loop.stt.finished_at_s = now
        if loop.stt.started_at_s is not None:
            loop.stt.elapsed_s = max(0.0, now - loop.stt.started_at_s)
        self._capture_loop_sample(conn_id, "stt")

    def start_llm(self, conn_id: str, model_name: Any | None = None) -> None:
        loop = self._ensure_loop(conn_id)
        if loop is None:
            return
        st = self._state(conn_id)
        resolved_model_name = self._format_model_name(model_name) or st.llm_model_name
        if loop.llm.started_at_s is not None:
            if loop.llm.model_name is None:
                loop.llm.model_name = resolved_model_name
            return
        now = time.perf_counter()
        loop.llm.started_at_s = now
        loop.llm.model_name = resolved_model_name
        self._capture_loop_sample(conn_id, "llm")

    def mark_llm_first_sentence(self, conn_id: str) -> None:
        loop = self._ensure_loop(conn_id)
        if loop is None:
            return
        now = time.perf_counter()
        if loop.llm.started_at_s is not None and loop.llm.first_sentence_latency_s is None:
            loop.llm.first_sentence_latency_s = max(0.0, now - loop.llm.started_at_s)
        self._capture_loop_sample(conn_id, "llm")

    def finish_llm(self, conn_id: str, status: str | None = None) -> None:
        loop = self._ensure_loop(conn_id)
        if loop is None:
            return
        now = time.perf_counter()
        loop.llm.finished_at_s = now
        if loop.llm.started_at_s is not None:
            loop.llm.elapsed_s = max(0.0, now - loop.llm.started_at_s)
        loop.llm.status = status
        if loop.llm.model_name is None:
            session_model = self._format_model_name(self._state(conn_id).runtime_config.session.model)
            loop.llm.model_name = session_model
        self._capture_loop_sample(conn_id, "llm")

    def start_tts(self, conn_id: str, model_name: Any | None = None) -> None:
        loop = self._ensure_loop(conn_id)
        if loop is None:
            return
        st = self._state(conn_id)
        resolved_model_name = self._format_model_name(model_name) or st.tts_model_name
        if loop.tts.started_at_s is not None:
            if loop.tts.model_name is None:
                loop.tts.model_name = resolved_model_name
            return
        now = time.perf_counter()
        loop.tts.started_at_s = now
        loop.tts.model_name = resolved_model_name
        self._capture_loop_sample(conn_id, "tts")

    def finish_tts(self, conn_id: str) -> None:
        loop = self._ensure_loop(conn_id)
        if loop is None:
            return
        now = time.perf_counter()
        loop.tts.finished_at_s = now
        if loop.tts.started_at_s is not None:
            loop.tts.elapsed_s = max(0.0, now - loop.tts.started_at_s)
        loop.tts_completed_at_s = now
        if loop.speech_started_at_s is not None:
            loop.total_loop_time_s = max(0.0, now - loop.speech_started_at_s)
        self._capture_loop_sample(conn_id, "tts")

    def complete_loop(self, conn_id: str) -> None:
        st = self._state(conn_id)
        loop = st.current_loop
        if loop is None:
            return
        if loop.tts_completed_at_s is None:
            return
        record = loop.to_dict()
        st.loop_history.append(record)
        if len(st.loop_history) > 20:
            st.loop_history = st.loop_history[-20:]
        self._append_loop_csv(record)
        self._append_loop_markdown(loop)
        logger.info("Loop summary for session %s:\n%s", conn_id, loop.to_human_readable())
        st.current_loop = None

    def _append_loop_csv(self, record: dict[str, Any]) -> None:
        def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    next_prefix = f"{prefix}{key}." if prefix else f"{key}."
                    _flatten(next_prefix, item, out)
            else:
                out[prefix[:-1]] = value

        flat: dict[str, Any] = {}
        _flatten("", record, flat)
        fieldnames = list(flat)
        file_exists = self.loop_csv_path.exists()
        with self.loop_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(flat)

    def _append_loop_markdown(self, loop: LoopRecord) -> None:
        section = loop.to_human_readable()
        heading = f"## Loop {loop.loop_index}"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        body = "\n".join(
            [
                heading,
                f"- Logged at: {timestamp}",
                "```text",
                section,
                "```",
                "",
            ]
        )
        need_separator = self.loop_md_path.exists() and self.loop_md_path.stat().st_size > 0
        with self.loop_md_path.open("a", encoding="utf-8") as f:
            if need_separator:
                f.write("\n")
            f.write(body)

    @property
    def connection_ids(self) -> list[str]:
        return list(self._conns)

    # ── Client event parsing ─────────────────────

    @staticmethod
    def _next_event_id() -> str:
        return _generate_id("event")

    def parse_client_event(self, raw: Mapping[str, object]) -> Optional[ClientEvent]:
        raw_type = raw.get("type")
        event_type: Optional[str] = raw_type if isinstance(raw_type, str) else None
        if event_type is None:
            logger.warning("Client event missing 'type' field")
            return None
        model_cls = _EVENT_TYPE_TO_MODEL.get(event_type)
        if model_cls is None:
            logger.warning(f"Unknown client event type: {event_type}")
            return None
        try:
            return model_cls.model_validate(raw)  # type: ignore[return-value]
        except ValidationError as e:
            logger.error(f"Invalid {event_type} payload: {e}")
            return None

    # ── Client event handlers ────────────────────

    def build_session_created(self, conn_id: str) -> SessionCreatedEvent:
        return self.session.build_session_created(conn_id)

    def handle_session_update(self, conn_id: str, event: SessionUpdateEvent) -> Optional[RealtimeErrorEvent]:
        return self.session.handle_session_update(conn_id, event)

    def handle_audio_append(self, conn_id: str, event: InputAudioBufferAppendEvent) -> list[bytes]:
        self._set_phase(conn_id, "audio_ingest", "receiving microphone audio")
        return self.audio.handle_audio_append(conn_id, event)

    def handle_audio_commit(self, conn_id: str) -> RealtimeErrorEvent | None:
        self._set_phase(conn_id, "stt", "audio committed, awaiting transcription")
        return self.audio.handle_audio_commit(conn_id)

    def encode_audio_chunk(self, conn_id: str, audio: bytes) -> list[ServerEvent]:
        return self.audio.encode_audio_chunk(conn_id, audio)

    def handle_response_create(self, conn_id: str, event: ResponseCreateEvent) -> ServerEvent | None:
        self._set_phase(conn_id, "llm", "creating response")
        return self.response.handle_response_create(conn_id, event)

    def handle_response_cancel(self, conn_id: str) -> list[ServerEvent]:
        return self.response.handle_response_cancel(conn_id)

    def finish_response(
        self,
        conn_id: str,
        status: _ResponseStatus = "completed",
        reason: _StatusReason | None = None,
    ) -> list[ServerEvent]:
        events = self.response.finish_response(conn_id, status, reason)
        self.complete_loop(conn_id)
        self._set_phase(conn_id, "idle", f"response {status}")
        return events

    def handle_conversation_item_create(self, conn_id: str, event: ConversationItemCreateEvent) -> list[ServerEvent]:
        return self.conversation.handle_conversation_item_create(conn_id, event)

    def dispatch_pipeline_event(self, conn_id: str, event: PipelineEvent) -> list[ServerEvent]:
        """Route a pipeline text_output_queue event to the appropriate handler."""
        events = self._dispatch_pipeline_event(conn_id, event, wait_for_pending_reopen=True)
        return [] if events is None else events

    def try_dispatch_pipeline_event(self, conn_id: str, event: PipelineEvent) -> list[ServerEvent] | None:
        """Non-blocking dispatch.

        Returns ``None`` when dispatch must be retried after a speculative
        reopen candidate resolves.
        """
        return self._dispatch_pipeline_event(conn_id, event, wait_for_pending_reopen=False)

    def should_defer_pipeline_event(self, event: PipelineEvent) -> bool:
        if self.speculative_turns is None or not isinstance(event, (AssistantTextEvent, TokenUsageEvent)):
            return False
        return self.speculative_turns.has_pending_reopen_or_grace(
            getattr(event, "turn_id", None),
            getattr(event, "turn_revision", None),
        )

    def _dispatch_pipeline_event(
        self,
        conn_id: str,
        event: PipelineEvent,
        *,
        wait_for_pending_reopen: bool,
    ) -> list[ServerEvent] | None:
        is_stale = self._is_stale_turn_event(event, wait_for_pending_reopen=wait_for_pending_reopen)
        if is_stale is None:
            return None
        if is_stale:
            logger.info(
                "Ignoring stale %s for turn=%s rev=%s",
                event.type,
                getattr(event, "turn_id", None),
                getattr(event, "turn_revision", None),
            )
            return []

        self._observe_turn_event(event)
        if isinstance(event, AssistantTextEvent):
            return self.response.on_assistant_text(
                conn_id,
                event,
                wait_for_pending_reopen=wait_for_pending_reopen,
            )
        handler = self._pipeline_dispatch.get(type(event))
        if handler is None:
            logger.debug("Unhandled pipeline event type: %s", type(event).__name__)
            return []
        return handler(conn_id, event)

    def _is_stale_turn_event(self, event: PipelineEvent, *, wait_for_pending_reopen: bool = True) -> bool | None:
        if self.speculative_turns is None:
            return False
        if not isinstance(
            event,
            (PartialTranscriptionEvent, TranscriptionCompletedEvent, AssistantTextEvent, TokenUsageEvent),
        ):
            return False
        turn_id = getattr(event, "turn_id", None)
        turn_revision = getattr(event, "turn_revision", None)
        if isinstance(event, (AssistantTextEvent, TokenUsageEvent)):
            is_latest: bool | None
            if wait_for_pending_reopen:
                is_latest = self.speculative_turns.is_latest_after_reopen_grace(turn_id, turn_revision)
            else:
                is_latest = self.speculative_turns.try_is_latest_after_reopen_grace(turn_id, turn_revision)
            if is_latest is None:
                return None
            return not is_latest
        return not self.speculative_turns.is_latest(turn_id, turn_revision)

    def _observe_turn_event(self, event: PipelineEvent) -> None:
        if self.speculative_turns is None:
            return
        self.speculative_turns.observe(
            getattr(event, "turn_id", None),
            getattr(event, "turn_revision", None),
        )

    # ── STT → LM bridge ────────────────────────────

    def _on_transcription_completed(self, conn_id: str, event: TranscriptionCompletedEvent) -> list[ServerEvent]:
        """Handle a final STT transcription: emit protocol event, append to chat, trigger LM."""
        self._set_phase(conn_id, "llm", "transcription completed, generating prompt")
        st = self._state(conn_id)
        self.finish_stt(conn_id)
        same_speculative_turn = event.turn_id is not None and event.turn_id == st.speculative_user_turn_id
        if same_speculative_turn:
            st.response_usage.audio_duration_s -= st.speculative_audio_duration_s
        else:
            st.speculative_audio_duration_s = 0.0

        events = self.conversation.on_transcription_completed(conn_id, event)
        if event.turn_id is not None:
            st.speculative_audio_duration_s = st.input_audio_duration_s

        cfg = st.runtime_config
        transcript = event.transcript
        if transcript:
            if same_speculative_turn and st.speculative_user_item_id:
                replaced = cfg.chat.replace_user_message_text(st.speculative_user_item_id, transcript)
                if not replaced:
                    item = cfg.chat.add_item(make_user_message(transcript))
                    st.speculative_user_item_id = item.id
            else:
                item = cfg.chat.add_item(make_user_message(transcript))
                st.speculative_user_item_id = item.id
        elif same_speculative_turn and st.speculative_user_item_id:
            cfg.chat.remove_user_message(st.speculative_user_item_id)
            st.speculative_user_item_id = None
        elif event.turn_id is not None and event.turn_id != st.speculative_user_turn_id:
            st.speculative_user_item_id = None

        if event.turn_id is not None:
            st.speculative_user_turn_id = event.turn_id
            st.speculative_user_turn_revision = event.turn_revision
            st.speculative_user_speech_stopped_at_s = event.speech_stopped_at_s

        queue = self.text_prompt_queue
        if queue and transcript:
            self.start_llm(conn_id, st.runtime_config.session.model)
            st.response_pending = True
            queue.put(
                GenerateResponseRequest(
                    runtime_config=cfg,
                    language_code=event.language_code,
                    turn_id=event.turn_id,
                    turn_revision=event.turn_revision,
                    speech_stopped_at_s=event.speech_stopped_at_s,
                )
            )

        return events

    # ── Metrics ────────────────────────────────────

    def _on_token_usage(self, conn_id: str, event: TokenUsageEvent) -> list[ServerEvent]:
        """Accumulate input/output token counts on the connection's usage metrics."""
        if self.speculative_turns and not self.speculative_turns.is_latest(
            event.turn_id,
            event.turn_revision,
        ):
            logger.debug("Dropping stale token usage for turn=%s rev=%s", event.turn_id, event.turn_revision)
            return []
        st = self._state(conn_id)
        st.response_usage.input_tokens += event.input_tokens
        st.response_usage.output_tokens += event.output_tokens
        logger.info(
            "Token usage (response): input=%d, output=%d",
            st.response_usage.input_tokens,
            st.response_usage.output_tokens,
        )
        return []

    def _on_response_failed(self, conn_id: str, event: ResponseFailedEvent) -> list[ServerEvent]:
        """Surface the failure to the client and close the response as ``failed``.

        Emitted when generation failed (e.g. invalid out-of-band input, or the
        provider rejecting an empty context). A top-level ``error`` event carries
        the human-readable reason — ``response.done.status_details.error`` only
        has code/type, no message — then ``finish_response`` closes the slot.

        Idempotent: gated on an active response, and ``finish_response`` is itself
        a no-op once the slot is closed, so a later EndOfResponse-driven close does
        nothing.
        """
        logger.info("Response failed: %s", event.message)
        if not self._state(conn_id).in_response:
            return []
        self._set_phase(conn_id, "idle", "response failed")
        events: list[ServerEvent] = [self.make_error(event.message, "response_failed")]
        events.extend(self.response.finish_response(conn_id, status="failed"))
        return events

    def get_usage(self) -> dict[str, Any]:
        """Return cumulative usage metrics across all completed responses."""
        data = self.total_usage.model_dump()
        data["total_tokens"] = data["input_tokens"] + data["output_tokens"]
        data["total_errors"] = self.total_usage.total_errors
        return data

    def get_diagnostics(self) -> dict[str, Any]:
        now = time.perf_counter()
        connections: list[dict[str, Any]] = []
        for conn_id, st in self._conns.items():
            current_phase_elapsed = max(0.0, now - st.phase_started_at_s)
            phase_timings = dict(st.phase_timings_s)
            phase_timings[st.current_phase] = round(phase_timings.get(st.current_phase, 0.0) + current_phase_elapsed, 3)
            session = st.runtime_config.session
            model_name = getattr(session, "model", None)
            audio = getattr(session, "audio", None)
            audio_in = getattr(audio, "input", None) if audio is not None else None
            audio_out = getattr(audio, "output", None) if audio is not None else None
            current_loop = st.current_loop.to_live_dict(now) if st.current_loop is not None else None
            connections.append(
                {
                    "session_id": conn_id,
                    "phase": st.current_phase,
                    "phase_note": st.current_phase_note,
                    "phase_elapsed_s": round(current_phase_elapsed, 3),
                    "phase_timings_s": phase_timings,
                    "phase_history": st.phase_history[-20:],
                    "model_name": None if model_name is None else str(model_name),
                    "audio_input_format": None if getattr(audio_in, "format", None) is None else str(getattr(audio_in, "format", None)),
                    "audio_output_format": None if getattr(audio_out, "format", None) is None else str(getattr(audio_out, "format", None)),
                    "in_response": st.in_response,
                    "response_pending": st.response_pending,
                    "input_audio_duration_s": round(st.input_audio_duration_s, 3),
                    "current_listening_time_s": (
                        round(max(0.0, now - st.current_loop.speech_started_at_s), 3)
                        if st.current_loop is not None and st.current_loop.speech_started_at_s is not None
                        else None
                    ),
                    "current_loop": current_loop,
                    "loop_summary": LoopRecord.summarize_loop_history(st.loop_history),
                    "last_completed_loop": st.loop_history[-1] if st.loop_history else None,
                    "turns": st.response_usage.turns,
                    "llm_model_name": st.llm_model_name,
                    "stt_model_name": st.stt_model_name,
                    "tts_model_name": st.tts_model_name,
                }
            )
        return {
            "usage": self.get_usage(),
            "active_connections": len(self._conns),
            "connections": connections,
            "should_listen": self.should_listen.is_set() if self.should_listen is not None else None,
        }

    # ── Error ───────────────────────────────────

    def make_error(self, message: str, _type: str) -> RealtimeErrorEvent:
        self.total_usage.record_error(_type)
        return build_error_event(message, _type)


def build_error_event(message: str, error_type: str) -> RealtimeErrorEvent:
    """Construct a RealtimeErrorEvent without touching any service-instance state.

    Used by the websocket route handler on pool rejection, where no unit's
    service should be charged with the error in its usage metrics.
    """
    return RealtimeErrorEvent(
        type="error",
        error=RealtimeError(message=message, type=error_type),
        event_id=_generate_id("event"),
    )
