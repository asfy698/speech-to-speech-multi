#rasp_client
from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
TEAMWORK = WORKSPACE / "teamwork_s2s"
EMOTION_PHASE_SCRIPT = TEAMWORK / "camera_to_gemma4.py"


def _launch_bot_face_gui() -> subprocess.Popen | None:
    bot_face = TEAMWORK / "bot_face_gui.py"
    if not bot_face.exists():
        print(f"Bot face GUI not found: {bot_face}", flush=True)
        return None
    print("\n=== Bot Face GUI ===", flush=True)
    return subprocess.Popen([sys.executable, str(bot_face)], cwd=TEAMWORK)


def _launch_status_window() -> subprocess.Popen | None:
    status_window = TEAMWORK / "status_window.py"
    if not status_window.exists():
        print(f"Status window not found: {status_window}", flush=True)
        return None
    print("\n=== Status Window ===", flush=True)
    return subprocess.Popen([sys.executable, str(status_window)], cwd=TEAMWORK)


def _run_emotion_phase() -> None:
    if not EMOTION_PHASE_SCRIPT.exists():
        print(f"Emotion phase script not found: {EMOTION_PHASE_SCRIPT}", flush=True)
        return

    print("\n=== Emotion Conversation Phase ===", flush=True)
    completed = subprocess.run([sys.executable, str(EMOTION_PHASE_SCRIPT)], cwd=TEAMWORK)
    if completed.returncode != 0:
        raise SystemExit(f"Emotion conversation phase failed with exit code {completed.returncode}")


@dataclass
class PiClientArguments:
    host: str = field(default_factory=lambda: os.getenv("S2S_SERVER_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("S2S_SERVER_PORT", "8765")))
    websocket_base_url: Optional[str] = field(default=None)
    gateway_token: Optional[str] = field(default=os.getenv("S2S_GATEWAY_TOKEN"))
    send_rate: int = field(default=16000)
    recv_rate: int = field(default=16000)
    chunk_size: int = field(default=1024)
    input_device: Optional[int] = field(default=None)
    output_device: Optional[int] = field(default=None)
    print_json: bool = field(default=False)
    launch_bot_face: bool = field(default=True)
    launch_status_window: bool = field(default=True)


def _append_query_param(url: str, key: str, value: str) -> str:
    parsed = urlparse(url)
    query_items = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query_items[key] = value
    return urlunparse(parsed._replace(query=urlencode(query_items)))


def _make_websocket_url(args: PiClientArguments) -> str:
    url = args.websocket_base_url or f"ws://{args.host}:{args.port}/v1/realtime"
    if args.gateway_token:
        url = _append_query_param(url, "token", args.gateway_token)
    return url


def _audio_bytes_from_ws_event(event: dict[str, Any]) -> bytes:
    return base64.b64decode(event.get("delta", ""))


def handle_face_state(_state: dict[str, Any]) -> None:
    # Intentionally lightweight: the bot face is shown by bot_face_gui.py.
    pass


async def run_client(args: PiClientArguments) -> None:
    import sounddevice as sd
    import websockets

    mic_queue: Queue[bytes] = Queue(maxsize=128)
    stop_event = Event()
    playback_buffer = bytearray()
    playback_lock = Lock()

    def clear_playback_buffer() -> None:
        with playback_lock:
            playback_buffer.clear()

    def callback_recv(outdata, _frames, _time_info, status):
        if status:
            print(f"Speaker status: {status}", flush=True)

        needed = len(outdata)
        with playback_lock:
            available = min(needed, len(playback_buffer))
            if available:
                outdata[:available] = playback_buffer[:available]
                del playback_buffer[:available]
            if available < needed:
                outdata[available:] = b"\x00" * (needed - available)

    def callback_send(indata, _frames, _time_info, status):
        if status:
            print(f"Mic status: {status}", flush=True)
        try:
            mic_queue.put_nowait(bytes(indata))
        except Exception:
            pass

    async def send_audio(ws) -> None:
        while not stop_event.is_set():
            try:
                chunk = await asyncio.to_thread(mic_queue.get, True, 0.1)
            except Empty:
                continue

            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            )

    async def receive_events(ws) -> None:
        while not stop_event.is_set():
            raw = await ws.recv()
            if isinstance(raw, bytes):
                with playback_lock:
                    playback_buffer.extend(raw)
                continue

            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                print(f"EVENT: {raw}", flush=True)
                continue

            if args.print_json:
                print(f"EVENT: {event}", flush=True)

            event_type = event.get("type")
            if event_type == "session.created":
                print("Connected.", flush=True)
            elif event_type == "input_audio_buffer.speech_started":
                clear_playback_buffer()
            elif event_type == "conversation.item.input_audio_transcription.delta":
                delta = str(event.get("delta", "")).strip()
                if delta:
                    print(f"USER: {delta}", flush=True)
            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = str(event.get("transcript", "")).strip()
                if transcript:
                    print(f"USER: {transcript}", flush=True)
            elif event_type == "response.created":
                print("ASSISTANT: <response started>", flush=True)
            elif event_type == "response.output_audio.delta":
                audio = _audio_bytes_from_ws_event(event)
                with playback_lock:
                    playback_buffer.extend(audio)
            elif event_type == "response.output_audio.done":
                print("ASSISTANT: <audio done>", flush=True)
            elif event_type == "response.output_audio_transcript.done":
                print(f"ASSISTANT: {event.get('transcript', '')}", flush=True)
            elif event_type in {"ui.face_state", "ui.animation"}:
                handle_face_state(event)
            elif event_type == "response.done":
                if event.get("response", {}).get("status") == "cancelled":
                    clear_playback_buffer()
                print(f"ASSISTANT: <response {event.get('response', {}).get('status', 'done')}>", flush=True)
            elif event_type == "error":
                error = event.get("error", {})
                print(f"ERROR: {error.get('type', 'error')}: {error.get('message', '')}", flush=True)
            else:
                print(f"EVENT: {event_type}", flush=True)

    async def wait_for_stop() -> None:
        await asyncio.to_thread(input, "Press Enter to stop...\n")
        stop_event.set()

    input_stream = sd.RawInputStream(
        samplerate=args.send_rate,
        channels=1,
        dtype="int16",
        blocksize=args.chunk_size,
        callback=callback_send,
        device=args.input_device,
    )
    output_stream = sd.RawOutputStream(
        samplerate=args.recv_rate,
        channels=1,
        dtype="int16",
        blocksize=args.chunk_size,
        callback=callback_recv,
        device=args.output_device,
    )

    input_stream.start()
    output_stream.start()

    try:
        async with websockets.connect(_make_websocket_url(args), max_size=None) as ws:
            tasks = [
                asyncio.create_task(send_audio(ws)),
                asyncio.create_task(receive_events(ws)),
                asyncio.create_task(wait_for_stop()),
            ]

            done, pending = await asyncio.wait(set(tasks), return_when=asyncio.FIRST_COMPLETED)

            stop_event.set()
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            for task in done:
                exc = task.exception()
                if exc is not None:
                    raise exc
    finally:
        stop_event.set()
        input_stream.stop()
        output_stream.stop()
        input_stream.close()
        output_stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Raspberry Pi client for the speech-to-speech server.")
    defaults = PiClientArguments()
    parser.add_argument("--host", default=defaults.host, help="Laptop/server IP address on the network.")
    parser.add_argument("--port", type=int, default=defaults.port)
    parser.add_argument("--websocket-base-url", default=defaults.websocket_base_url)
    parser.add_argument("--gateway-token", default=defaults.gateway_token)
    parser.add_argument("--send-rate", type=int, default=defaults.send_rate)
    parser.add_argument("--recv-rate", type=int, default=defaults.recv_rate)
    parser.add_argument("--chunk-size", type=int, default=defaults.chunk_size)
    parser.add_argument("--input-device", type=int, default=defaults.input_device)
    parser.add_argument("--output-device", type=int, default=defaults.output_device)
    parser.add_argument("--print-json", action="store_true", default=defaults.print_json)
    parser.add_argument("--no-bot-face", dest="launch_bot_face", action="store_false")
    parser.add_argument("--no-status-window", dest="launch_status_window", action="store_false")
    parser.set_defaults(launch_bot_face=defaults.launch_bot_face)
    parser.set_defaults(launch_status_window=defaults.launch_status_window)
    namespace = parser.parse_args()
    args = PiClientArguments(
        host=namespace.host,
        port=namespace.port,
        websocket_base_url=namespace.websocket_base_url,
        gateway_token=namespace.gateway_token,
        send_rate=namespace.send_rate,
        recv_rate=namespace.recv_rate,
        chunk_size=namespace.chunk_size,
        input_device=namespace.input_device,
        output_device=namespace.output_device,
        print_json=namespace.print_json,
        launch_bot_face=namespace.launch_bot_face,
        launch_status_window=namespace.launch_status_window,
    )
    bot_face_proc: subprocess.Popen | None = None
    status_window_proc: subprocess.Popen | None = None
    try:
        _run_emotion_phase()
        if args.launch_bot_face:
            bot_face_proc = _launch_bot_face_gui()
        if args.launch_status_window:
            status_window_proc = _launch_status_window()

        asyncio.run(run_client(args))
    except KeyboardInterrupt:
        pass
    finally:
        if bot_face_proc is not None and bot_face_proc.poll() is None:
            bot_face_proc.terminate()
            with contextlib.suppress(Exception):
                bot_face_proc.wait(timeout=5)
        if status_window_proc is not None and status_window_proc.poll() is None:
            status_window_proc.terminate()
            with contextlib.suppress(Exception):
                status_window_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
