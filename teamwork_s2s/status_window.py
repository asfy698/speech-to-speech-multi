from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import tkinter as tk
from tkinter import ttk

APP_TITLE = "Speech to Speech Status"
DEFAULT_STATS_URL = os.environ.get("S2S_REALTIME_STATS_URL", "http://127.0.0.1:8765/v1/stats")
POLL_INTERVAL_MS = 1000


@dataclass
class LoopView:
    index: int | None
    listening_time_s: float | None
    total_loop_time_s: float | None
    stt: dict[str, Any] | None
    llm: dict[str, Any] | None
    tts: dict[str, Any] | None


def _fetch_stats(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=2) as response:
        return json.loads(response.read().decode("utf-8"))


def _fmt_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _metric_line(data: dict[str, Any] | None, key: str) -> str:
    if not data:
        return f"{key}: n/a"
    resource = data.get("resources", {})
    return (
        f"{key}: model={data.get('model_name') or 'n/a'} | "
        f"status={data.get('status') or 'n/a'} | "
        f"elapsed={_fmt_number(data.get('elapsed_s'))}s | "
        f"avg_cpu={_fmt_number(resource.get('cpu_percent', {}).get('average'))}% | "
        f"min_cpu={_fmt_number(resource.get('cpu_percent', {}).get('minimum'))}% | "
        f"max_cpu={_fmt_number(resource.get('cpu_percent', {}).get('maximum'))}% | "
        f"avg_ram={_fmt_number(resource.get('ram_percent', {}).get('average'))}% | "
        f"min_ram={_fmt_number(resource.get('ram_percent', {}).get('minimum'))}% | "
        f"max_ram={_fmt_number(resource.get('ram_percent', {}).get('maximum'))}% | "
        f"avg_gpu={_fmt_number(resource.get('gpu_percent', {}).get('average'))}% | "
        f"avg_vram={_fmt_number(resource.get('vram_percent', {}).get('average'))}% | "
        f"avg_vram_bytes={_fmt_number(resource.get('vram_bytes', {}).get('average'), 0)}"
    )


def _loop_view(connection: dict[str, Any]) -> LoopView:
    loop = connection.get("current_loop") or connection.get("last_completed_loop") or {}
    return LoopView(
        index=loop.get("loop_index"),
        listening_time_s=loop.get("listening_time_s"),
        total_loop_time_s=loop.get("total_loop_time_s"),
        stt=loop.get("stt"),
        llm=loop.get("llm"),
        tts=loop.get("tts"),
    )


class StatusWindow:
    def __init__(self, stats_url: str) -> None:
        self.stats_url = stats_url
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("980x760")
        self.root.configure(bg="#0b1020")
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Title.TLabel", background="#0b1020", foreground="#edf3ff", font=("Segoe UI", 20, "bold"))
        style.configure("Body.TLabel", background="#0b1020", foreground="#d7e0f4", font=("Segoe UI", 10))
        style.configure("Section.TLabel", background="#0b1020", foreground="#8ddcff", font=("Consolas", 11, "bold"))
        style.configure("Status.TLabel", background="#0b1020", foreground="#d7e0f4", font=("Consolas", 10))

        header = ttk.Frame(self.root, padding=16)
        header.pack(fill="x")
        ttk.Label(header, text="Speech to Speech Status", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text=f"Polling server: {self.stats_url}",
            style="Body.TLabel",
        ).pack(anchor="w", pady=(6, 0))

        self.summary = ttk.Label(self.root, text="Connecting to server...", style="Status.TLabel", padding=(16, 0))
        self.summary.pack(fill="x", pady=(0, 8))

        body = ttk.Frame(self.root, padding=(16, 0, 16, 16))
        body.pack(fill="both", expand=True)

        self.text = tk.Text(
            body,
            wrap="word",
            bg="#11182c",
            fg="#edf3ff",
            insertbackground="#edf3ff",
            relief="flat",
            padx=14,
            pady=14,
            font=("Consolas", 10),
        )
        self.text.pack(fill="both", expand=True)
        self.text.configure(state="disabled")

        self.root.after(POLL_INTERVAL_MS, self._refresh)

    def _close(self) -> None:
        self.root.destroy()

    def _render(self, payload: dict[str, Any]) -> None:
        connection = (payload.get("connections") or [{}])[0] if payload.get("connections") else {}
        loop = _loop_view(connection)
        listening_enabled = payload.get("should_listen")
        listening_state = "Listening" if listening_enabled else "Paused"
        current_phase = connection.get("phase") or "idle"
        current_phase_note = connection.get("phase_note") or ""
        pool_size = payload.get("pool_size")
        active = payload.get("active_connections")
        loop_summary = connection.get("loop_summary") or {}
        stt = connection.get("stt_model_name") or connection.get("model_name") or "n/a"
        llm = connection.get("llm_model_name") or "n/a"
        tts = connection.get("tts_model_name") or "n/a"

        lines = [
            f"Server: online",
            f"Listening state: {listening_state}",
            f"Seconds listened: {_fmt_number(loop.listening_time_s)}",
            f"Phase: {current_phase}{f' - {current_phase_note}' if current_phase_note else ''}",
            f"Active connections: {active}/{pool_size}",
            "",
            "STT",
            f"Model: {stt}",
            _metric_line(loop.stt, "STT stats"),
            "",
            "LLM",
            f"Model: {llm}",
            f"Status: {loop.llm.get('status') if loop.llm else 'n/a'}",
            f"First-sentence latency: {_fmt_number(loop.llm.get('first_sentence_latency_s') if loop.llm else None)}s",
            _metric_line(loop.llm, "LLM stats"),
            "",
            "TTS",
            f"Model: {tts}",
            _metric_line(loop.tts, "TTS stats"),
            "",
            "Total loop stats",
            f"Current loop index: {_fmt_number(loop.index, 0)}",
            f"Current loop time: {_fmt_number(loop.total_loop_time_s)}s",
            f"Completed loops: {_fmt_number(loop_summary.get('completed_loops'), 0)}",
            f"Average loop time: {_fmt_number(loop_summary.get('total_loop_time_s', {}).get('average'))}s",
            f"Minimum loop time: {_fmt_number(loop_summary.get('total_loop_time_s', {}).get('minimum'))}s",
            f"Maximum loop time: {_fmt_number(loop_summary.get('total_loop_time_s', {}).get('maximum'))}s",
            f"Average listening time: {_fmt_number(loop_summary.get('listening_time_s', {}).get('average'))}s",
            f"Average STT time: {_fmt_number(loop_summary.get('stt_elapsed_s', {}).get('average'))}s",
            f"Average LLM time: {_fmt_number(loop_summary.get('llm_elapsed_s', {}).get('average'))}s",
            f"Average TTS time: {_fmt_number(loop_summary.get('tts_elapsed_s', {}).get('average'))}s",
        ]

        text = "\n".join(lines)
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", text)
        self.text.configure(state="disabled")
        self.summary.configure(
            text=(
                f"Server online | listening={listening_state.lower()} | "
                f"phase={current_phase} | loop={_fmt_number(loop.index, 0)}"
            )
        )

    def _render_offline(self, message: str) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert(
            "1.0",
            "\n".join(
                [
                    "Server: offline",
                    f"Reason: {message}",
                    "",
                    "Waiting for the realtime server...",
                ]
            ),
        )
        self.text.configure(state="disabled")
        self.summary.configure(text=f"Server offline | {message}")

    def _refresh(self) -> None:
        try:
            payload = _fetch_stats(self.stats_url)
            self._render(payload)
        except (URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError) as exc:
            self._render_offline(str(exc))
        finally:
            if self.root.winfo_exists():
                self.root.after(POLL_INTERVAL_MS, self._refresh)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    StatusWindow(DEFAULT_STATS_URL).run()


if __name__ == "__main__":
    main()
