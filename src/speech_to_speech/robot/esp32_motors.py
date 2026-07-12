from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import requests


def _default_esp32_base_url() -> str:
    host = os.getenv("ESP32_BASE_URL") or os.getenv("ESP32_IP") or "192.168.137.208"
    if host.startswith("http://") or host.startswith("https://"):
        return host.rstrip("/")
    return f"http://{host}".rstrip("/")


@dataclass(frozen=True)
class ESP32MotorCommand:
    name: str
    endpoint: str
    description: str


MOTOR_COMMANDS: tuple[ESP32MotorCommand, ...] = (
    ESP32MotorCommand(
        name="move_forward",
        endpoint="/forward",
        description="Drive the ESP32 robot forward.",
    ),
    ESP32MotorCommand(
        name="move_backward",
        endpoint="/backward",
        description="Drive the ESP32 robot backward.",
    ),
    ESP32MotorCommand(
        name="stop_robot",
        endpoint="/stop",
        description="Stop all robot movement immediately.",
    ),
)


def build_motor_tools() -> list[dict[str, Any]]:
    """Return realtime function-tool definitions for the ESP32 motor commands."""
    return [
        {
            "type": "function",
            "name": command.name,
            "description": command.description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        }
        for command in MOTOR_COMMANDS
    ]


class ESP32MotorController:
    """Tiny HTTP client for the ESP32 motor endpoints."""

    def __init__(self, base_url: str | None = None, timeout_s: float = 5.0) -> None:
        self.base_url = (base_url or _default_esp32_base_url()).rstrip("/")
        self.timeout_s = timeout_s

    def _call(self, endpoint: str) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, timeout=self.timeout_s)
        response.raise_for_status()
        return {
            "ok": True,
            "url": url,
            "status_code": response.status_code,
            "body": response.text,
        }

    def execute(self, tool_name: str) -> dict[str, Any]:
        command = next((cmd for cmd in MOTOR_COMMANDS if cmd.name == tool_name), None)
        if command is None:
            return {
                "ok": False,
                "error": f"Unsupported ESP32 motor command: {tool_name}",
            }
        try:
            return self._call(command.endpoint)
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "url": f"{self.base_url}{command.endpoint}",
                "error": str(exc),
            }

