from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
TEAMWORK = WORKSPACE / "teamwork_s2s"
DEFAULT_CONTAINER_NAME = "speech-to-speech"
LISTEN_COMMAND = (
    "PULSE_SERVER=host.docker.internal:4713 "
    "python scripts/listen_and_play_realtime.py --host host.docker.internal --port 8765"
)


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str = "realtime"
    stt: str = "parakeet-tdt"
    llm_backend: str = "responses-api"
    tts: str = "qwen3"
    log_level: str = "info"
    device: str | None = None
    enable_live_transcription: bool = True
    num_pipelines: int = 1
    stt_device: str = "cuda"
    stt_torch_type: str = "float16"
    model_name: str = "ggml-org/gemma-4-E4B-it-GGUF"
    stream_batch_sentences: int = 3
    enable_lang_prompt: bool = False
    tts_lang: str = "auto"
    qwen3_tts_model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    qwen3_tts_device: str = "cuda"
    qwen3_tts_backend: str = "ggml"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "":
        return '""'
    if any(ch in text for ch in [":", "#", "{", "}", "[", "]", ",", "&", "*", "?", "|", "-", "<", ">", "=", "!", "%", "@", "`", "\n"]) or text.strip() != text:
        return repr(text)
    return text


def write_runtime_config() -> Path:
    run_dir = ROOT / "run"
    run_dir.mkdir(exist_ok=True)
    config_path = run_dir / "config.yaml"
    config = asdict(RuntimeConfig())
    lines = ["runtime:"]
    for key, value in config.items():
        lines.append(f"  {key}: {_yaml_scalar(value)}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def run_step(title: str, args: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"\n=== {title} ===", flush=True)
    completed = subprocess.run(args, cwd=str(ROOT), env=env)
    if completed.returncode != 0:
        raise SystemExit(f"{title} failed with exit code {completed.returncode}")


def start_background_step(
    title: str, args: list[str], *, env: dict[str, str] | None = None
) -> subprocess.Popen:
    print(f"\n=== {title} ===", flush=True)
    return subprocess.Popen(args, cwd=str(ROOT), env=env)


def run_realtime_in_container(container_name: str = DEFAULT_CONTAINER_NAME) -> None:
    exec_args = [
        "docker",
        "exec",
        "-it",
        container_name,
        "bash",
        "-lc",
        LISTEN_COMMAND,
    ]
    run_step("Listen And Play Realtime", exec_args)


def run_realtime_with_fallback() -> None:
    attempts = [DEFAULT_CONTAINER_NAME, "speech-to-speech-main-pipeline-1"]
    last_error: SystemExit | None = None

    for container_name in attempts:
        try:
            print(f"\n=== Trying container: {container_name} ===", flush=True)
            run_realtime_in_container(container_name)
            return
        except SystemExit as exc:
            last_error = exc
            print(f"=== Container {container_name} failed ===", flush=True)

    if last_error is not None:
        raise last_error


def main() -> int:
    python = sys.executable

    if not TEAMWORK.exists():
        raise SystemExit(f"Missing teamwork_s2s folder: {TEAMWORK}")

    config_path = write_runtime_config()
    print(f"\n=== Saved runtime config ===\n{config_path}", flush=True)

    bot_face_gui = TEAMWORK / "bot_face_gui.py"
    camera_to_gemma4 = TEAMWORK / "camera_to_gemma4.py"

    if not bot_face_gui.exists():
        raise SystemExit(f"Missing bot face GUI script: {bot_face_gui}")
    if not camera_to_gemma4.exists():
        raise SystemExit(f"Missing camera to gemma script: {camera_to_gemma4}")

    # 1) Bot face GUI stays open while the later steps run.
    gui_process = start_background_step(
        "Bot Face GUI",
        [python, str(bot_face_gui)],
    )

    # 2) Camera -> Gemma 4
    run_step(
        "Camera To Gemma 4",
        [python, str(camera_to_gemma4)],
    )

    # 3) Realtime listener/player in the dev container
    run_realtime_with_fallback()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
