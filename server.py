from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if src.exists():
        src_str = str(src)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


def main() -> None:
    _ensure_src_on_path()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--gateway-token",
        default=os.getenv("S2S_GATEWAY_TOKEN"),
        help="Optional token required by the realtime websocket gateway.",
    )
    args, remaining = parser.parse_known_args()

    if args.gateway_token:
        os.environ["S2S_GATEWAY_TOKEN"] = args.gateway_token

    sys.argv = [sys.argv[0], *remaining]

    from speech_to_speech.s2s_pipeline import main as pipeline_main

    pipeline_main()


if __name__ == "__main__":
    main()

